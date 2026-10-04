"""Learned-routing MLP controls with molecule-level or historical token routing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class _MLPExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, d_out: int,
                 activation: str = "mish", dropout: float = 0.1):
        super().__init__()
        act_map = {
            "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
            "mish": nn.Mish, "tanh": nn.Tanh,
        }
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_out)
        self.act = act_map.get(activation.lower(), nn.Mish)()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_2(self.dropout(self.act(self.W_1(x))))


class StandardMoEFFN(nn.Module):
    """TopK-gated MoE FFN with learnable router.

    ``routing_granularity='token'`` preserves the historical implementation.
    ``routing_granularity='molecule'`` pools valid node states, selects one
    branch per molecule, and broadcasts that assignment to all of its nodes.

    Attributes:
        num_experts: Number of experts.
        top_k: Number of experts activated per token or molecule.
        gate: Linear gating layer.
        experts: ModuleList of _MLPExpert.
        _aux_loss: Load-balancing loss from last forward pass.
    """

    def __init__(self,
                 d_model: int,
                 d_ff: int = 0,
                 activation: str = "mish",
                 dropout: float = 0.1,
                 d_out: Optional[int] = None,
                 num_experts: int = 8,
                 top_k: int = 2,
                 aux_loss_coeff: float = 0.01,
                 routing_granularity: str = "token"):
        super().__init__()
        if d_out is None:
            d_out = d_model
        if d_ff <= 0:
            d_ff = d_model

        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.d_out = d_out
        self.aux_loss_coeff = aux_loss_coeff
        if routing_granularity not in {"token", "molecule"}:
            raise ValueError(
                "routing_granularity must be 'token' or 'molecule'")
        if routing_granularity == "molecule" and self.top_k != 1:
            raise ValueError("Molecule-level routing currently requires top_k=1")
        self.routing_granularity = routing_granularity

        # Molecule routing uses a biased linear projection.
        # The bias-free token router is retained for old P6 checkpoints.
        self.gate = nn.Linear(
            d_model, num_experts, bias=(routing_granularity == "molecule"))

        # All experts are standard MLP (same architecture as PriorMoEFFN's _MLPExpert)
        self.experts = nn.ModuleList([
            _MLPExpert(d_model, d_ff, d_out, activation, dropout)
            for _ in range(num_experts)
        ])

        # Store aux loss for external access
        self._aux_loss: torch.Tensor = torch.tensor(0.0)
        self._last_top1_ids: Optional[torch.Tensor] = None
        self._last_gate_probs: Optional[torch.Tensor] = None

    @property
    def aux_loss(self) -> torch.Tensor:
        return self._aux_loss

    def forward(self,
                x: torch.Tensor,
                expert_ids: Optional[torch.Tensor] = None,
                node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass. expert_ids is IGNORED (accepted for API compatibility)."""
        if self.routing_granularity == "molecule":
            return self._forward_molecule(x, node_mask)
        return self._forward_token(x)

    def _set_aux_loss(self, gate_probs: torch.Tensor,
                      top1_idx: torch.Tensor) -> None:
        """Compute Switch-style balance loss over the routed entities."""
        self._last_top1_ids = top1_idx.detach()
        self._last_gate_probs = gate_probs.detach()
        if not self.training:
            self._aux_loss = gate_probs.new_tensor(0.0)
            return
        fractions = torch.stack([
            (top1_idx == expert).float().mean()
            for expert in range(self.num_experts)
        ])
        mean_probabilities = gate_probs.mean(dim=0)
        self._aux_loss = (
            self.aux_loss_coeff
            * self.num_experts
            * (fractions * mean_probabilities).sum()
        )

    def _forward_token(self, x: torch.Tensor) -> torch.Tensor:
        """Historical token-level routing retained for P6 reproducibility."""
        # Handle 3D input: (B, L, D) -> (B*L, D)
        reshaped = False
        if x.dim() == 3:
            B, L, D = x.shape
            x = x.reshape(B * L, D)
            reshaped = True

        N, D = x.shape

        # Gating: (N, num_experts)
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (N, E)

        # TopK selection
        topk_vals, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)  # (N, K)
        # Renormalize topk weights
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-12)  # (N, K)

        # Compute expert outputs (scatter-gather)
        out = x.new_zeros(N, self.d_out)
        for k in range(self.top_k):
            expert_indices = topk_idx[:, k]  # (N,)
            weights = topk_weights[:, k]     # (N,)
            for e in range(self.num_experts):
                mask = (expert_indices == e)
                if mask.any():
                    expert_out = self.experts[e](x[mask])  # (n_e, d_out)
                    out[mask] += weights[mask].unsqueeze(-1) * expert_out

        self._set_aux_loss(gate_probs, topk_idx[:, 0])

        if reshaped:
            out = out.view(B, L, self.d_out)

        return out

    def _forward_molecule(self, x: torch.Tensor,
                          node_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Route every molecule to one branch using masked mean node pooling."""
        if x.dim() != 3:
            raise ValueError(
                "Molecule-level routing requires input shaped (batch, nodes, hidden)")
        if node_mask is None:
            raise ValueError("Molecule-level routing requires node_mask")
        if node_mask.shape != x.shape[:2]:
            raise ValueError(
                f"node_mask shape {tuple(node_mask.shape)} does not match "
                f"input shape {tuple(x.shape[:2])}")

        mask = node_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        pooled = (x * mask).sum(dim=1) / counts
        gate_probs = F.softmax(self.gate(pooled), dim=-1)
        selected_prob, selected_idx = gate_probs.max(dim=-1)

        batch_size, _, _ = x.shape
        out = x.new_zeros(batch_size, x.size(1), self.d_out)
        for expert in range(self.num_experts):
            molecule_mask = selected_idx == expert
            if molecule_mask.any():
                expert_out = self.experts[expert](x[molecule_mask])
                out[molecule_mask] = (
                    selected_prob[molecule_mask, None, None] * expert_out)

        self._set_aux_loss(gate_probs, selected_idx)
        return out


__all__ = ["StandardMoEFFN"]
