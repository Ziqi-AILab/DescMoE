"""Standard MoE (learnable gating) for ablation comparison with PriorMoEFFN.

Implements a TopK-gated Mixture-of-Experts FFN that can be plugged into the same
position as PriorMoEFFN in the Graph Transformer's EncoderLayer.

Key differences from PriorMoEFFN:
  - Uses a learnable linear gating network (no chemical prior)
  - TopK routing with softmax gating weights
  - Load-balancing auxiliary loss to prevent routing collapse
"""

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

    Supports both 2D input (N, d_model) and 3D input (batch, seq_len, d_model).
    The gating network is a simple linear layer: gate_logits = W_gate @ x.
    TopK experts are selected per token, outputs are weighted-summed.

    Attributes:
        num_experts: Number of experts.
        top_k: Number of experts activated per token.
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
                 aux_loss_coeff: float = 0.01):
        super().__init__()
        if d_out is None:
            d_out = d_model
        if d_ff <= 0:
            d_ff = d_model

        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.d_out = d_out
        self.aux_loss_coeff = aux_loss_coeff

        # Learnable gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # All experts are standard MLP (same architecture as PriorMoEFFN's _MLPExpert)
        self.experts = nn.ModuleList([
            _MLPExpert(d_model, d_ff, d_out, activation, dropout)
            for _ in range(num_experts)
        ])

        # Store aux loss for external access
        self._aux_loss: torch.Tensor = torch.tensor(0.0)

    @property
    def aux_loss(self) -> torch.Tensor:
        return self._aux_loss

    def forward(self,
                x: torch.Tensor,
                expert_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass. expert_ids is IGNORED (accepted for API compatibility)."""
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

        # Load-balancing auxiliary loss (Switch Transformer style)
        # f_i = fraction of tokens routed to expert i
        # P_i = mean gate probability for expert i
        # loss = num_experts * sum(f_i * P_i)
        if self.training:
            # Use top-1 for load balance computation
            top1_idx = topk_idx[:, 0]
            f = torch.zeros(self.num_experts, device=x.device)
            for e in range(self.num_experts):
                f[e] = (top1_idx == e).float().mean()
            P = gate_probs.mean(dim=0)  # (E,)
            self._aux_loss = self.aux_loss_coeff * self.num_experts * (f * P).sum()
        else:
            self._aux_loss = torch.tensor(0.0, device=x.device)

        if reshaped:
            out = out.view(B, L, self.d_out)

        return out


__all__ = ["StandardMoEFFN"]
