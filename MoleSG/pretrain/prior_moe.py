"""Prior-knowledge based MoE (no router) + KAN expert + contrastive expert loss.

Adapted from KERMT/kermt/model/prior_moe.py for MoleSG's Graph Transformer.
Key difference: supports 3D input (batch, seq_len, d_model) in addition to 2D.

Three innovations (all switchable via if-branches):
  1. PriorMoEFFN: No-router MoE with offline expert assignment from chemical priors.
  2. KANExpert: B-spline KAN expert as drop-in replacement for MLP expert.
  3. expert_contrastive_loss: SupCon-style loss pulling same-expert embeddings together.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------
def _get_activation(name: str) -> nn.Module:
    name = (name or "mish").lower()
    table = {
        "relu": nn.ReLU,
        "prelu": nn.PReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "leakyrelu": nn.LeakyReLU,
        "elu": nn.ELU,
        "mish": nn.Mish,
    }
    if name not in table:
        raise ValueError(f"Unsupported activation: {name}")
    return table[name]()


# ---------------------------------------------------------------------------
# MLP Expert
# ---------------------------------------------------------------------------
class _MLPExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, d_out: int,
                 activation: str = "mish", dropout: float = 0.1):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_out)
        self.act_func = _get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_2(self.dropout(self.act_func(self.W_1(x))))


# ---------------------------------------------------------------------------
# KAN Expert
# ---------------------------------------------------------------------------
class KANExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, d_out: int,
                 grid_size: int = 5, spline_order: int = 3,
                 dropout: float = 0.1, grid_range: Tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        self.layer1 = _KANLinear(d_model, d_ff,
                                 grid_size=grid_size, spline_order=spline_order,
                                 grid_range=grid_range)
        self.layer2 = _KANLinear(d_ff, d_out,
                                 grid_size=grid_size, spline_order=spline_order,
                                 grid_range=grid_range)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)
        h = self.dropout(h)
        return self.layer2(h)


class _KANLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 5, spline_order: int = 3,
                 grid_range: Tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        )
        grid = grid.expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)] + 1e-8)
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k] + 1e-8)
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(F.silu(x), self.base_weight)
        bases = self._b_splines(x)
        spline = torch.einsum("nik,oik->no", bases, self.spline_weight)
        return base + spline


# ---------------------------------------------------------------------------
# No-router Prior MoE FFN (supports 2D and 3D input)
# ---------------------------------------------------------------------------
class PriorMoEFFN(nn.Module):
    """No-router MoE FFN with externally assigned expert ids.

    Supports both 2D input (N, d_model) and 3D input (batch, seq_len, d_model).
    For 3D input, expert_ids should be (batch,) — each molecule gets one expert,
    broadcast across all its nodes.
    """

    def __init__(self,
                 d_model: int,
                 d_ff: int = 0,
                 activation: str = "mish",
                 dropout: float = 0.1,
                 d_out: Optional[int] = None,
                 num_experts: int = 8,
                 expert_type: str = "ffn",
                 kan_expert_indices: Optional[Sequence[int]] = None,
                 kan_grid_size: int = 5,
                 kan_spline_order: int = 3):
        super().__init__()
        if d_out is None:
            d_out = d_model
        if d_ff <= 0:
            d_ff = d_model  # MoleSG FFN uses d_model → d_model (no expansion)
        self.num_experts = num_experts
        self.d_out = d_out
        self.expert_type = expert_type

        kan_set = set(kan_expert_indices or [])
        experts: List[nn.Module] = []
        for i in range(num_experts):
            if expert_type == "ffn":
                experts.append(_MLPExpert(d_model, d_ff, d_out, activation, dropout))
            elif expert_type == "kan":
                experts.append(KANExpert(d_model, d_ff, d_out,
                                         grid_size=kan_grid_size,
                                         spline_order=kan_spline_order,
                                         dropout=dropout))
            elif expert_type == "mixed":
                if i in kan_set:
                    experts.append(KANExpert(d_model, d_ff, d_out,
                                             grid_size=kan_grid_size,
                                             spline_order=kan_spline_order,
                                             dropout=dropout))
                else:
                    experts.append(_MLPExpert(d_model, d_ff, d_out, activation, dropout))
            else:
                raise ValueError(f"Unknown expert_type: {expert_type}")
        self.experts = nn.ModuleList(experts)

        self._last_expert_ids: Optional[torch.Tensor] = None
        self._last_output: Optional[torch.Tensor] = None
        self._cache_for_contrastive: bool = False

    def enable_contrastive_cache(self, flag: bool = True) -> None:
        self._cache_for_contrastive = flag

    def pop_cached(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        out, ids = self._last_output, self._last_expert_ids
        self._last_output, self._last_expert_ids = None, None
        return out, ids

    # -- Expert freezing for finetune ------------------------------------------
    def freeze_inactive_experts(self, active_ids: torch.Tensor) -> None:
        """Freeze experts not activated by *active_ids*; unfreeze active ones.

        Call *before* forward+backward so the optimizer skips weight-decay
        and moment updates on inactive experts.
        """
        active_set = set(active_ids.unique().tolist())
        for i, expert in enumerate(self.experts):
            flag = (i in active_set)
            for p in expert.parameters():
                p.requires_grad = flag

    def unfreeze_all_experts(self) -> None:
        """Restore requires_grad=True on every expert (call after step)."""
        for expert in self.experts:
            for p in expert.parameters():
                p.requires_grad = True

    def forward(self,
                x: torch.Tensor,
                expert_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (N, d) or (B, L, d).  expert_ids: (N,) or (B,)."""
        # --- handle 3D input ---
        reshaped = False
        if x.dim() == 3:
            B, L, D = x.shape
            x = x.reshape(B * L, D)
            if expert_ids is not None:
                # (B,) → (B, L) → (B*L,)
                expert_ids = expert_ids.unsqueeze(1).expand(B, L).contiguous().reshape(B * L)
            reshaped = True

        N = x.shape[0]
        if expert_ids is None:
            expert_ids = x.new_zeros(N, dtype=torch.long)
        else:
            expert_ids = expert_ids.to(device=x.device, dtype=torch.long)
            expert_ids = expert_ids.clamp(0, self.num_experts - 1)

        out = x.new_zeros(N, self.d_out)
        for e in range(self.num_experts):
            mask = (expert_ids == e)
            if mask.any():
                out[mask] = self.experts[e](x[mask])

        if self._cache_for_contrastive:
            self._last_expert_ids = expert_ids.detach()
            self._last_output = out

        if reshaped:
            out = out.view(B, L, self.d_out)

        return out


# ---------------------------------------------------------------------------
# Contrastive expert loss
# ---------------------------------------------------------------------------
def expert_contrastive_loss(embeddings: torch.Tensor,
                            expert_ids: torch.Tensor,
                            temperature: float = 0.1,
                            reduction: str = "mean",
                            max_anchors: Optional[int] = 4096) -> torch.Tensor:
    """SupCon loss: same-expert pulled together, different-expert pushed apart."""
    if embeddings.numel() == 0:
        return embeddings.new_zeros(())

    N = embeddings.shape[0]
    if max_anchors is not None and N > max_anchors:
        idx = torch.randperm(N, device=embeddings.device)[:max_anchors]
        embeddings = embeddings[idx]
        expert_ids = expert_ids[idx]
        N = embeddings.shape[0]

    z = F.normalize(embeddings, dim=-1)
    sim = z @ z.t() / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    pos_mask = (expert_ids.unsqueeze(0) == expert_ids.unsqueeze(1)) & ~eye

    if not pos_mask.any():
        return embeddings.new_zeros(())

    exp_sim = torch.exp(sim).masked_fill(eye, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1).clamp_min(1).to(log_prob.dtype)
    loss_per_row = -(log_prob * pos_mask).sum(dim=1) / pos_count

    valid = pos_mask.any(dim=1)
    loss_per_row = loss_per_row[valid]

    if reduction == "mean":
        return loss_per_row.mean()
    if reduction == "sum":
        return loss_per_row.sum()
    return loss_per_row


# ---------------------------------------------------------------------------
# Default bucket edges based on chemical knowledge (dataset-independent)
#   TPSA: 20 Å² steps, aligned with Veber rule (TPSA ≤ 140 Å²)
#   LogP: 1.0 steps, aligned with Lipinski rule (LogP ≤ 5)
# ---------------------------------------------------------------------------
_DEFAULT_CHEM_EDGES = {
    "tpsa":  [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0],
    "logp":  [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
}


# ---------------------------------------------------------------------------
# Prior assignment
# ---------------------------------------------------------------------------
def assign_expert_ids(smiles_list: Sequence[str],
                      property_name: str = "tpsa",
                      num_experts: int = 8,
                      bucket_edges: Optional[Sequence[float]] = None) -> List[int]:
    """Assign one expert id per molecule from a chemical-prior rule.

    Supported property_name: "tpsa", "logp", "mw", "n_heavy", "n_rings".
    TPSA and MolLogP use the fixed, dataset-independent edges defined in
    _DEFAULT_CHEM_EDGES.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    except ImportError as e:
        raise ImportError("rdkit is required for prior expert assignment") from e

    def _prop(mol) -> float:
        if property_name == "tpsa":
            return Descriptors.TPSA(mol)
        if property_name == "mw":
            return Descriptors.MolWt(mol)
        if property_name == "logp":
            return Crippen.MolLogP(mol)
        if property_name == "n_heavy":
            return float(mol.GetNumHeavyAtoms())
        if property_name == "n_rings":
            return float(rdMolDescriptors.CalcNumRings(mol))
        raise ValueError(f"Unknown property_name: {property_name}")

    if bucket_edges is None:
        if property_name in _DEFAULT_CHEM_EDGES:
            edges = list(_DEFAULT_CHEM_EDGES[property_name])
        else:
            edges = [50 + (550 / num_experts) * i for i in range(1, num_experts)]
    else:
        edges = list(bucket_edges)

    ids: List[int] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else smi
        if mol is None:
            ids.append(0)
            continue
        v = _prop(mol)
        e = 0
        for thr in edges:
            if v >= thr:
                e += 1
            else:
                break
        ids.append(min(e, num_experts - 1))
    return ids


def assign_expert_ids_by_num_atoms(num_atoms_list, num_experts: int = 8,
                                    bucket_edges: Optional[Sequence[float]] = None) -> List[int]:
    """Assign expert ids based on number of atoms (no rdkit needed).

    Useful for pretrain where SMILES are not available in the data pipeline.
    """
    if bucket_edges is None:
        # default: equal-width buckets from 5 to 50 atoms
        edges = [5 + (45 / num_experts) * i for i in range(1, num_experts)]
    else:
        edges = list(bucket_edges)

    ids: List[int] = []
    for n in num_atoms_list:
        n = float(n)
        e = 0
        for thr in edges:
            if n >= thr:
                e += 1
            else:
                break
        ids.append(min(e, num_experts - 1))
    return ids


__all__ = [
    "PriorMoEFFN",
    "KANExpert",
    "expert_contrastive_loss",
    "assign_expert_ids",
    "assign_expert_ids_by_num_atoms",
]
