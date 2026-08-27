#!/usr/bin/env python3
"""Smoke-test the last-layer mixed MLP/KAN configuration and ConLoss."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PRETRAIN = ROOT / "MoleSG" / "pretrain"
sys.path.insert(0, str(PRETRAIN))

from prior_moe import KANExpert, PriorMoEFFN, expert_contrastive_loss  # noqa: E402
from transformer_graph import PositionwiseFeedForward, make_model  # noqa: E402


def main() -> None:
    selected = [0, 1, 2, 3, 5]
    model = make_model(
        d_atom=16,
        d_edge=8,
        N=8,
        d_model=32,
        h=8,
        dropout=0.0,
        N_dense=2,
        distance_matrix_kernel="exp",
        use_prior_moe=True,
        num_experts=8,
        expert_type="mixed",
        kan_expert_indices=selected,
        moe_layer_mode="last",
    )

    feed_forwards = [layer.feed_forward for layer in model.encoder.layers]
    assert all(isinstance(module, PositionwiseFeedForward) for module in feed_forwards[:7])
    assert isinstance(feed_forwards[7], PriorMoEFFN)
    branch_types = [isinstance(branch, KANExpert) for branch in feed_forwards[7].experts]
    assert branch_types == [index in selected for index in range(8)]

    hidden = torch.randn(4, 3, 32)
    region_ids = torch.tensor([0, 4, 5, 7])
    output = feed_forwards[7](hidden, region_ids)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()

    pooled = output.mean(dim=1)
    loss = expert_contrastive_loss(pooled, region_ids, temperature=0.1)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    print("model component smoke test OK layers=8 descriptor_layer=8 branches=8")


if __name__ == "__main__":
    main()
