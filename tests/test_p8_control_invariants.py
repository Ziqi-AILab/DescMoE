#!/usr/bin/env python3
"""Fast local invariants for P8 stable assignment and molecule routing."""

from __future__ import annotations

import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MOLESG = ROOT / "MoleSG"
sys.path.insert(0, str(MOLESG))
sys.path.insert(0, str(MOLESG / "pretrain"))

from Data_process.stable_random_assignment import (  # noqa: E402
    assign_scores,
    canonicalize_smiles,
    occupancy_matched_score_edges,
    stable_random_score,
    stable_random_scores,
)
from standard_moe import StandardMoEFFN  # noqa: E402


def test_stable_assignment() -> None:
    ethanol_a = canonicalize_smiles("CCO")
    ethanol_b = canonicalize_smiles("OCC")
    assert ethanol_a == ethanol_b
    assert stable_random_score(ethanol_a, "tpsa", 42) == stable_random_score(
        ethanol_b, "tpsa", 42)
    assert stable_random_score(ethanol_a, "tpsa", 42) != stable_random_score(
        ethanol_a, "logp", 42)

    molecules = [canonicalize_smiles(value) for value in (
        "CCO", "OCC", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CO", "CN")]
    scores = stable_random_scores(molecules, "tpsa", 42)
    edges = occupancy_matched_score_edges(scores, [2, 1, 1, 1, 1, 1, 1, 0])
    assignments = assign_scores(scores, edges)
    assert assignments[0] == assignments[1]
    assert assignments.min() >= 0 and assignments.max() <= 7


def test_molecule_router_gradient() -> None:
    torch.manual_seed(42)
    router = StandardMoEFFN(
        d_model=8,
        d_ff=8,
        d_out=8,
        num_experts=8,
        top_k=1,
        dropout=0.0,
        routing_granularity="molecule",
    )
    router.train()
    nodes = torch.randn(4, 5, 8, requires_grad=True)
    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ], dtype=torch.bool)
    output = router(nodes, node_mask=mask)
    main_loss = output.square().mean()
    main_loss.backward()
    assert output.shape == nodes.shape
    assert router._last_top1_ids is not None
    assert router._last_top1_ids.shape == (nodes.shape[0],)
    assert router.gate.weight.grad is not None
    assert float(router.gate.weight.grad.abs().sum()) > 0.0

    original_ids = router._last_top1_ids.clone()
    modified = nodes.detach().clone()
    modified[~mask] = 1.0e6
    router(modified, node_mask=mask)
    assert torch.equal(original_ids, router._last_top1_ids)


def test_pretrain_to_downstream_checkpoint_load() -> None:
    """Build both model variants in their native import contexts and load strictly."""
    with tempfile.TemporaryDirectory(prefix="p8_model_test_") as directory:
        checkpoint = Path(directory) / "p8_molecule_router.pt"
        build_code = """
import sys
import torch
from transformer_graph import make_model

model = make_model(
    d_atom=115, d_edge=13, N=8, d_model=256, h=8, dropout=0.0,
    distance_matrix_kernel='exp', scale_norm=True,
    use_standard_moe=True, num_experts=8, moe_top_k=1,
    moe_aux_loss_coeff=0.01, router_granularity='molecule',
    moe_layer_mode='last')
assert model.encoder.layers[-1].feed_forward.routing_granularity == 'molecule'
assert all(not hasattr(layer.feed_forward, 'routing_granularity')
           for layer in model.encoder.layers[:-1])
torch.save({'state_dict': model.state_dict()}, sys.argv[1])
"""
        load_code = """
import sys
import torch
from train_graph import _load_pretrained_checkpoint
from transformer_graph_finetune import make_model

model = make_model(
    d_atom=115, d_edge=13, N=8, d_model=256, h=8, dropout=0.0,
    distance_matrix_kernel='exp', scale_norm=True,
    use_standard_moe=True, num_experts=8, moe_top_k=1,
    moe_aux_loss_coeff=0.01, router_granularity='molecule',
    moe_layer_mode='last')
_load_pretrained_checkpoint(model, sys.argv[1], torch.device('cpu'),
                            strict_backbone_load=True)
print('P8_STRICT_CHECKPOINT_LOAD_OK')
"""
        environment = dict(**__import__("os").environ)
        environment["PYTHONPATH"] = str(MOLESG)
        subprocess.run(
            [sys.executable, "-c", build_code, str(checkpoint)],
            cwd=MOLESG / "pretrain", env=environment, check=True)
        subprocess.run(
            [sys.executable, "-c", load_code, str(checkpoint)],
            cwd=MOLESG / "Downstream", env=environment, check=True)


def main() -> int:
    test_stable_assignment()
    test_molecule_router_gradient()
    test_pretrain_to_downstream_checkpoint_load()
    print("P8_CONTROL_INVARIANTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
