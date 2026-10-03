#!/usr/bin/env python3
"""Prepare deterministic descriptor values and matched branch assignments."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors

ROOT = Path(__file__).resolve().parents[1]
MOLESG = ROOT / "MoleSG"
sys.path.insert(0, str(MOLESG))

from Data_process.stable_random_assignment import (  # noqa: E402
    assign_scores,
    canonicalize_smiles,
    occupancy_matched_score_edges,
    stable_random_scores,
)


DATASETS = ["bbbp", "tox21", "toxcast", "sider", "clintox", "bace", "hiv", "muv"]
FIXED_EDGES = {
    "tpsa": [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0],
    "logp": [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zinc-csv", type=Path, required=True)
    parser.add_argument("--downstream-data", type=Path, required=True)
    parser.add_argument(
        "--output-dir", default=str(ROOT / "runs/control_data"))
    parser.add_argument(
        "--zinc-output", default=str(ROOT / "runs/control_data/descriptor_controls.npz"))
    return parser.parse_args()


def descriptor_value(molecule: Chem.Mol, partition: str) -> float:
    if partition == "tpsa":
        return float(Descriptors.TPSA(molecule))
    if partition == "logp":
        return float(Crippen.MolLogP(molecule))
    raise ValueError(partition)


def values_from_smiles(smiles: list[str], partition: str) -> np.ndarray:
    values = []
    for row_index, smile in enumerate(smiles):
        molecule = Chem.MolFromSmiles(smile)
        if molecule is None:
            raise ValueError(f"Invalid SMILES at row {row_index}: {smile}")
        values.append(descriptor_value(molecule, partition))
    return np.asarray(values, dtype=np.float64)


def assign(values: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges), values, side="right").astype(np.int64)


def quantile_edges(values: np.ndarray) -> list[float]:
    edges = np.quantile(values, np.arange(1, 8) / 8, method="linear")
    if len(np.unique(edges)) != 7:
        raise ValueError("Quantile edges are not unique")
    return [float(value) for value in edges]


def randomize_assignments(ids: np.ndarray, seed: int) -> np.ndarray:
    randomized = ids.copy()
    np.random.default_rng(seed).shuffle(randomized)
    if not np.array_equal(np.bincount(ids, minlength=8),
                          np.bincount(randomized, minlength=8)):
        raise AssertionError("Random assignment changed occupancy")
    return randomized


def minimum_occupancy_edges(values: np.ndarray, minimum_count: int) -> list[float]:
    """Build eight deterministic hybrid bins while retaining valid fixed edges."""
    edges = FIXED_EDGES["logp"][:5]
    while len(edges) < 7:
        ids = assign(values, edges)
        counts = np.bincount(ids, minlength=len(edges) + 1)
        candidates = [
            (int(count), region)
            for region, count in enumerate(counts)
            if count >= 2 * minimum_count
        ]
        if not candidates:
            raise ValueError("No region can be split while preserving minimum occupancy")
        _, region = max(candidates, key=lambda item: (item[0], -item[1]))
        region_values = values[ids == region]
        split = float(np.median(region_values))
        lower = float(region_values.min())
        upper = float(region_values.max())
        if not lower < split < upper or split in edges:
            unique = np.unique(region_values)
            split = float(unique[len(unique) // 2])
        if not lower < split < upper or split in edges:
            raise ValueError(f"Cannot split MolLogP region {region}")
        edges = sorted([*edges, split])
    counts = np.bincount(assign(values, edges), minlength=8)
    if counts.min() < minimum_count:
        raise ValueError(
            f"Minimum-occupancy edges failed: minimum={counts.min()} target={minimum_count}")
    return [float(edge) for edge in edges]


def prepare_zinc(output_dir: Path, zinc_output: Path, seed: int, zinc_csv: Path) -> dict:
    frame = pd.read_csv(zinc_csv)
    smiles = frame["smiles"].astype(str).tolist()
    canonical_smiles = [canonicalize_smiles(smile) for smile in smiles]
    arrays: dict[str, np.ndarray] = {}
    metadata = {
        "seed": seed,
        "molecules": len(smiles),
        "canonical_molecules": len(set(canonical_smiles)),
        "stable_random_generator": "NumPy SeedSequence with PCG64",
        "stable_random_key": "RDKit canonical SMILES bytes, descriptor code, seed",
        "partitions": {},
    }
    for partition in ("tpsa", "logp"):
        values = values_from_smiles(smiles, partition)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        q_edges = quantile_edges(values)
        fixed = assign(values, FIXED_EDGES[partition])
        quantile = assign(values, q_edges)
        random_occ = randomize_assignments(fixed, seed)
        stable_scores = stable_random_scores(canonical_smiles, partition, seed)
        stable_edges = occupancy_matched_score_edges(
            stable_scores, np.bincount(fixed, minlength=8))
        stable_random = assign_scores(stable_scores, stable_edges)
        arrays[f"{partition}_values"] = values.astype(np.float32)
        arrays[f"{partition}_standardized"] = ((values - mean) / std).astype(np.float32)
        arrays[f"{partition}_fixed"] = fixed
        arrays[f"{partition}_quantile"] = quantile
        arrays[f"{partition}_random_occ"] = random_occ
        arrays[f"{partition}_stable_random"] = stable_random
        metadata["partitions"][partition] = {
            "mean": mean,
            "std": std,
            "fixed_edges": FIXED_EDGES[partition],
            "quantile_edges": q_edges,
            "fixed_occupancy": np.bincount(fixed, minlength=8).tolist(),
            "quantile_occupancy": np.bincount(quantile, minlength=8).tolist(),
            "random_occupancy": np.bincount(random_occ, minlength=8).tolist(),
            "stable_random_score_edges": stable_edges,
            "stable_random_target_occupancy": np.bincount(
                fixed, minlength=8).tolist(),
            "stable_random_occupancy": np.bincount(
                stable_random, minlength=8).tolist(),
        }
        if partition == "logp":
            merge_tail_edges = FIXED_EDGES["logp"][:5]
            min_edges = minimum_occupancy_edges(values, minimum_count=2500)
            merge_tail = assign(values, merge_tail_edges)
            min_occupancy = assign(values, min_edges)
            arrays["logp_merge_tail"] = merge_tail
            arrays["logp_min_occupancy"] = min_occupancy
            metadata["partitions"][partition].update({
                "merge_tail_edges": merge_tail_edges,
                "merge_tail_occupancy": np.bincount(
                    merge_tail, minlength=8).tolist(),
                "min_occupancy_threshold": 2500,
                "min_occupancy_edges": min_edges,
                "min_occupancy_occupancy": np.bincount(
                    min_occupancy, minlength=8).tolist(),
            })
    zinc_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(zinc_output, **arrays)
    metadata["zinc_control_file"] = str(zinc_output)
    (output_dir / "descriptor_control_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    return metadata


def prepare_downstream(output_dir: Path, metadata: dict, seed: int,
                       downstream_data: Path) -> None:
    mapping_dir = output_dir / "downstream_random_assignments"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    occupancy_rows = []
    for dataset in DATASETS:
        with (downstream_data / dataset / "preprocess" / f"{dataset}.pickle").open("rb") as handle:
            molecules, _ = pickle.load(handle)
        smiles = [Chem.MolToSmiles(molecule) for molecule in molecules]
        for partition in ("tpsa", "logp"):
            values = np.asarray([
                descriptor_value(molecule, partition) for molecule in molecules
            ])
            fixed = assign(values, FIXED_EDGES[partition])
            quantile = assign(
                values, metadata["partitions"][partition]["quantile_edges"])
            random_occ = randomize_assignments(fixed, seed)
            stable_scores = stable_random_scores(smiles, partition, seed)
            stable_random = assign_scores(
                stable_scores,
                metadata["partitions"][partition]["stable_random_score_edges"])
            sample_ids = [f"{dataset}:{index}" for index in range(len(molecules))]
            output = {
                "sample_id": sample_ids,
                "source_row_index": np.arange(len(molecules)),
                "canonical_smiles": smiles,
                "fixed_expert_id": fixed,
                "quantile_expert_id": quantile,
                "random_occ_expert_id": random_occ,
                "stable_random_score": stable_scores,
                "stable_random_expert_id": stable_random,
                "descriptor_value": values,
            }
            schemes = [("fixed", fixed), ("quantile", quantile),
                       ("random_occ", random_occ),
                       ("stable_random", stable_random)]
            if partition == "logp":
                merge_tail = assign(
                    values, metadata["partitions"][partition]["merge_tail_edges"])
                min_occupancy = assign(
                    values, metadata["partitions"][partition]["min_occupancy_edges"])
                output["merge_tail_expert_id"] = merge_tail
                output["min_occupancy_expert_id"] = min_occupancy
                schemes.extend([
                    ("merge_tail", merge_tail),
                    ("min_occupancy", min_occupancy),
                ])
            pd.DataFrame(output).to_csv(
                mapping_dir / f"{dataset}_{partition}.csv", index=False)
            for scheme, ids in schemes:
                counts = np.bincount(ids, minlength=8)
                for region, count in enumerate(counts):
                    occupancy_rows.append({
                        "dataset": dataset,
                        "partition": partition,
                        "scheme": scheme,
                        "region": region,
                        "count": int(count),
                    })
    pd.DataFrame(occupancy_rows).to_csv(
        output_dir / "downstream_assignment_occupancy.csv", index=False)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = prepare_zinc(output_dir, Path(args.zinc_output), args.seed, args.zinc_csv)
    prepare_downstream(output_dir, metadata, args.seed, args.downstream_data)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
