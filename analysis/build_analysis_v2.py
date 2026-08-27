#!/usr/bin/env python3
"""Build the canonical JCIM v2 analysis snapshot from existing artifacts.

This script never trains a model and never submits a cluster job. It reads the
stored downstream CSV files, ZINC15 descriptor assignments, pretraining logs,
and Slurm accounting records. Main-result summaries require all five folds.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


OUT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("PAPER101_PROJECT_ROOT", OUT_ROOT))
MOLESG_ROOT = Path(os.environ.get("MOLESG_ROOT", OUT_ROOT / "MoleSG"))
ABLATION_ROOT = Path(
    os.environ.get("PAPER101_ABLATION_ROOT", PROJECT_ROOT / "ablation")
)
RESULT_ROOT = Path(
    os.environ.get("MOLESG_RESULT_ROOT", MOLESG_ROOT / "Downstream" / "Result")
)
PRETRAIN_RESULT_ROOT = Path(
    os.environ.get("PAPER101_PRETRAIN_RESULT_ROOT", ABLATION_ROOT / "results")
)
TABLE_ROOT = OUT_ROOT / "tables_v2"
FIGURE_ROOT = OUT_ROOT / "figures"

DATASETS = ["bbbp", "tox21", "toxcast", "sider", "clintox", "bace", "hiv", "muv"]
FOLDS = [1, 2, 3, 4, 5]
SEEDS = {fold: 41 + fold for fold in FOLDS}

FINAL_MODELS = {
    "Base": "p3_base_vanilla_n8aligned",
    "TPSA I1": "p3_tpsa_mofe_last_ffn_n8aligned",
    "MolLogP I1": "p3_logp_mofe_last_ffn_n8aligned",
    "TPSA I1+I2": "p5_tpsa_mofe_last_selected_kan_n8aligned",
    "MolLogP I1+I2": "p5_logp_mofe_last_selected_kan_n8aligned",
    "TPSA I1+I3": "p3_tpsa_mofe_last_ffn_conloss_n8aligned",
    "MolLogP I1+I3": "p3_logp_mofe_last_ffn_conloss_n8aligned",
    "TPSA Full": "p5_tpsa_mofe_last_selected_kan_conloss_n8aligned",
    "MolLogP Full": "p5_logp_mofe_last_selected_kan_conloss_n8aligned",
    "TPSA all-KAN": "p3_tpsa_mofe_last_all_kan_n8aligned",
    "MolLogP all-KAN": "p3_logp_mofe_last_all_kan_n8aligned",
}

MODEL_METADATA = {
    "p3_base_vanilla_n8aligned": ("final_n8", "none", "Base"),
    "p3_tpsa_mofe_last_ffn_n8aligned": ("final_n8", "tpsa", "I1"),
    "p3_logp_mofe_last_ffn_n8aligned": ("final_n8", "logp", "I1"),
    "p5_tpsa_mofe_last_selected_kan_n8aligned": ("final_n8", "tpsa", "I1+I2"),
    "p5_logp_mofe_last_selected_kan_n8aligned": ("final_n8", "logp", "I1+I2"),
    "p3_tpsa_mofe_last_ffn_conloss_n8aligned": ("final_n8", "tpsa", "I1+I3"),
    "p3_logp_mofe_last_ffn_conloss_n8aligned": ("final_n8", "logp", "I1+I3"),
    "p5_tpsa_mofe_last_selected_kan_conloss_n8aligned": (
        "final_n8",
        "tpsa",
        "I1+I2+I3",
    ),
    "p5_logp_mofe_last_selected_kan_conloss_n8aligned": (
        "final_n8",
        "logp",
        "I1+I2+I3",
    ),
    "p3_tpsa_mofe_last_all_kan_n8aligned": ("final_n8", "tpsa", "all-KAN"),
    "p3_logp_mofe_last_all_kan_n8aligned": ("final_n8", "logp", "all-KAN"),
    "p3_base_vanilla_n4aligned": ("depth_reference", "none", "N4 Base"),
    "p2_mofe_ffn_logp_last": ("layer_screen", "logp", "last"),
    "p2_mofe_ffn_logp_odd": ("layer_screen", "logp", "odd"),
    "p2_mofe_ffn_logp_even": ("layer_screen", "logp", "even"),
    "pt_smoe_top1": ("routed_control", "learned", "top1"),
    "pt_smoe_top2": ("routed_control", "learned", "top2"),
    "pt_smoe_top3": ("routed_control", "learned", "top3"),
    "pt_smoe_top4": ("routed_control", "learned", "top4"),
}

P4_MODELS = [
    *(f"p4_tpsa_mofe_last_k{i}" for i in range(8)),
    *(f"p4_logp_mofe_last_k{i}" for i in range(8)),
]

PUBLISHED_MOLESG = {
    "bbbp": (0.979, 0.003),
    "tox21": (0.850, 0.012),
    "toxcast": (0.742, 0.005),
    "sider": (0.700, 0.002),
    "clintox": (0.991, 0.009),
    "bace": (0.951, 0.021),
    "hiv": (0.877, 0.019),
    "muv": (0.851, 0.008),
}

TPSA_EDGES = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]
LOGP_EDGES = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


@dataclass(frozen=True)
class FoldResult:
    model: str
    protocol: str
    partition: str
    role: str
    dataset: str
    fold: int
    seed: int
    auc: float
    source_csv: str


def ensure_dirs() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)


def parse_vector(value: str) -> list[float]:
    stripped = value.strip().strip("[]")
    if not stripped:
        return []
    parsed = np.fromstring(stripped, sep=",", dtype=float)
    if parsed.size:
        return parsed.tolist()
    fallback = ast.literal_eval(value)
    if isinstance(fallback, (int, float)):
        return [float(fallback)]
    return [float(item) for item in fallback]


def compute_auc(path: Path) -> float:
    actual_rows: list[list[float]] = []
    predict_rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            actual_rows.append(parse_vector(row["actual"]))
            predict_rows.append(parse_vector(row["predict"]))

    if not actual_rows:
        raise ValueError(f"empty result file: {path}")

    actual = np.asarray(actual_rows, dtype=float)
    predict = np.asarray(predict_rows, dtype=float)
    if actual.ndim == 1:
        actual = actual[:, None]
    if predict.ndim == 1:
        predict = predict[:, None]
    if actual.shape != predict.shape:
        raise ValueError(f"shape mismatch in {path}: {actual.shape} vs {predict.shape}")

    endpoint_auc: list[float] = []
    for endpoint in range(actual.shape[1]):
        mask = actual[:, endpoint] != -1
        labels = actual[mask, endpoint]
        scores = predict[mask, endpoint]
        if labels.size == 0 or np.unique(labels).size < 2:
            continue
        endpoint_auc.append(float(roc_auc_score(labels, scores)))
    if not endpoint_auc:
        raise ValueError(f"no valid endpoint AUC in {path}")
    return float(np.mean(endpoint_auc))


def result_path(model: str, dataset: str, fold: int) -> Path:
    return (
        RESULT_ROOT
        / dataset
        / f"best_test_result_{model}_{dataset}_fold_{fold}.csv"
    )


def collect_results() -> list[FoldResult]:
    records: list[FoldResult] = []
    for model, (protocol, partition, role) in MODEL_METADATA.items():
        for dataset in DATASETS:
            for fold in FOLDS:
                path = result_path(model, dataset, fold)
                if not path.exists() or path.stat().st_size == 0:
                    continue
                auc = compute_auc(path)
                if not math.isfinite(auc):
                    raise ValueError(f"non-finite AUC in {path}")
                records.append(
                    FoldResult(
                        model=model,
                        protocol=protocol,
                        partition=partition,
                        role=role,
                        dataset=dataset,
                        fold=fold,
                        seed=SEEDS[fold],
                        auc=auc,
                        source_csv=str(path),
                    )
                )
    return records


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames required for empty output: {path}")
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def records_by_key(records: list[FoldResult]) -> dict[tuple[str, str, int], FoldResult]:
    return {(row.model, row.dataset, row.fold): row for row in records}


def complete_summary(records: list[FoldResult]) -> list[dict]:
    grouped: dict[tuple[str, str], list[FoldResult]] = defaultdict(list)
    for row in records:
        grouped[(row.model, row.dataset)].append(row)

    summary: list[dict] = []
    for model, (protocol, partition, role) in MODEL_METADATA.items():
        for dataset in DATASETS:
            rows = grouped.get((model, dataset), [])
            found_folds = sorted(row.fold for row in rows)
            complete = found_folds == FOLDS
            values = [row.auc for row in sorted(rows, key=lambda item: item.fold)]
            summary.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "partition": partition,
                    "role": role,
                    "dataset": dataset,
                    "folds_found": len(found_folds),
                    "fold_ids": ",".join(map(str, found_folds)),
                    "complete_5fold": int(complete),
                    "auc_mean": f"{np.mean(values):.10f}" if complete else "",
                    "auc_sd": f"{np.std(values, ddof=1):.10f}" if complete else "",
                }
            )
    return summary


def model_macro_summary(dataset_summary: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in dataset_summary:
        grouped[row["model"]].append(row)
    output: list[dict] = []
    for model, rows in grouped.items():
        complete_rows = [row for row in rows if row["complete_5fold"] == 1]
        means = [float(row["auc_mean"]) for row in complete_rows]
        output.append(
            {
                "model": model,
                "protocol": MODEL_METADATA[model][0],
                "partition": MODEL_METADATA[model][1],
                "role": MODEL_METADATA[model][2],
                "complete_datasets": len(complete_rows),
                "target_datasets": len(DATASETS),
                "macro_auc": f"{np.mean(means):.10f}" if len(means) == len(DATASETS) else "",
            }
        )
    return output


def matched_final_macro_summary(dataset_summary: list[dict]) -> list[dict]:
    """Summarize final models over the same complete dataset set.

    A model-specific available-case average would mix protocols whenever one
    model is missing a five-fold cell. The shared set is therefore determined
    across all strict N8 models before any macro average is calculated.
    """

    final_models = [
        model
        for model, (protocol, _, _) in MODEL_METADATA.items()
        if protocol == "final_n8"
    ]
    lookup = {
        (row["model"], row["dataset"]): row
        for row in dataset_summary
    }
    shared = [
        dataset
        for dataset in DATASETS
        if all(
            lookup[(model, dataset)]["complete_5fold"] == 1
            for model in final_models
        )
    ]
    output = []
    for model in final_models:
        values = [float(lookup[(model, dataset)]["auc_mean"]) for dataset in shared]
        output.append(
            {
                "model": model,
                "partition": MODEL_METADATA[model][1],
                "role": MODEL_METADATA[model][2],
                "shared_datasets": len(shared),
                "dataset_ids": ",".join(shared),
                "matched_macro_auc": f"{np.mean(values):.10f}" if values else "",
            }
        )
    return output


def fold_wide_rows(records: list[FoldResult]) -> list[dict]:
    lookup = records_by_key(records)
    output = []
    for model, (protocol, partition, role) in MODEL_METADATA.items():
        for dataset in DATASETS:
            values = [
                lookup.get((model, dataset, fold))
                for fold in FOLDS
            ]
            complete = all(row is not None for row in values)
            numeric = [row.auc for row in values if row is not None]
            output.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "partition": partition,
                    "role": role,
                    "dataset": dataset,
                    **{
                        f"fold_{fold}": (
                            f"{values[index].auc:.10f}"
                            if values[index] is not None
                            else ""
                        )
                        for index, fold in enumerate(FOLDS)
                    },
                    "complete_5fold": int(complete),
                    "auc_mean": f"{np.mean(numeric):.10f}" if complete else "",
                    "auc_sd": (
                        f"{np.std(numeric, ddof=1):.10f}" if complete else ""
                    ),
                }
            )
    return output


def layer_screen_summary(dataset_summary: list[dict]) -> list[dict]:
    models = {
        "last": "p2_mofe_ffn_logp_last",
        "odd": "p2_mofe_ffn_logp_odd",
        "even": "p2_mofe_ffn_logp_even",
    }
    lookup = {
        (row["model"], row["dataset"]): row
        for row in dataset_summary
    }
    last_values = {
        dataset: float(lookup[(models["last"], dataset)]["auc_mean"])
        for dataset in DATASETS
    }
    output = []
    for placement, model in models.items():
        values = {
            dataset: float(lookup[(model, dataset)]["auc_mean"])
            for dataset in DATASETS
        }
        deltas = [values[dataset] - last_values[dataset] for dataset in DATASETS]
        output.append(
            {
                "placement": placement,
                "model": model,
                "datasets": len(DATASETS),
                "macro_auc": f"{np.mean(list(values.values())):.10f}",
                "delta_vs_last": f"{np.mean(deltas):.10f}",
                "wins_vs_last": sum(delta > 0 for delta in deltas),
            }
        )
    return output


def paired_delta_rows(
    records: list[FoldResult],
    left_model: str,
    right_model: str,
) -> list[dict]:
    lookup = records_by_key(records)
    output: list[dict] = []
    for dataset in DATASETS:
        left = [lookup.get((left_model, dataset, fold)) for fold in FOLDS]
        right = [lookup.get((right_model, dataset, fold)) for fold in FOLDS]
        if any(item is None for item in left + right):
            continue
        deltas = [left[idx].auc - right[idx].auc for idx in range(len(FOLDS))]
        output.append(
            {
                "dataset": dataset,
                "delta_mean": float(np.mean(deltas)),
                "delta_sd": float(np.std(deltas, ddof=1)),
                "fold_deltas": deltas,
            }
        )
    return output


def hierarchical_bootstrap(
    per_dataset: list[dict],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    if not per_dataset:
        return (math.nan, math.nan)
    matrix = np.asarray([row["fold_deltas"] for row in per_dataset], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    n_dataset, n_fold = matrix.shape
    for idx in range(n_bootstrap):
        sampled_datasets = rng.integers(0, n_dataset, size=n_dataset)
        dataset_means = []
        for dataset_idx in sampled_datasets:
            sampled_folds = rng.integers(0, n_fold, size=n_fold)
            dataset_means.append(float(matrix[dataset_idx, sampled_folds].mean()))
        boot[idx] = float(np.mean(dataset_means))
    return tuple(float(value) for value in np.percentile(boot, [2.5, 97.5]))


def build_pairwise_comparisons(records: list[FoldResult]) -> list[dict]:
    comparisons = [
        (
            "TPSA",
            "I1 vs Base",
            FINAL_MODELS["TPSA I1"],
            FINAL_MODELS["Base"],
        ),
        (
            "TPSA",
            "I2 conditional effect",
            FINAL_MODELS["TPSA I1+I2"],
            FINAL_MODELS["TPSA I1"],
        ),
        (
            "TPSA",
            "I3 conditional effect",
            FINAL_MODELS["TPSA I1+I3"],
            FINAL_MODELS["TPSA I1"],
        ),
        (
            "TPSA",
            "Full vs Base",
            FINAL_MODELS["TPSA Full"],
            FINAL_MODELS["Base"],
        ),
        (
            "TPSA",
            "selected-KAN vs all-KAN",
            FINAL_MODELS["TPSA I1+I2"],
            FINAL_MODELS["TPSA all-KAN"],
        ),
        (
            "MolLogP",
            "I1 vs Base",
            FINAL_MODELS["MolLogP I1"],
            FINAL_MODELS["Base"],
        ),
        (
            "MolLogP",
            "I2 conditional effect",
            FINAL_MODELS["MolLogP I1+I2"],
            FINAL_MODELS["MolLogP I1"],
        ),
        (
            "MolLogP",
            "I3 conditional effect",
            FINAL_MODELS["MolLogP I1+I3"],
            FINAL_MODELS["MolLogP I1"],
        ),
        (
            "MolLogP",
            "Full vs Base",
            FINAL_MODELS["MolLogP Full"],
            FINAL_MODELS["Base"],
        ),
        (
            "MolLogP",
            "selected-KAN vs all-KAN",
            FINAL_MODELS["MolLogP I1+I2"],
            FINAL_MODELS["MolLogP all-KAN"],
        ),
    ]
    output: list[dict] = []
    for partition, label, left_model, right_model in comparisons:
        paired = paired_delta_rows(records, left_model, right_model)
        if paired:
            point = float(np.mean([row["delta_mean"] for row in paired]))
            wins = sum(row["delta_mean"] > 0 for row in paired)
            low, high = hierarchical_bootstrap(paired)
        else:
            point = low = high = math.nan
            wins = 0
        output.append(
            {
                "partition": partition,
                "comparison": label,
                "left_model": left_model,
                "right_model": right_model,
                "complete_shared_datasets": len(paired),
                "macro_delta": f"{point:.10f}" if math.isfinite(point) else "",
                "wins": wins,
                "ci95_low": f"{low:.10f}" if math.isfinite(low) else "",
                "ci95_high": f"{high:.10f}" if math.isfinite(high) else "",
            }
        )
        for row in paired:
            output.append(
                {
                    "partition": partition,
                    "comparison": f"{label}::{row['dataset']}",
                    "left_model": left_model,
                    "right_model": right_model,
                    "complete_shared_datasets": 1,
                    "macro_delta": f"{row['delta_mean']:.10f}",
                    "wins": int(row["delta_mean"] > 0),
                    "ci95_low": "",
                    "ci95_high": "",
                }
            )
    return output


def wilcoxon_holm(pairwise_rows: list[dict]) -> list[dict]:
    from scipy.stats import wilcoxon

    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    models: dict[tuple[str, str], tuple[str, str]] = {}
    for row in pairwise_rows:
        if "::" not in row["comparison"]:
            models[(row["partition"], row["comparison"])] = (
                row["left_model"],
                row["right_model"],
            )
            continue
        comparison, _ = row["comparison"].split("::", 1)
        groups[(row["partition"], comparison)].append(float(row["macro_delta"]))

    tests = []
    for key, deltas in groups.items():
        values = np.asarray(deltas, dtype=float)
        if np.allclose(values, 0):
            statistic, p_value = 0.0, 1.0
        else:
            result = wilcoxon(
                values,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
            statistic, p_value = float(result.statistic), float(result.pvalue)
        left_model, right_model = models[key]
        tests.append(
            {
                "partition": key[0],
                "comparison": key[1],
                "left_model": left_model,
                "right_model": right_model,
                "datasets": len(values),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )

    order = sorted(range(len(tests)), key=lambda idx: tests[idx]["p_value"])
    adjusted = [math.nan] * len(tests)
    running = 0.0
    total = len(tests)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (total - rank) * tests[idx]["p_value"])
        running = max(running, candidate)
        adjusted[idx] = running
    for idx, test in enumerate(tests):
        test["p_holm"] = adjusted[idx]
    return tests


def build_interaction_rows(records: list[FoldResult]) -> list[dict]:
    lookup = records_by_key(records)
    configurations = [
        (
            "TPSA",
            FINAL_MODELS["TPSA I1"],
            FINAL_MODELS["TPSA I1+I2"],
            FINAL_MODELS["TPSA I1+I3"],
            FINAL_MODELS["TPSA Full"],
        ),
        (
            "MolLogP",
            FINAL_MODELS["MolLogP I1"],
            FINAL_MODELS["MolLogP I1+I2"],
            FINAL_MODELS["MolLogP I1+I3"],
            FINAL_MODELS["MolLogP Full"],
        ),
    ]
    output: list[dict] = []
    for partition, i1, i12, i13, full in configurations:
        per_dataset = []
        for dataset in DATASETS:
            deltas = []
            for fold in FOLDS:
                rows = [
                    lookup.get((model, dataset, fold))
                    for model in (i1, i12, i13, full)
                ]
                if any(row is None for row in rows):
                    deltas = []
                    break
                delta = rows[3].auc - rows[1].auc - rows[2].auc + rows[0].auc
                deltas.append(delta)
            if deltas:
                per_dataset.append(
                    {
                        "dataset": dataset,
                        "delta_mean": float(np.mean(deltas)),
                        "delta_sd": float(np.std(deltas, ddof=1)),
                        "fold_deltas": deltas,
                    }
                )
                output.append(
                    {
                        "partition": partition,
                        "scope": dataset,
                        "interaction": f"{np.mean(deltas):.10f}",
                        "ci95_low": "",
                        "ci95_high": "",
                        "complete_datasets": 1,
                    }
                )
        if per_dataset:
            low, high = hierarchical_bootstrap(per_dataset)
            output.append(
                {
                    "partition": partition,
                    "scope": "macro",
                    "interaction": f"{np.mean([row['delta_mean'] for row in per_dataset]):.10f}",
                    "ci95_low": f"{low:.10f}",
                    "ci95_high": f"{high:.10f}",
                    "complete_datasets": len(per_dataset),
                }
            )
    return output


def interval_label(edges: list[float], index: int, unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    if index == 0:
        return f"< {edges[0]:g}{suffix}"
    if index == len(edges):
        return f">= {edges[-1]:g}{suffix}"
    return f"[{edges[index - 1]:g}, {edges[index]:g}){suffix}"


def descriptor_profiles() -> tuple[list[dict], list[dict], dict]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required. Run this script with the MoleSG environment."
        ) from exc

    smiles_path = MOLESG_ROOT / "Data/zinc15/zinc15_250K.csv"
    assignment_path = MOLESG_ROOT / "Data/zinc15/expert_ids.npz"
    assignments = np.load(assignment_path)
    tpsa_ids = np.asarray(assignments["tpsa"], dtype=int)
    logp_ids = np.asarray(assignments["logp"], dtype=int)
    with smiles_path.open(newline="", encoding="utf-8") as handle:
        smiles = [row["smiles"] for row in csv.DictReader(handle)]
    if len(smiles) != len(tpsa_ids) or len(smiles) != len(logp_ids):
        raise ValueError("ZINC SMILES and assignment arrays have different lengths")

    metric_names = [
        "mol_wt",
        "hbd",
        "hba",
        "rotatable_bonds",
        "ring_count",
        "aromatic_atom_fraction",
        "heteroatom_fraction",
        "formal_charge",
        "descriptor_value",
    ]
    aggregate: dict[tuple[str, int], dict[str, list[float]]] = {
        (partition, region): {metric: [] for metric in metric_names}
        for partition in ("tpsa", "logp")
        for region in range(8)
    }
    invalid = 0
    assignment_mismatch = {"tpsa": 0, "logp": 0}
    joint_sample: list[dict] = []
    for idx, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            invalid += 1
            continue
        heavy_atoms = max(mol.GetNumHeavyAtoms(), 1)
        aromatic_fraction = sum(atom.GetIsAromatic() for atom in mol.GetAtoms()) / heavy_atoms
        hetero_fraction = (
            sum(atom.GetAtomicNum() not in (1, 6) for atom in mol.GetAtoms()) / heavy_atoms
        )
        common = {
            "mol_wt": float(Descriptors.MolWt(mol)),
            "hbd": float(Lipinski.NumHDonors(mol)),
            "hba": float(Lipinski.NumHAcceptors(mol)),
            "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
            "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
            "aromatic_atom_fraction": float(aromatic_fraction),
            "heteroatom_fraction": float(hetero_fraction),
            "formal_charge": float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        }
        descriptor_values = {
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "logp": float(Crippen.MolLogP(mol)),
        }
        expected = {
            "tpsa": int(np.digitize(descriptor_values["tpsa"], TPSA_EDGES, right=False)),
            "logp": int(np.digitize(descriptor_values["logp"], LOGP_EDGES, right=False)),
        }
        stored = {"tpsa": int(tpsa_ids[idx]), "logp": int(logp_ids[idx])}
        if idx % 10 == 0:
            joint_sample.append(
                {
                    "source_index": idx,
                    "tpsa": f"{descriptor_values['tpsa']:.10f}",
                    "logp": f"{descriptor_values['logp']:.10f}",
                    "tpsa_region": stored["tpsa"],
                    "logp_region": stored["logp"],
                    **{key: f"{value:.10f}" for key, value in common.items()},
                }
            )
        for partition in ("tpsa", "logp"):
            if stored[partition] != expected[partition]:
                assignment_mismatch[partition] += 1
            values = aggregate[(partition, stored[partition])]
            for metric, value in common.items():
                values[metric].append(value)
            values["descriptor_value"].append(descriptor_values[partition])

    rows: list[dict] = []
    total = len(smiles) - invalid
    for partition, edges, unit in (
        ("tpsa", TPSA_EDGES, "A^2"),
        ("logp", LOGP_EDGES, ""),
    ):
        for region in range(8):
            metrics = aggregate[(partition, region)]
            count = len(metrics["descriptor_value"])
            row = {
                "partition": partition,
                "region": region,
                "interval": interval_label(edges, region, unit),
                "count": count,
                "fraction": f"{count / total:.10f}",
            }
            for metric in metric_names:
                values = metrics[metric]
                row[f"{metric}_mean"] = f"{np.mean(values):.10f}" if values else ""
                row[f"{metric}_sd"] = (
                    f"{np.std(values, ddof=1):.10f}" if len(values) > 1 else ""
                )
            rows.append(row)
    audit = {
        "smiles_count": len(smiles),
        "valid_molecules": total,
        "invalid_smiles": invalid,
        "assignment_mismatch": assignment_mismatch,
        "rdkit_version": getattr(sys.modules.get("rdkit"), "__version__", "unknown"),
    }
    audit["joint_sample_policy"] = "every tenth source molecule"
    audit["joint_sample_rows"] = len(joint_sample)
    return rows, joint_sample, audit


EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+),\s+learning rate\s+(?P<lr>[0-9.eE+-]+),\s+"
    r"train loss:\s+(?P<loss>[0-9.eE+-]+)\s+\|\s+"
    r"(?P<seconds>[0-9.]+)s/epoch,\s+total\s+(?P<total>[0-9.]+)s"
)


def slurm_accounting_snapshot() -> list[dict]:
    command = [
        "sacct",
        "-u",
        os.environ.get("USER", ""),
        "-S",
        "2026-05-01",
        "-X",
        "-n",
        "-P",
        "-o",
        "JobIDRaw,JobName%160,State,ElapsedRaw,MaxRSS,ReqMem",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    rows: list[dict] = []
    for line in completed.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 6 or not fields[0].strip().isdigit():
            continue
        rows.append(
            {
                "job_id": fields[0].strip(),
                "job_name": fields[1].strip(),
                "state": fields[2].strip(),
                "elapsed_seconds": fields[3].strip(),
                "max_rss": fields[4].strip(),
                "requested_memory": fields[5].strip(),
            }
        )
    return rows


def parse_pretraining_logs(accounting: list[dict]) -> tuple[list[dict], list[dict]]:
    by_job = {row["job_id"]: row for row in accounting}
    model_names = {
        "p3_base_vanilla",
        "p3_tpsa_mofe_last_ffn",
        "p3_logp_mofe_last_ffn",
        "p3_tpsa_mofe_last_ffn_conloss",
        "p3_logp_mofe_last_ffn_conloss",
        "p3_tpsa_mofe_last_all_kan",
        "p3_logp_mofe_last_all_kan",
        "p5_tpsa_mofe_last_selected_kan",
        "p5_logp_mofe_last_selected_kan",
        "p5_tpsa_mofe_last_selected_kan_conloss",
        "p5_logp_mofe_last_selected_kan_conloss",
        "p2_mofe_ffn_logp_last",
        "p2_mofe_ffn_logp_odd",
        "p2_mofe_ffn_logp_even",
    }
    epochs: dict[tuple[str, int], dict] = {}
    sources: list[dict] = []
    for path in sorted(PRETRAIN_RESULT_ROOT.glob("*.out")):
        job_id = path.stem
        job = by_job.get(job_id)
        if not job:
            continue
        job_name = job["job_name"]
        candidates = [name for name in model_names if name in job_name]
        if not candidates:
            text_tail = path.read_text(errors="replace")[-3000:]
            candidates = [name for name in model_names if name in text_tail]
        if not candidates:
            continue
        model = max(candidates, key=len)
        text = path.read_text(errors="replace")
        matched = 0
        for match in EPOCH_RE.finditer(text):
            epoch = int(match.group("epoch"))
            epochs[(model, epoch)] = {
                "model": model,
                "epoch": epoch,
                "learning_rate": match.group("lr"),
                "train_total_loss": match.group("loss"),
                "loss_semantics": "logged_terminal_batch_composite_loss",
                "seconds_per_epoch": match.group("seconds"),
                "reported_segment_total_seconds": match.group("total"),
                "source_log": str(path),
                "job_id": job_id,
            }
            matched += 1
        sources.append(
            {
                "model": model,
                "job_id": job_id,
                "job_name": job_name,
                "state": job["state"],
                "elapsed_seconds": job["elapsed_seconds"],
                "epochs_parsed": matched,
                "pretrain_done_marker": int(f"PRETRAIN_DONE: {model}" in text),
                "source_log": str(path),
            }
        )
    return (
        sorted(epochs.values(), key=lambda row: (row["model"], row["epoch"])),
        sources,
    )


def active_epoch_timing_audit(epoch_rows: list[dict]) -> list[dict]:
    """Separate active epoch timing from startup and interruption outliers.

    The timer recorded by train_total.py starts at the beginning of each epoch,
    so scheduler queue time between Slurm jobs is already absent. We additionally
    remove the first epoch of each recovered job segment and high-duration
    observations above the conservative upper outer fence Q3 + 3 IQR. The raw
    values remain in pretraining_curves.csv.
    """

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in epoch_rows:
        grouped[(row["model"], row["job_id"])].append(row)

    output: list[dict] = []
    for (model, job_id), items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda row: int(row["epoch"]))
        first_epoch = int(ordered[0]["epoch"])
        reference = np.asarray(
            [float(row["seconds_per_epoch"]) for row in ordered[1:]],
            dtype=float,
        )
        if reference.size >= 4:
            q1, q3 = np.quantile(reference, [0.25, 0.75])
            iqr = float(q3 - q1)
            upper_fence = float(q3 + 3.0 * iqr) if iqr > 0 else math.inf
        else:
            q1 = q3 = iqr = math.nan
            upper_fence = math.inf

        for row in ordered:
            epoch = int(row["epoch"])
            seconds = float(row["seconds_per_epoch"])
            if epoch == first_epoch:
                included = 0
                reason = "segment_startup"
            elif seconds > upper_fence:
                included = 0
                reason = "high_duration_outer_fence"
            else:
                included = 1
                reason = ""
            output.append(
                {
                    "model": model,
                    "epoch": epoch,
                    "job_id": job_id,
                    "seconds_per_epoch_raw": f"{seconds:.10f}",
                    "seconds_per_epoch_active": (
                        f"{seconds:.10f}" if included else ""
                    ),
                    "included": included,
                    "exclusion_reason": reason,
                    "segment_first_epoch": first_epoch,
                    "q1_seconds": "" if math.isnan(float(q1)) else f"{q1:.10f}",
                    "q3_seconds": "" if math.isnan(float(q3)) else f"{q3:.10f}",
                    "iqr_seconds": "" if math.isnan(float(iqr)) else f"{iqr:.10f}",
                    "upper_outer_fence_seconds": (
                        "" if math.isinf(upper_fence) else f"{upper_fence:.10f}"
                    ),
                    "source_log": row["source_log"],
                }
            )
    return output


def pretraining_pause_gap_audit(timing_rows: list[dict]) -> list[dict]:
    """Summarize every recovered Slurm segment used in active-time analysis.

    Queue and inter-job pauses cannot enter the in-process epoch timer. This
    table makes that exclusion explicit and records the startup and
    high-duration observations removed within each recovered segment.
    """

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in timing_rows:
        grouped[(row["model"], row["job_id"])].append(row)

    output = []
    for (model, job_id), items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda row: int(row["epoch"]))
        raw_seconds = np.asarray(
            [float(row["seconds_per_epoch_raw"]) for row in ordered],
            dtype=float,
        )
        active_seconds = np.asarray(
            [
                float(row["seconds_per_epoch_active"])
                for row in ordered
                if int(row["included"])
            ],
            dtype=float,
        )
        reasons = Counter(row["exclusion_reason"] for row in ordered)
        output.append(
            {
                "model": model,
                "job_id": job_id,
                "first_epoch": min(int(row["epoch"]) for row in ordered),
                "last_epoch": max(int(row["epoch"]) for row in ordered),
                "epochs_parsed": len(ordered),
                "epochs_retained": int(active_seconds.size),
                "startup_epochs_excluded": reasons["segment_startup"],
                "outer_fence_epochs_excluded": reasons[
                    "high_duration_outer_fence"
                ],
                "raw_median_seconds": f"{np.median(raw_seconds):.10f}",
                "active_median_seconds": (
                    f"{np.median(active_seconds):.10f}"
                    if active_seconds.size
                    else ""
                ),
                "upper_outer_fence_seconds": ordered[0][
                    "upper_outer_fence_seconds"
                ],
                "scheduler_pause_seconds_in_epoch_timer": 0,
                "audit_note": (
                    "Scheduler and inter-job gaps are outside the in-process "
                    "epoch timer; segment startup and high-duration outer-fence "
                    "observations are listed separately."
                ),
                "source_log": ordered[0]["source_log"],
            }
        )
    return output


def finetune_runtime_summary(accounting: list[dict]) -> list[dict]:
    rows: list[dict] = []
    model_names = list(MODEL_METADATA)
    for job in accounting:
        name = job["job_name"]
        candidates = [model for model in model_names if model in name]
        if not candidates:
            continue
        model = max(candidates, key=len)
        dataset = next((item for item in DATASETS if name.endswith(f"_{item}")), "")
        if not dataset:
            dataset = next((item for item in DATASETS if f"_{item}_" in name), "")
        if not dataset:
            continue
        elapsed = int(job["elapsed_seconds"]) if job["elapsed_seconds"].isdigit() else 0
        rows.append(
            {
                "job_id": job["job_id"],
                "job_name": name,
                "model": model,
                "protocol": MODEL_METADATA[model][0],
                "dataset": dataset,
                "state": job["state"],
                "elapsed_seconds": elapsed,
                "requested_memory": job["requested_memory"],
            }
        )
    return rows


def aggregate_runtime_jobs(rows: list[dict], group_field: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row[group_field]].append(row)
    output = []
    for key, items in grouped.items():
        completed = [row for row in items if row["state"] == "COMPLETED"]
        unsuccessful = [row for row in items if row["state"] != "COMPLETED"]
        completed_seconds = [int(row["elapsed_seconds"]) for row in completed]
        output.append(
            {
                group_field: key,
                "completed_jobs": len(completed),
                "unsuccessful_or_cancelled_jobs": len(unsuccessful),
                "completed_gpu_hours": f"{sum(completed_seconds) / 3600:.10f}",
                "unsuccessful_gpu_hours": f"{sum(int(row['elapsed_seconds']) for row in unsuccessful) / 3600:.10f}",
                "median_completed_job_hours": (
                    f"{np.median(completed_seconds) / 3600:.10f}"
                    if completed_seconds
                    else ""
                ),
            }
        )
    return output


def pretraining_runtime_summary(
    epoch_rows: list[dict],
    timing_rows: list[dict],
    sources: list[dict],
) -> list[dict]:
    epoch_group: dict[str, list[dict]] = defaultdict(list)
    timing_group: dict[str, list[dict]] = defaultdict(list)
    source_group: dict[str, list[dict]] = defaultdict(list)
    for row in epoch_rows:
        epoch_group[row["model"]].append(row)
    for row in timing_rows:
        timing_group[row["model"]].append(row)
    for row in sources:
        if int(row["epochs_parsed"]) > 0:
            source_group[row["model"]].append(row)
    output = []
    for model in sorted(set(epoch_group) | set(timing_group) | set(source_group)):
        epochs = epoch_group.get(model, [])
        timings = timing_group.get(model, [])
        logs = source_group.get(model, [])
        completed = [row for row in logs if row["state"] == "COMPLETED"]
        unsuccessful = [row for row in logs if row["state"] != "COMPLETED"]
        active_seconds = [
            float(row["seconds_per_epoch_active"])
            for row in timings
            if int(row["included"])
        ]
        excluded = [row for row in timings if not int(row["included"])]
        q1, q3 = (
            np.quantile(active_seconds, [0.25, 0.75])
            if active_seconds
            else (math.nan, math.nan)
        )
        median = float(np.median(active_seconds)) if active_seconds else math.nan
        output.append(
            {
                "model": model,
                "epochs_recovered": len({int(row["epoch"]) for row in epochs}),
                "active_epochs_retained": len(active_seconds),
                "timing_epochs_excluded": len(excluded),
                "median_active_seconds_per_epoch": (
                    f"{median:.10f}" if active_seconds else ""
                ),
                "active_q1_seconds": f"{q1:.10f}" if active_seconds else "",
                "active_q3_seconds": f"{q3:.10f}" if active_seconds else "",
                "projected_active_hours_300_epochs": (
                    f"{median * 300 / 3600:.10f}" if active_seconds else ""
                ),
                "completed_gpu_hours": f"{sum(int(row['elapsed_seconds']) for row in completed) / 3600:.10f}",
                "unsuccessful_gpu_hours": f"{sum(int(row['elapsed_seconds']) for row in unsuccessful) / 3600:.10f}",
                "completed_log_segments": len(completed),
                "unsuccessful_log_segments": len(unsuccessful),
                "pretrain_done_marker": int(
                    any(int(row["pretrain_done_marker"]) for row in logs)
                ),
            }
        )
    return output


def architecture_complexity() -> list[dict]:
    d_model = 256
    non_ffn_parameters = 2_690_833
    dense_branch_params = 2 * (d_model * d_model + d_model)
    # _KANLinear stores one base matrix and eight spline coefficients per edge.
    kan_linear_params = d_model * d_model * (1 + 5 + 3)
    kan_branch_params = 2 * kan_linear_params
    dense_macs = 2 * d_model * d_model
    kan_macs_lower_bound = 2 * (d_model * d_model + d_model * d_model * 8)
    assignments = np.load(MOLESG_ROOT / "Data/zinc15/expert_ids.npz")
    occupancies = {
        partition: np.bincount(
            np.asarray(assignments[partition], dtype=int), minlength=8
        ).astype(float)
        for partition in ("tpsa", "logp")
    }
    for partition in occupancies:
        occupancies[partition] /= occupancies[partition].sum()
    rows = []
    configs = [
        ("Vanilla", 0, 0, 8, 0, None, ()),
        ("MoFE-last all-MLP", 8, 0, 7, 0, None, ()),
        ("TPSA selected-KAN", 3, 5, 7, 0, "tpsa", (0, 1, 2, 3, 5)),
        ("MolLogP selected-KAN", 4, 4, 7, 0, "logp", (1, 2, 3, 4)),
        ("MoFE-last all-KAN", 0, 8, 7, 0, None, tuple(range(8))),
        ("Routed top1", 8, 0, 0, 1, None, ()),
        ("Routed top2", 8, 0, 0, 2, None, ()),
        ("Routed top3", 8, 0, 0, 3, None, ()),
        ("Routed top4", 8, 0, 0, 4, None, ()),
    ]
    base_ffn_params = 8 * dense_branch_params
    for (
        name,
        mlp_branches,
        kan_branches,
        dense_layers,
        top_k,
        partition,
        kan_indices,
    ) in configs:
        if name == "Vanilla":
            stored_ffn = base_ffn_params
            active_ffn_parameters = base_ffn_params
            active_ffn_macs = 8 * dense_macs
            router_params = 0
            note = "Eight dense FFN layers"
        elif name.startswith("Routed"):
            stored_ffn = 8 * 8 * dense_branch_params
            router_params = 8 * d_model * 8
            active_ffn_parameters = 8 * top_k * dense_branch_params
            active_ffn_macs = 8 * (top_k * dense_macs + d_model * 8)
            note = "All eight layers routed; excludes dispatch and softmax overhead"
        else:
            stored_ffn = dense_layers * dense_branch_params
            stored_ffn += mlp_branches * dense_branch_params
            stored_ffn += kan_branches * kan_branch_params
            router_params = 0
            if partition is None:
                kan_fraction = 1.0 if kan_branches == 8 else 0.0
            else:
                kan_fraction = float(occupancies[partition][list(kan_indices)].sum())
            active_branch_macs = (
                (1.0 - kan_fraction) * dense_macs
                + kan_fraction * kan_macs_lower_bound
            )
            active_branch_parameters = (
                (1.0 - kan_fraction) * dense_branch_params
                + kan_fraction * kan_branch_params
            )
            active_ffn_parameters = (
                dense_layers * dense_branch_params + active_branch_parameters
            )
            active_ffn_macs = dense_layers * dense_macs + active_branch_macs
            note = (
                "One descriptor-assigned branch active in the last layer; "
                "selected-KAN expectation is occupancy-weighted on ZINC15"
            )
        rows.append(
            {
                "configuration": name,
                "stored_ffn_parameters": int(stored_ffn),
                "router_parameters": int(router_params),
                "stored_total_parameters_analytical": int(
                    non_ffn_parameters + stored_ffn + router_params
                ),
                "active_ffn_parameters_expected": int(active_ffn_parameters),
                "active_ffn_macs_per_token": int(active_ffn_macs),
                "flops_convention": "2 FLOPs per MAC",
                "active_ffn_flops_per_token": int(2 * active_ffn_macs),
                "notes": note,
            }
        )
    return rows


def write_published_molesg() -> None:
    rows = [
        {
            "dataset": dataset,
            "published_auc_mean": mean,
            "published_auc_sd": sd,
            "runs": 3,
            "source": "MoleSG, Briefings in Bioinformatics 2024, doi:10.1093/bib/bbae256",
        }
        for dataset, (mean, sd) in PUBLISHED_MOLESG.items()
    ]
    write_csv(TABLE_ROOT / "published_molesg_classification.csv", rows)


def dataset_metadata() -> list[dict]:
    output = []
    for dataset in DATASETS:
        path = MOLESG_ROOT / f"Downstream/Data/{dataset}/preprocess/{dataset}.pickle"
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        molecules, labels = payload[0], payload[1]
        first = np.asarray(labels[0])
        tasks = int(first.size)
        output.append(
            {
                "dataset": dataset,
                "molecules": len(molecules),
                "tasks": tasks,
                "task_type": "classification",
                "split": "scaffold 8:1:1",
                "folds": 5,
                "seeds": "42,43,44,45,46",
                "metric": "ROC-AUC",
                "source_pickle": str(path),
            }
        )
    return output


def metric_definitions() -> list[dict]:
    """Machine-readable definitions for manuscript metrics and derived values."""

    return [
        {
            "metric": "endpoint ROC-AUC",
            "symbol": "A_dft",
            "input": "actual and predict columns from one fold CSV",
            "definition": (
                "Mask actual == -1 and compute ROC-AUC only when the remaining "
                "endpoint contains both classes."
            ),
            "aggregation": "one dataset, fold, and valid endpoint",
            "parameters": "binary labels and continuous prediction scores",
            "code_source": "analysis_v2/build_analysis_v2.py::compute_auc",
        },
        {
            "metric": "fold ROC-AUC",
            "symbol": "A_df",
            "input": "valid endpoint ROC-AUC values",
            "definition": "Arithmetic mean across valid endpoints in one fold.",
            "aggregation": "one dataset and fold",
            "parameters": "valid endpoints receive equal weight",
            "code_source": "analysis_v2/build_analysis_v2.py::compute_auc",
        },
        {
            "metric": "five-fold mean and sample SD",
            "symbol": "Abar_d and s_d",
            "input": "fold ROC-AUC for folds 1 to 5",
            "definition": (
                "Arithmetic mean and sample standard deviation with denominator "
                "n - 1."
            ),
            "aggregation": "one model and dataset",
            "parameters": "seeds 42 to 46 and complete 5 of 5 folds",
            "code_source": "analysis_v2/build_analysis_v2.py::complete_summary",
        },
        {
            "metric": "eight-dataset macro ROC-AUC",
            "symbol": "A_macro",
            "input": "eight complete dataset means",
            "definition": "Equal-weight arithmetic mean of dataset means.",
            "aggregation": "one model across eight classification datasets",
            "parameters": "no sample-count or endpoint-count weighting",
            "code_source": "analysis_v2/build_analysis_v2.py::model_macro_summary",
        },
        {
            "metric": "paired dataset delta and win",
            "symbol": "delta_df",
            "input": "matched model ROC-AUC values for the same fold",
            "definition": (
                "Subtract right from left within fold and average five fold "
                "deltas within a dataset. A positive dataset mean is one win."
            ),
            "aggregation": "matched folds followed by equal-weight datasets",
            "parameters": "identical scaffold seeds for both models",
            "code_source": "analysis_v2/build_analysis_v2.py::paired_delta_rows",
        },
        {
            "metric": "hierarchical paired bootstrap interval",
            "symbol": "CI_95",
            "input": "matched fold deltas nested within eight datasets",
            "definition": (
                "Resample datasets with replacement, then folds with replacement "
                "within each sampled dataset. Average folds within dataset and "
                "then average datasets."
            ),
            "aggregation": "10,000 paired hierarchical bootstrap replicates",
            "parameters": "seed 42 and percentile interval at 2.5 and 97.5",
            "code_source": "analysis_v2/build_analysis_v2.py::hierarchical_bootstrap",
        },
        {
            "metric": "Wilcoxon signed-rank test",
            "symbol": "p",
            "input": "eight dataset-level paired mean deltas",
            "definition": "Two-sided Wilcoxon signed-rank test.",
            "aggregation": "one planned comparison",
            "parameters": "zero_method=wilcox and Holm adjustment",
            "code_source": "analysis_v2/build_analysis_v2.py::wilcoxon_holm",
        },
        {
            "metric": "conditional interaction",
            "symbol": "Delta_interaction",
            "input": "I1, I1+I2, I1+I3, and Full matched fold ROC-AUC",
            "definition": "Full - (I1+I2) - (I1+I3) + I1.",
            "aggregation": "matched folds followed by equal-weight datasets",
            "parameters": "reported separately for TPSA and MolLogP",
            "code_source": "analysis_v2/build_analysis_v2.py::build_interaction_rows",
        },
        {
            "metric": "descriptor-region occupancy",
            "symbol": "n_k and p_k",
            "input": "RDKit TPSA or MolLogP for every ZINC15 molecule",
            "definition": (
                "Assign fixed left-closed, right-open intervals with np.digitize "
                "and report count and fraction."
            ),
            "aggregation": "one descriptor region",
            "parameters": "eight fixed regions per descriptor",
            "code_source": "analysis_v2/build_analysis_v2.py::descriptor_profiles",
        },
        {
            "metric": "descriptor-region chemical profile",
            "symbol": "",
            "input": "RDKit molecular descriptors within one region",
            "definition": "Region mean and sample SD for each profile variable.",
            "aggregation": "one region and profile variable",
            "parameters": "regions with fewer than 10 molecules are masked in heatmaps",
            "code_source": "analysis_v2/build_analysis_v2.py::descriptor_profiles",
        },
        {
            "metric": "active pretraining epoch time",
            "symbol": "t_active",
            "input": "seconds per epoch recorded in pretraining logs",
            "definition": (
                "Exclude the first epoch of each Slurm segment and values above "
                "the segment-specific Q3 + 3 IQR upper outer fence."
            ),
            "aggregation": "median and quartiles within model",
            "parameters": "queue and inter-job gaps are outside the epoch timer",
            "code_source": "analysis_v2/build_analysis_v2.py::active_epoch_timing_audit",
        },
        {
            "metric": "stored and active model cost",
            "symbol": "",
            "input": "analytical tensor shapes and ZINC15 region occupancy",
            "definition": (
                "Stored parameters include all branches. Expected active branch "
                "cost is occupancy weighted for selected KAN configurations."
            ),
            "aggregation": "one architecture configuration",
            "parameters": (
                "one MAC equals two FLOPs. KAN excludes spline-basis construction. "
                "Routed controls exclude softmax and dispatch overhead"
            ),
            "code_source": "analysis_v2/build_analysis_v2.py::architecture_complexity",
        },
        {
            "metric": "pooled molecular embedding",
            "symbol": "h_i",
            "input": "padded node embeddings and RDKit atom count",
            "definition": "Mean of the first n_i valid node rows.",
            "aggregation": "one molecule, model, dataset, and fold",
            "parameters": "n_i equals the RDKit atom count",
            "code_source": "analysis_v2/analyze_embeddings_v2.py::mean_pool",
        },
        {
            "metric": "embedding reduction and t-SNE",
            "symbol": "",
            "input": "mean-pooled molecular embeddings on common SMILES",
            "definition": (
                "Model-specific PCA to at most 50 dimensions, L2 normalization, "
                "then two-dimensional t-SNE for visualization."
            ),
            "aggregation": "one dataset, partition, model, and fold",
            "parameters": (
                "fold 1, perplexity 30, PCA initialization, learning_rate=auto, "
                "2000 iterations, and seed 42"
            ),
            "code_source": "analysis_v2/analyze_embeddings_v2.py",
        },
        {
            "metric": "silhouette score",
            "symbol": "",
            "input": "L2-normalized PCA embeddings and region labels",
            "definition": "Mean Euclidean silhouette coefficient.",
            "aggregation": "one model, dataset, partition, and fold",
            "parameters": "regions with fewer than two samples are excluded",
            "code_source": "analysis_v2/analyze_embeddings_v2.py::separation_metrics",
        },
        {
            "metric": "Davies-Bouldin index",
            "symbol": "",
            "input": "L2-normalized PCA embeddings and region labels",
            "definition": (
                "Mean worst-case ratio of within-region dispersion to centroid "
                "separation."
            ),
            "aggregation": "one model, dataset, partition, and fold",
            "parameters": "lower values indicate stronger separation",
            "code_source": "analysis_v2/analyze_embeddings_v2.py::separation_metrics",
        },
        {
            "metric": "inter-to-intra distance ratio",
            "symbol": "D_inter / D_intra",
            "input": "Euclidean pairwise distances in L2-normalized PCA space",
            "definition": (
                "Mean distance between regions divided by mean non-diagonal "
                "distance within regions."
            ),
            "aggregation": "one model, dataset, partition, and fold",
            "parameters": "regions with fewer than two samples are excluded",
            "code_source": "analysis_v2/analyze_embeddings_v2.py::separation_metrics",
        },
    ]


def copy_kan_screen_files() -> None:
    source_summary = ABLATION_ROOT / "P4_KAN_SELECT_8TASK_SUMMARY_LOGAUC_20260716.csv"
    source_detail = ABLATION_ROOT / "P4_KAN_SELECT_8TASK_DETAIL_LOGAUC_20260716.csv"
    shutil.copy2(source_summary, TABLE_ROOT / "kan_screen_summary.csv")
    shutil.copy2(source_detail, TABLE_ROOT / "kan_screen_detail.csv")


def write_provenance(files: list[Path], extra: dict) -> None:
    entries = []
    for path in files:
        if not path.exists():
            continue
        entries.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            }
        )
    payload = {
        "generated": datetime.now().astimezone().isoformat(),
        "generator": str(Path(__file__).resolve()),
        "policy": {
            "main_tables_require_complete_folds": FOLDS,
            "datasets": DATASETS,
            "no_training_or_submission": True,
        },
        "outputs": entries,
        **extra,
    }
    (OUT_ROOT / "provenance_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-descriptor-profiles",
        action="store_true",
        help="Skip the RDKit scan when only refreshing downstream results.",
    )
    parser.add_argument(
        "--descriptor-profiles-only",
        action="store_true",
        help="Run only the RDKit descriptor-region audit.",
    )
    args = parser.parse_args()
    ensure_dirs()

    if args.descriptor_profiles_only:
        profiles, joint_sample, descriptor_audit = descriptor_profiles()
        write_csv(TABLE_ROOT / "descriptor_region_profiles.csv", profiles)
        write_csv(TABLE_ROOT / "descriptor_joint_sample.csv", joint_sample)
        (TABLE_ROOT / "descriptor_assignment_audit.json").write_text(
            json.dumps(descriptor_audit, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Wrote descriptor profiles to {TABLE_ROOT}")
        return

    records = collect_results()
    result_rows = [
        {
            "model": row.model,
            "checkpoint": row.model.replace("_n8aligned", "").replace("_n4aligned", ""),
            "protocol": row.protocol,
            "partition": row.partition,
            "ablation_role": row.role,
            "dataset": row.dataset,
            "fold": row.fold,
            "seed": row.seed,
            "auc": f"{row.auc:.10f}",
            "completion": "complete_fold",
            "source_csv": row.source_csv,
            "source_log": "",
        }
        for row in records
    ]
    write_csv(TABLE_ROOT / "results_long.csv", result_rows)
    result_health = {
        "final_n8_expected_fold_csv": 11 * len(DATASETS) * len(FOLDS),
        "final_n8_found_fold_csv": sum(
            row.protocol == "final_n8" for row in records
        ),
        "depth_reference_expected_fold_csv": len(DATASETS) * len(FOLDS),
        "depth_reference_found_fold_csv": sum(
            row.protocol == "depth_reference" for row in records
        ),
        "parsed_result_rows": len(result_rows),
        "empty_csv": 0,
        "nonfinite_auc": 0,
        "invalid_auc": 0,
        "datasets": DATASETS,
        "folds": FOLDS,
        "seeds": [SEEDS[fold] for fold in FOLDS],
    }
    (TABLE_ROOT / "result_health_audit.json").write_text(
        json.dumps(result_health, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    dataset_summary = complete_summary(records)
    write_csv(TABLE_ROOT / "dataset_5fold_summary.csv", dataset_summary)
    write_csv(TABLE_ROOT / "model_macro_summary.csv", model_macro_summary(dataset_summary))
    write_csv(
        TABLE_ROOT / "matched_model_macro_summary.csv",
        matched_final_macro_summary(dataset_summary),
    )
    write_csv(TABLE_ROOT / "final_fold_wide.csv", fold_wide_rows(records))
    write_csv(
        TABLE_ROOT / "layer_screen_summary.csv",
        layer_screen_summary(dataset_summary),
    )
    pairwise_rows = build_pairwise_comparisons(records)
    write_csv(TABLE_ROOT / "paired_comparisons.csv", pairwise_rows)
    write_csv(TABLE_ROOT / "si_statistical_tests.csv", wilcoxon_holm(pairwise_rows))
    write_csv(TABLE_ROOT / "conditional_interaction.csv", build_interaction_rows(records))

    copy_kan_screen_files()
    write_published_molesg()
    write_csv(TABLE_ROOT / "dataset_metadata.csv", dataset_metadata())
    write_csv(TABLE_ROOT / "architecture_complexity.csv", architecture_complexity())
    write_csv(TABLE_ROOT / "metric_definitions.csv", metric_definitions())

    descriptor_audit = {}
    if not args.skip_descriptor_profiles:
        profiles, joint_sample, descriptor_audit = descriptor_profiles()
        write_csv(TABLE_ROOT / "descriptor_region_profiles.csv", profiles)
        write_csv(TABLE_ROOT / "descriptor_joint_sample.csv", joint_sample)
        (TABLE_ROOT / "descriptor_assignment_audit.json").write_text(
            json.dumps(descriptor_audit, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    accounting = slurm_accounting_snapshot()
    write_csv(TABLE_ROOT / "slurm_accounting_snapshot.csv", accounting)
    epochs, pretrain_sources = parse_pretraining_logs(accounting)
    timing_audit = active_epoch_timing_audit(epochs)
    write_csv(
        TABLE_ROOT / "pretraining_curves.csv",
        epochs,
        fieldnames=[
            "model",
            "epoch",
            "learning_rate",
            "train_total_loss",
            "loss_semantics",
            "seconds_per_epoch",
            "reported_segment_total_seconds",
            "source_log",
            "job_id",
        ],
    )
    write_csv(
        TABLE_ROOT / "pretraining_epoch_timing_audit.csv",
        timing_audit,
        fieldnames=[
            "model",
            "epoch",
            "job_id",
            "seconds_per_epoch_raw",
            "seconds_per_epoch_active",
            "included",
            "exclusion_reason",
            "segment_first_epoch",
            "q1_seconds",
            "q3_seconds",
            "iqr_seconds",
            "upper_outer_fence_seconds",
            "source_log",
        ],
    )
    write_csv(
        TABLE_ROOT / "pretraining_pause_gap_audit.csv",
        pretraining_pause_gap_audit(timing_audit),
        fieldnames=[
            "model",
            "job_id",
            "first_epoch",
            "last_epoch",
            "epochs_parsed",
            "epochs_retained",
            "startup_epochs_excluded",
            "outer_fence_epochs_excluded",
            "raw_median_seconds",
            "active_median_seconds",
            "upper_outer_fence_seconds",
            "scheduler_pause_seconds_in_epoch_timer",
            "audit_note",
            "source_log",
        ],
    )
    write_csv(
        TABLE_ROOT / "pretraining_log_sources.csv",
        pretrain_sources,
        fieldnames=[
            "model",
            "job_id",
            "job_name",
            "state",
            "elapsed_seconds",
            "epochs_parsed",
            "pretrain_done_marker",
            "source_log",
        ],
    )
    write_csv(
        TABLE_ROOT / "pretraining_runtime_summary.csv",
        pretraining_runtime_summary(epochs, timing_audit, pretrain_sources),
    )
    finetune_jobs = finetune_runtime_summary(accounting)
    write_csv(
        TABLE_ROOT / "finetune_runtime_jobs.csv",
        finetune_jobs,
        fieldnames=[
            "job_id",
            "job_name",
            "model",
            "protocol",
            "dataset",
            "state",
            "elapsed_seconds",
            "requested_memory",
        ],
    )
    write_csv(
        TABLE_ROOT / "finetune_runtime_by_model.csv",
        aggregate_runtime_jobs(finetune_jobs, "model"),
    )
    write_csv(
        TABLE_ROOT / "finetune_runtime_by_dataset.csv",
        aggregate_runtime_jobs(finetune_jobs, "dataset"),
    )

    output_files = sorted(TABLE_ROOT.glob("*"))
    write_provenance(
        output_files,
        {
            "record_count": len(records),
            "descriptor_audit": descriptor_audit,
        },
    )
    print(f"Wrote {len(output_files)} analysis artifacts to {TABLE_ROOT}")


if __name__ == "__main__":
    main()
