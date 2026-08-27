#!/usr/bin/env python3
"""Analyze existing N8 embeddings without running model inference.

The script uses one common SMILES set for every model within a
dataset/partition/fold comparison. Quantitative separation metrics are
calculated in model-specific PCA spaces. t-SNE coordinates are generated only
for fold 1 and are used for visualization.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, rdMolDescriptors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


OUT_ROOT = Path(__file__).resolve().parents[1]
MOLESG_ROOT = Path(os.environ.get("MOLESG_ROOT", OUT_ROOT / "MoleSG"))
RESULT_ROOT = Path(
    os.environ.get("MOLESG_RESULT_ROOT", MOLESG_ROOT / "Downstream" / "Result")
)
TABLE_ROOT = OUT_ROOT / "tables_v2"
TABLE_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = ["bbbp", "tox21", "toxcast", "sider", "clintox", "bace", "hiv", "muv"]
FOLDS = [1, 2, 3, 4, 5]
LARGE_DATASETS = {"hiv", "muv"}
MAX_LARGE_SAMPLE = 2000
TPSA_EDGES = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]
LOGP_EDGES = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

MODEL_GROUPS = {
    "tpsa": {
        "Base": "p3_base_vanilla_n8aligned",
        "I1": "p3_tpsa_mofe_last_ffn_n8aligned",
        "I1+I2": "p5_tpsa_mofe_last_selected_kan_n8aligned",
        "I1+I3": "p3_tpsa_mofe_last_ffn_conloss_n8aligned",
        "Full": "p5_tpsa_mofe_last_selected_kan_conloss_n8aligned",
    },
    "logp": {
        "Base": "p3_base_vanilla_n8aligned",
        "I1": "p3_logp_mofe_last_ffn_n8aligned",
        "I1+I2": "p5_logp_mofe_last_selected_kan_n8aligned",
        "I1+I3": "p3_logp_mofe_last_ffn_conloss_n8aligned",
        "Full": "p5_logp_mofe_last_selected_kan_conloss_n8aligned",
    },
}

RDLogger.DisableLog("rdApp.warning")


def embedding_path(model: str, dataset: str, fold: int) -> Path:
    return (
        RESULT_ROOT
        / dataset
        / f"total_test_embedding_{model}_{dataset}_fold_{fold}.pickle"
    )


def load_embedding(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"expected dict in {path}")
    return data


def embedding_keys(path: Path) -> set[str]:
    data = load_embedding(path)
    keys = set(data)
    del data
    gc.collect()
    return keys


def mean_pool(smiles: str, padded: np.ndarray) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES in embedding file: {smiles}")
    atom_count = mol.GetNumAtoms()
    if padded.ndim != 2 or atom_count > padded.shape[0]:
        raise ValueError(f"unexpected embedding shape for {smiles}: {padded.shape}")
    return np.asarray(padded[:atom_count], dtype=np.float64).mean(axis=0)


def pooled_matrix(path: Path, smiles: list[str]) -> np.ndarray:
    data = load_embedding(path)
    try:
        return np.stack([mean_pool(text, data[text]) for text in smiles])
    finally:
        del data
        gc.collect()


def assignments(smiles: list[str], partition: str) -> np.ndarray:
    edges = TPSA_EDGES if partition == "tpsa" else LOGP_EDGES
    values = []
    for text in smiles:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            raise ValueError(f"invalid SMILES while assigning regions: {text}")
        if partition == "tpsa":
            value = rdMolDescriptors.CalcTPSA(mol)
        else:
            value = Crippen.MolLogP(mol)
        values.append(value)
    return np.digitize(np.asarray(values), edges, right=False).astype(int)


def stratified_sample(
    smiles: list[str],
    labels: np.ndarray,
    limit: int,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    if len(smiles) <= limit:
        return smiles, labels

    rng = np.random.default_rng(seed)
    groups = {
        region: np.flatnonzero(labels == region)
        for region in sorted(np.unique(labels))
    }
    raw_targets = {
        region: len(indices) * limit / len(smiles)
        for region, indices in groups.items()
    }
    targets = {
        region: min(len(groups[region]), max(1, int(math.floor(raw_targets[region]))))
        for region in groups
    }
    while sum(targets.values()) < limit:
        candidates = [
            region
            for region in groups
            if targets[region] < len(groups[region])
        ]
        if not candidates:
            break
        region = max(
            candidates,
            key=lambda item: (raw_targets[item] - targets[item], len(groups[item])),
        )
        targets[region] += 1
    while sum(targets.values()) > limit:
        candidates = [region for region in groups if targets[region] > 1]
        region = min(
            candidates,
            key=lambda item: (raw_targets[item] - targets[item], -targets[item]),
        )
        targets[region] -= 1

    selected = []
    for region, indices in groups.items():
        chosen = rng.choice(indices, size=targets[region], replace=False)
        selected.extend(int(index) for index in chosen)
    selected = sorted(selected)
    return [smiles[index] for index in selected], labels[selected]


def separation_metrics(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    unique, counts = np.unique(labels, return_counts=True)
    valid_regions = unique[counts >= 2]
    keep = np.isin(labels, valid_regions)
    x = x[keep]
    labels = labels[keep]
    if len(np.unique(labels)) < 2:
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "intra_distance": float("nan"),
            "inter_distance": float("nan"),
            "inter_intra_ratio": float("nan"),
        }
    distance = pairwise_distances(x, metric="euclidean")
    same = labels[:, None] == labels[None, :]
    diagonal = np.eye(len(labels), dtype=bool)
    intra = float(distance[same & ~diagonal].mean())
    inter = float(distance[~same].mean())
    return {
        "silhouette": float(silhouette_score(x, labels, metric="euclidean")),
        "davies_bouldin": float(davies_bouldin_score(x, labels)),
        "intra_distance": intra,
        "inter_distance": inter,
        "inter_intra_ratio": inter / intra if intra else float("nan"),
    }


def write_csv(path: Path, output: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)


def summarize_metrics(metric_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in metric_rows:
        grouped[
            (
                row["partition"],
                row["model_role"],
                row["model"],
                row["dataset"],
            )
        ].append(row)

    metrics = [
        "silhouette",
        "davies_bouldin",
        "intra_distance",
        "inter_distance",
        "inter_intra_ratio",
    ]
    output = []
    for partition, models in MODEL_GROUPS.items():
        for role, model in models.items():
            for dataset in DATASETS:
                rows = sorted(
                    grouped.get((partition, role, model, dataset), []),
                    key=lambda row: int(row["fold"]),
                )
                complete = [int(row["fold"]) for row in rows] == FOLDS
                record = {
                    "partition": partition,
                    "model_role": role,
                    "model": model,
                    "dataset": dataset,
                    "folds_found": len(rows),
                    "fold_ids": ",".join(str(row["fold"]) for row in rows),
                    "complete_5fold": int(complete),
                    "common_molecules_min": (
                        min(int(row["common_molecules"]) for row in rows)
                        if rows
                        else ""
                    ),
                }
                for metric in metrics:
                    values = np.asarray(
                        [float(row[metric]) for row in rows],
                        dtype=float,
                    )
                    record[f"{metric}_mean"] = (
                        f"{np.nanmean(values):.10f}" if complete else ""
                    )
                    record[f"{metric}_sd"] = (
                        f"{np.nanstd(values, ddof=1):.10f}" if complete else ""
                    )
                output.append(record)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsne-only",
        action="store_true",
        help=(
            "Reuse an existing complete embedding_separation_metrics.csv and "
            "recompute only fold-1 t-SNE coordinates."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_path = TABLE_ROOT / "embedding_separation_metrics.csv"
    if args.tsne_only:
        if not metric_path.exists():
            raise FileNotFoundError(
                "--tsne-only requires embedding_separation_metrics.csv"
            )
        with metric_path.open(newline="", encoding="utf-8") as handle:
            output_metrics = list(csv.DictReader(handle))
    else:
        output_metrics = []
    output_tsne: list[dict] = []
    audit_rows = []

    for dataset_index, dataset in enumerate(DATASETS):
        for partition_index, (partition, models) in enumerate(MODEL_GROUPS.items()):
            for fold in FOLDS:
                paths = {
                    role: embedding_path(model, dataset, fold)
                    for role, model in models.items()
                }
                missing = [str(path) for path in paths.values() if not path.exists()]
                if missing:
                    audit_rows.append(
                        {
                            "dataset": dataset,
                            "partition": partition,
                            "fold": fold,
                            "status": "missing_embedding",
                            "details": missing,
                        }
                    )
                    continue

                common = sorted(
                    set.intersection(*(embedding_keys(path) for path in paths.values()))
                )
                if not common:
                    raise RuntimeError(
                        f"no common SMILES for {dataset}/{partition}/fold{fold}"
                    )
                region_ids = assignments(common, partition)
                sample_policy = "all_common"
                if dataset in LARGE_DATASETS:
                    common, region_ids = stratified_sample(
                        common,
                        region_ids,
                        MAX_LARGE_SAMPLE,
                        seed=42 + dataset_index * 10 + partition_index,
                    )
                    sample_policy = f"region_stratified_max_{MAX_LARGE_SAMPLE}"

                occupancy = {
                    int(region): int(count)
                    for region, count in zip(
                        *np.unique(region_ids, return_counts=True)
                    )
                }
                audit_rows.append(
                    {
                        "dataset": dataset,
                        "partition": partition,
                        "fold": fold,
                        "status": "analyzed",
                        "common_molecules": len(common),
                        "sample_policy": sample_policy,
                        "occupancy": occupancy,
                    }
                )

                if args.tsne_only and fold != 1:
                    continue

                for role, model in models.items():
                    matrix = pooled_matrix(paths[role], common)
                    n_components = min(50, matrix.shape[0] - 1, matrix.shape[1])
                    pca = PCA(n_components=n_components, random_state=42)
                    x_pca = normalize(pca.fit_transform(matrix), norm="l2")
                    if not args.tsne_only:
                        metrics = separation_metrics(x_pca, region_ids)
                        output_metrics.append(
                            {
                                "partition": partition,
                                "model_role": role,
                                "model": model,
                                "dataset": dataset,
                                "fold": fold,
                                "common_molecules": len(common),
                                "sample_policy": sample_policy,
                                "pca_components": n_components,
                                **{
                                    key: f"{value:.10f}"
                                    for key, value in metrics.items()
                                },
                            }
                        )

                    if fold == 1:
                        tsne = TSNE(
                            n_components=2,
                            perplexity=30,
                            init="pca",
                            learning_rate="auto",
                            n_iter=2000,
                            random_state=42,
                        ).fit_transform(x_pca)
                        for index, text in enumerate(common):
                            output_tsne.append(
                                {
                                    "partition": partition,
                                    "model_role": role,
                                    "model": model,
                                    "dataset": dataset,
                                    "fold": fold,
                                    "smiles": text,
                                    "region": int(region_ids[index]),
                                    "tsne_x": f"{tsne[index, 0]:.10f}",
                                    "tsne_y": f"{tsne[index, 1]:.10f}",
                                    "sample_policy": sample_policy,
                                }
                            )
                    del matrix, x_pca
                    gc.collect()
                if fold == 1:
                    write_csv(
                        TABLE_ROOT / "embedding_tsne_coordinates.csv",
                        output_tsne,
                        [
                            "partition",
                            "model_role",
                            "model",
                            "dataset",
                            "fold",
                            "smiles",
                            "region",
                            "tsne_x",
                            "tsne_y",
                            "sample_policy",
                        ],
                    )
                print(
                    f"Analyzed {dataset} {partition} fold {fold}: "
                    f"{len(common)} molecules",
                    flush=True,
                )

    metric_fields = [
        "partition",
        "model_role",
        "model",
        "dataset",
        "fold",
        "common_molecules",
        "sample_policy",
        "pca_components",
        "silhouette",
        "davies_bouldin",
        "intra_distance",
        "inter_distance",
        "inter_intra_ratio",
    ]
    write_csv(
        TABLE_ROOT / "embedding_separation_metrics.csv",
        output_metrics,
        metric_fields,
    )
    write_csv(
        TABLE_ROOT / "embedding_metric_summary.csv",
        summarize_metrics(output_metrics),
        [
            "partition",
            "model_role",
            "model",
            "dataset",
            "folds_found",
            "fold_ids",
            "complete_5fold",
            "common_molecules_min",
            "silhouette_mean",
            "silhouette_sd",
            "davies_bouldin_mean",
            "davies_bouldin_sd",
            "intra_distance_mean",
            "intra_distance_sd",
            "inter_distance_mean",
            "inter_distance_sd",
            "inter_intra_ratio_mean",
            "inter_intra_ratio_sd",
        ],
    )
    write_csv(
        TABLE_ROOT / "embedding_tsne_coordinates.csv",
        output_tsne,
        [
            "partition",
            "model_role",
            "model",
            "dataset",
            "fold",
            "smiles",
            "region",
            "tsne_x",
            "tsne_y",
            "sample_policy",
        ],
    )
    (TABLE_ROOT / "embedding_analysis_audit.json").write_text(
        json.dumps(
            {
                "datasets": DATASETS,
                "folds_for_metrics": FOLDS,
                "fold_for_tsne": 1,
                "large_dataset_sample_limit": MAX_LARGE_SAMPLE,
                "pooling": "mean of the first RDKit atom-count node rows",
                "pca": "50 components or maximum available, then L2 normalization",
                "tsne": {
                    "perplexity": 30,
                    "initialization": "PCA",
                    "iterations": 2000,
                    "seed": 42,
                },
                "comparison_policy": (
                    "common SMILES within each dataset/partition/fold; "
                    "no coordinate-space merging across folds"
                ),
                "runs": audit_rows,
                "inference_run": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(output_metrics)} metric rows and "
        f"{len(output_tsne)} t-SNE coordinate rows"
    )


if __name__ == "__main__":
    main()
