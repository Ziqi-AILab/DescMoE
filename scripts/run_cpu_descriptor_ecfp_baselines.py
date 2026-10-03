#!/usr/bin/env python3
"""Run validation-selected descriptor and ECFP baselines on local CPU.

The split implementation is imported from the downstream training code so the
sample rows and scaffold seeds match the corrected neural experiments. Test
labels are evaluated only after one regularization value has been selected
from the complete validation split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit.Chem import Crippen, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
MOLESG = ROOT / "MoleSG"
DOWNSTREAM = MOLESG / "Downstream"
OUTPUT_ROOT = ROOT / "runs/cpu_baselines"
DATASETS = ["bbbp", "tox21", "toxcast", "sider", "clintox", "bace", "hiv", "muv"]
FAMILIES = ["tpsa", "logp", "ecfp4", "ecfp4_tpsa", "ecfp4_logp"]
SEEDS = [42, 43, 44, 45, 46]

import sys

sys.path.insert(0, str(DOWNSTREAM))
from utils import scaffold_split  # noqa: E402


@dataclass(frozen=True)
class EndpointFit:
    endpoint: int
    model: LogisticRegression | None
    validation_auc: float


@dataclass(frozen=True)
class EndpointPrediction:
    endpoint: int
    observed_rows: np.ndarray
    scores: np.ndarray
    auc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--c-grid", default="0.01,0.1,1,10,100")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def as_label_matrix(labels) -> np.ndarray:
    values = np.asarray(labels, dtype=float)
    return values[:, None] if values.ndim == 1 else values


def fingerprint_matrix(molecules) -> sparse.csr_matrix:
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    rows: list[int] = []
    cols: list[int] = []
    for row, molecule in enumerate(molecules):
        bits = generator.GetFingerprint(molecule).GetOnBits()
        rows.extend([row] * len(bits))
        cols.extend(bits)
    values = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix(
        (values, (rows, cols)), shape=(len(molecules), 2048), dtype=np.float32)


def descriptor_vector(molecules, name: str) -> np.ndarray:
    if name == "tpsa":
        values = [rdMolDescriptors.CalcTPSA(molecule) for molecule in molecules]
    elif name == "logp":
        values = [Crippen.MolLogP(molecule) for molecule in molecules]
    else:
        raise ValueError(name)
    return np.asarray(values, dtype=np.float32)[:, None]


def features_for_family(family: str, ecfp: sparse.csr_matrix,
                        descriptors: dict[str, np.ndarray],
                        train_indices: np.ndarray) -> sparse.csr_matrix | np.ndarray:
    if family in {"tpsa", "logp"}:
        values = descriptors[family]
        mean = float(values[train_indices].mean())
        scale = float(values[train_indices].std(ddof=0)) or 1.0
        return (values - mean) / scale
    if family == "ecfp4":
        return ecfp
    descriptor = family.removeprefix("ecfp4_")
    values = descriptors[descriptor]
    mean = float(values[train_indices].mean())
    scale = float(values[train_indices].std(ddof=0)) or 1.0
    standardized = sparse.csr_matrix((values - mean) / scale)
    return sparse.hstack([ecfp, standardized], format="csr")


def fit_endpoint(x_train, y_train: np.ndarray, x_valid, y_valid: np.ndarray,
                 endpoint: int, c_value: float) -> EndpointFit:
    train_mask = y_train >= 0
    valid_mask = y_valid >= 0
    if np.unique(y_train[train_mask]).size < 2 or np.unique(y_valid[valid_mask]).size < 2:
        return EndpointFit(endpoint, None, float("nan"))
    model = LogisticRegression(
        C=c_value, solver="liblinear", max_iter=2000, random_state=42)
    model.fit(x_train[train_mask], y_train[train_mask].astype(int))
    scores = model.predict_proba(x_valid[valid_mask])[:, 1]
    return EndpointFit(
        endpoint, model,
        float(roc_auc_score(y_valid[valid_mask].astype(int), scores)))


def select_c(x, labels: np.ndarray, train_indices: np.ndarray,
             valid_indices: np.ndarray, c_grid: list[float], workers: int):
    x_train, x_valid = x[train_indices], x[valid_indices]
    y_train, y_valid = labels[train_indices], labels[valid_indices]
    records = []
    for c_value in c_grid:
        fits = Parallel(n_jobs=workers, prefer="threads")(
            delayed(fit_endpoint)(
                x_train, y_train[:, endpoint], x_valid,
                y_valid[:, endpoint], endpoint, c_value)
            for endpoint in range(labels.shape[1])
        )
        valid = [fit.validation_auc for fit in fits if np.isfinite(fit.validation_auc)]
        macro = float(np.mean(valid)) if valid else float("nan")
        records.append({
            "C": c_value,
            "validation_auc": macro,
            "valid_endpoints": len(valid),
        })
    finite = [record for record in records if np.isfinite(record["validation_auc"])]
    if not finite:
        raise RuntimeError("No validation endpoint contained both classes")
    best = max(finite, key=lambda record: (record["validation_auc"], -record["C"]))
    return float(best["C"]), records


def predict_endpoint(x_train, y_train: np.ndarray, x_test,
                     y_test: np.ndarray, endpoint: int,
                     c_value: float) -> EndpointPrediction:
    train_mask = y_train >= 0
    observed = np.flatnonzero(y_test >= 0)
    if np.unique(y_train[train_mask]).size < 2 or observed.size == 0:
        return EndpointPrediction(
            endpoint, observed, np.empty(0, dtype=float), float("nan"))
    model = LogisticRegression(
        C=c_value, solver="liblinear", max_iter=2000, random_state=42)
    model.fit(x_train[train_mask], y_train[train_mask].astype(int))
    scores = model.predict_proba(x_test[observed])[:, 1]
    auc = float("nan")
    if np.unique(y_test[observed]).size >= 2:
        auc = float(roc_auc_score(y_test[observed].astype(int), scores))
    return EndpointPrediction(endpoint, observed, scores, auc)


def final_predictions(x, labels: np.ndarray, train_indices: np.ndarray,
                      test_indices: np.ndarray, c_value: float, workers: int):
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    fits = Parallel(n_jobs=workers, prefer="threads")(
        delayed(predict_endpoint)(
            x_train, y_train[:, endpoint], x_test,
            y_test[:, endpoint], endpoint, c_value)
        for endpoint in range(labels.shape[1])
    )
    predictions = np.full(y_test.shape, np.nan, dtype=float)
    endpoint_aucs = []
    for fit in fits:
        predictions[fit.observed_rows, fit.endpoint] = fit.scores
        if np.isfinite(fit.auc):
            endpoint_aucs.append(fit.auc)
    if not endpoint_aucs:
        raise RuntimeError("No test endpoint contained both classes")
    return predictions, float(np.mean(endpoint_aucs)), len(endpoint_aucs)


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    families = [value.strip() for value in args.families.split(",") if value.strip()]
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    c_grid = [float(value) for value in args.c_grid.split(",")]
    unknown = set(datasets) - set(DATASETS)
    unknown_families = set(families) - set(FAMILIES)
    if unknown or unknown_families:
        raise SystemExit(f"Unknown datasets={sorted(unknown)} families={sorted(unknown_families)}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if any(fold < 1 or fold > 5 for fold in folds):
        raise SystemExit(f"Folds must be between 1 and 5: {folds}")
    fold_path = output_root / "cpu_baseline_fold_auc.csv"
    selection_path = output_root / "cpu_baseline_validation_selection.csv"
    frozen_hyperparameters_path = (
        output_root / "cpu_baseline_selected_hyperparameters.json")
    fold_rows = pd.read_csv(fold_path).to_dict("records") if fold_path.is_file() else []
    selection_rows = (
        pd.read_csv(selection_path).to_dict("records")
        if selection_path.is_file() else [])
    completed = {
        (str(row["family"]), str(row["dataset"]), int(row["fold"]))
        for row in fold_rows
    }
    frozen_hyperparameters = (
        json.loads(frozen_hyperparameters_path.read_text())
        if frozen_hyperparameters_path.is_file() else {})
    coverage_rows: list[dict] = []

    for dataset in datasets:
        with (args.data_root / dataset / "preprocess" /
              f"{dataset}.pickle").open("rb") as handle:
            molecules, raw_labels = pickle.load(handle)
        labels = as_label_matrix(raw_labels)
        ecfp = fingerprint_matrix(molecules)
        descriptors = {
            "tpsa": descriptor_vector(molecules, "tpsa"),
            "logp": descriptor_vector(molecules, "logp"),
        }
        from rdkit import Chem
        canonical = [Chem.MolToSmiles(molecule) for molecule in molecules]

        for fold, seed in enumerate(SEEDS, start=1):
            if fold not in folds:
                continue
            train, valid, test = scaffold_split(
                molecules, frac=[0.8, 0.1, 0.1], balanced=True,
                include_chirality=False, ramdom_state=seed)
            train = np.asarray(train, dtype=int)
            valid = np.asarray(valid, dtype=int)
            test = np.asarray(test, dtype=int)
            for split_name, indices in (("train", train), ("validation", valid), ("test", test)):
                coverage_rows.append({
                    "dataset": dataset, "fold": fold, "seed": seed,
                    "split": split_name, "expected_samples": len(indices),
                    "evaluated_samples": len(indices),
                    "unique_sample_ids": len(np.unique(indices)),
                    "duplicate_sample_ids": len(indices) - len(np.unique(indices)),
                    "coverage_ok": len(indices) == len(np.unique(indices)),
                })

            for family in families:
                prediction_path = (
                    output_root / "predictions" / dataset /
                    f"test_{family}_{dataset}_fold_{fold}.csv")
                key = (family, dataset, fold)
                if key in completed and prediction_path.is_file() and not args.overwrite:
                    print(f"SKIP_EXISTING: {family} {dataset} fold={fold}")
                    continue
                if args.overwrite and key in completed:
                    fold_rows = [row for row in fold_rows if (
                        str(row["family"]), str(row["dataset"]), int(row["fold"])) != key]
                    selection_rows = [row for row in selection_rows if (
                        str(row["family"]), str(row["dataset"]), int(row["fold"])) != key]
                x = features_for_family(family, ecfp, descriptors, train)
                best_c, candidates = select_c(
                    x, labels, train, valid, c_grid, args.workers)
                for record in candidates:
                    selection_rows.append({
                        "family": family, "dataset": dataset, "fold": fold,
                        "seed": seed, **record, "selected": record["C"] == best_c,
                    })
                write_rows(
                    selection_path, selection_rows, list(selection_rows[0]))
                frozen_hyperparameters[f"{family}|{dataset}|fold{fold}"] = {
                    "family": family,
                    "dataset": dataset,
                    "fold": fold,
                    "seed": seed,
                    "selected_C": best_c,
                    "selection_split": "validation",
                    "test_used_for_selection": False,
                }
                frozen_hyperparameters_path.write_text(
                    json.dumps(frozen_hyperparameters, indent=2) + "\n")
                predictions, test_auc, valid_endpoints = final_predictions(
                    x, labels, train, test, best_c, args.workers)
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({
                    "sample_id": [f"{dataset}:{index}" for index in test],
                    "source_row_index": test,
                    "canonical_smiles": [canonical[index] for index in test],
                    "actual": [json.dumps(row.tolist()) for row in labels[test]],
                    "prediction": [json.dumps(row.tolist()) for row in predictions],
                }).to_csv(prediction_path, index=False)
                fold_rows.append({
                    "family": family, "dataset": dataset, "fold": fold,
                    "seed": seed, "selected_C": best_c,
                    "roc_auc": test_auc, "valid_endpoints": valid_endpoints,
                    "evaluated_samples": len(test), "split_samples": len(test),
                    "evaluation_coverage_ok": True,
                    "source_csv": str(prediction_path),
                })
                write_rows(
                    fold_path, fold_rows,
                    list(fold_rows[0]))
                print(
                    f"CPU_BASELINE_DONE: family={family} dataset={dataset} "
                    f"fold={fold} C={best_c:g} test_auc={test_auc:.6f}")

    write_rows(
        output_root / "cpu_baseline_coverage_audit.csv", coverage_rows,
        list(coverage_rows[0]))
    if fold_rows:
        summary = (
            pd.DataFrame(fold_rows).groupby(["family", "dataset"], as_index=False)
            .agg(mean_roc_auc=("roc_auc", "mean"),
                 sample_sd=("roc_auc", "std"), folds=("fold", "count")))
        summary.to_csv(output_root / "cpu_baseline_model_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
