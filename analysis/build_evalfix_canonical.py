#!/usr/bin/env python3
"""Build the corrected N8 canonical table after the frozen final test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "runs/results"
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "FINAL_N8_EVALFIX_FINAL_TEST.tsv"))
    parser.add_argument("--result-root", default=str(RESULT_ROOT))
    parser.add_argument("--model-definition-lock", default=str(ROOT / "model_definition_frozen.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/final_n8_canonical"))
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def parse_vector(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def roc_auc(path: Path) -> tuple[float, int, int]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty prediction file: {path}")
    if frame.sample_id.duplicated().any():
        raise ValueError(f"Duplicate sample_id in {path}")
    labels = np.vstack(frame.actual.map(parse_vector))
    predictions = np.vstack(frame.prediction.map(parse_vector))
    if not np.isfinite(predictions).all():
        raise ValueError(f"Non-finite prediction in {path}")
    values = []
    for endpoint in range(labels.shape[1]):
        mask = labels[:, endpoint] >= 0
        endpoint_labels = labels[mask, endpoint]
        if len(np.unique(endpoint_labels)) < 2:
            continue
        values.append(roc_auc_score(endpoint_labels, predictions[mask, endpoint]))
    if not values:
        raise ValueError(f"No valid classification endpoint in {path}")
    return float(np.mean(values)), len(values), len(frame)


def verify_sample_manifest(prediction_path: Path, sample_manifest_path: Path) -> None:
    prediction = pd.read_csv(prediction_path, usecols=["sample_id"])
    sample_manifest = pd.read_csv(sample_manifest_path, usecols=["sample_id"])
    if sample_manifest.sample_id.duplicated().any():
        raise ValueError(f"Duplicate sample_id in {sample_manifest_path}")
    if prediction.sample_id.tolist() != sample_manifest.sample_id.tolist():
        raise ValueError(
            f"Prediction and split manifest order differ: {prediction_path}")


def load_lock(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Model definition lock is absent: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "frozen":
        raise SystemExit(f"Model definition is not frozen: {path}")
    if payload.get("test_results_used_for_selection") is not False:
        raise SystemExit("Model definition lock does not prove test isolation")
    return payload


def hierarchical_ci(delta_frame: pd.DataFrame) -> tuple[float, float]:
    matrix = delta_frame.pivot(index="dataset", columns="fold", values="delta")
    if matrix.shape != (8, 5) or matrix.isna().any().any():
        return float("nan"), float("nan")
    values = matrix.to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        dataset_indices = rng.integers(0, values.shape[0], values.shape[0])
        dataset_means = []
        for dataset_index in dataset_indices:
            fold_indices = rng.integers(0, values.shape[1], values.shape[1])
            dataset_means.append(values[dataset_index, fold_indices].mean())
        estimates[iteration] = np.mean(dataset_means)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def build_contrasts(canonical: pd.DataFrame) -> pd.DataFrame:
    complete = canonical[canonical.completion == "complete"].copy()
    lookup = {
        (row.partition, row.ablation_role): row.model
        for row in complete[["partition", "ablation_role", "model"]]
        .drop_duplicates().itertuples(index=False)
    }
    base_models = complete[complete.ablation_role == "base"].model.unique()
    if len(base_models) != 1:
        return pd.DataFrame()
    base_model = base_models[0]
    comparisons = [
        ("I1_vs_base", "I1", "base"),
        ("I2_effect", "I1+I2", "I1"),
        ("I3_effect", "I1+I3", "I1"),
        ("full_vs_I1", "full", "I1"),
        ("full_vs_base", "full", "base"),
        ("all_KAN_vs_selected", "all_kan", "I1+I2"),
    ]
    output = []
    for partition in ("tpsa", "logp"):
        for name, left_role, right_role in comparisons:
            left_model = lookup.get((partition, left_role))
            right_model = base_model if right_role == "base" else lookup.get(
                (partition, right_role))
            if left_model is None or right_model is None:
                continue
            left = complete[complete.model == left_model][
                ["dataset", "fold", "roc_auc"]].rename(columns={"roc_auc": "left_auc"})
            right = complete[complete.model == right_model][
                ["dataset", "fold", "roc_auc"]].rename(columns={"roc_auc": "right_auc"})
            paired = left.merge(right, on=["dataset", "fold"], validate="one_to_one")
            paired["delta"] = paired.left_auc - paired.right_auc
            dataset_delta = paired.groupby("dataset", as_index=False).delta.mean()
            for row in dataset_delta.itertuples(index=False):
                output.append({
                    "partition": partition, "comparison": name,
                    "left_model": left_model, "right_model": right_model,
                    "level": "dataset", "dataset": row.dataset,
                    "delta": float(row.delta), "wins": np.nan,
                    "datasets": 1, "ci_low": np.nan, "ci_high": np.nan,
                })
            ci_low, ci_high = hierarchical_ci(paired)
            output.append({
                "partition": partition, "comparison": name,
                "left_model": left_model, "right_model": right_model,
                "level": "macro", "dataset": "macro",
                "delta": float(dataset_delta.delta.mean()),
                "wins": int((dataset_delta.delta > 0).sum()),
                "datasets": len(dataset_delta), "ci_low": ci_low,
                "ci_high": ci_high,
            })
    return pd.DataFrame(output)


def main() -> int:
    args = parse_args()
    lock_path = Path(args.model_definition_lock)
    lock = load_lock(lock_path)
    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.manifest).open() as handle:
        manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(manifest_rows) not in {56, 88}:
        raise SystemExit(
            f"Expected a 56-row or 88-row final matrix, found {len(manifest_rows)}")
    expected_folds = len(manifest_rows) * 5

    records = []
    errors = []
    for row in manifest_rows:
        for fold in range(1, 6):
            prediction_path = result_root / row["dataset"] / (
                f"test_result_{row['result_exp']}_{row['dataset']}_fold_{fold}.csv")
            metadata_path = prediction_path.with_suffix(".json")
            record = {
                "model": row["checkpoint_model"],
                "checkpoint": "",
                "protocol": "evalfix_full_coverage_v1",
                "partition": row["partition"],
                "ablation_role": row["role"],
                "dataset": row["dataset"],
                "fold": fold,
                "seed": 41 + fold,
                "roc_auc": np.nan,
                "completion": "missing",
                "source_csv": str(prediction_path),
                "source_log": "",
                "sample_id_manifest": "",
                "validation_best_epoch": np.nan,
                "validation_best_auc": np.nan,
                "evaluated_samples": 0,
                "split_samples": 0,
                "test_valid_endpoints": 0,
                "checkpoint_load_status": "missing",
                "evaluation_coverage_ok": False,
                "selection_source": str(lock_path),
            }
            if not prediction_path.is_file() or not metadata_path.is_file():
                errors.append(f"missing output: {prediction_path}")
                records.append(record)
                continue
            try:
                metadata = json.loads(metadata_path.read_text())
                sample_manifest = Path(metadata["sample_id_manifest"])
                verify_sample_manifest(prediction_path, sample_manifest)
                auc, endpoints, samples = roc_auc(prediction_path)
                coverage_ok = (
                    metadata.get("evaluation_coverage_ok") is True
                    and metadata.get("evaluated_samples") == metadata.get("split_samples")
                    and samples == metadata.get("split_samples")
                )
                if not coverage_ok:
                    raise ValueError(f"Coverage mismatch in {metadata_path}")
                record.update({
                    "checkpoint": metadata.get("checkpoint", ""),
                    "roc_auc": auc,
                    "completion": "complete",
                    "sample_id_manifest": str(sample_manifest),
                    "validation_best_epoch": metadata.get("validation_best_epoch"),
                    "validation_best_auc": metadata.get("validation_best_auc"),
                    "evaluated_samples": samples,
                    "split_samples": metadata.get("split_samples"),
                    "test_valid_endpoints": endpoints,
                    "checkpoint_load_status": metadata.get("checkpoint_load_status"),
                    "evaluation_coverage_ok": True,
                })
            except Exception as exc:
                errors.append(f"{prediction_path}: {exc}")
                record["completion"] = "invalid"
            records.append(record)

    canonical = pd.DataFrame(records)
    canonical.to_csv(output_dir / "final_n8_metrics_long.csv", index=False)
    canonical.to_csv(output_dir / "corrected_fold_auc.csv", index=False)
    complete = canonical[canonical.completion == "complete"]
    summary = (
        complete.groupby(["model", "partition", "ablation_role", "dataset"])
        .agg(mean_roc_auc=("roc_auc", "mean"), sd_roc_auc=("roc_auc", "std"),
             folds=("roc_auc", "size"))
        .reset_index()
    )
    summary.to_csv(output_dir / "final_n8_dataset_summary.csv", index=False)
    model_summary = (
        summary.groupby(["model", "partition", "ablation_role"])
        .agg(macro_roc_auc=("mean_roc_auc", "mean"), datasets=("dataset", "size"),
             complete_folds=("folds", "sum"))
        .reset_index()
    )
    model_summary.to_csv(output_dir / "corrected_model_summary.csv", index=False)
    coverage_columns = [
        "model", "dataset", "fold", "seed", "completion", "evaluated_samples",
        "split_samples", "evaluation_coverage_ok", "sample_id_manifest", "source_csv",
    ]
    canonical[coverage_columns].to_csv(output_dir / "coverage_audit.csv", index=False)
    checkpoint_columns = [
        "model", "dataset", "fold", "seed", "checkpoint",
        "validation_best_epoch", "validation_best_auc", "checkpoint_load_status",
        "selection_source",
    ]
    canonical[checkpoint_columns].to_csv(
        output_dir / "checkpoint_selection.csv", index=False)
    contrasts = build_contrasts(canonical)
    contrasts.to_csv(output_dir / "corrected_paired_contrasts.csv", index=False)
    status = {
        "expected_folds": expected_folds,
        "complete_folds": int(len(complete)),
        "invalid_or_missing_folds": int(expected_folds - len(complete)),
        "errors": errors,
        "model_definition_lock": lock,
        "ready_for_manuscript_refresh": len(complete) == expected_folds and not errors,
    }
    (output_dir / "completion_audit.json").write_text(json.dumps(status, indent=2) + "\n")
    (output_dir / "completion_audit.md").write_text(
        "# Corrected N8 Completion Audit\n\n"
        f"Complete folds: `{len(complete)}/{expected_folds}`\n\n"
        f"Ready for manuscript refresh: `{status['ready_for_manuscript_refresh']}`\n")
    print(json.dumps({key: value for key, value in status.items() if key != "errors"}, indent=2))
    if status["ready_for_manuscript_refresh"] or args.allow_incomplete:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
