#!/usr/bin/env python3
"""Build the frozen 16-model P0 panel from full-coverage test outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_evalfix_canonical import hierarchical_ci, roc_auc, verify_sample_manifest


ROOT = Path(__file__).resolve().parents[1]
CORE_RESULT_ROOT = ROOT / "runs/results"
P6_RESULT_ROOT = CORE_RESULT_ROOT
P8_RESULT_ROOT = CORE_RESULT_ROOT
DEFAULT_LOCK = ROOT / "runs/model_definition_frozen.json"
DEFAULT_OUTPUT = ROOT / "runs/summary"
CPU_FOLDS = ROOT / "runs/cpu_baselines/cpu_baseline_fold_auc.csv"
DATASETS = ["bbbp", "tox21", "toxcast", "sider", "clintox", "bace", "hiv", "muv"]

CORE_CONTROL = {
    "base": "dense_base",
    "I1": "fixed_assignment",
    "I1+I3": "fixed_assignment_conloss",
}
P6_CONTROL = {
    "descriptor_concat": "descriptor_concatenation",
    "descriptor_auxiliary": "auxiliary_descriptor_prediction",
    "quantile_routing": "quantile_assignment",
}
P8_CONTROL = {
    "stable_random": "stable_random_assignment",
    "stable_random_conloss": "stable_random_assignment_conloss",
    "learned_molecule_top1": "learned_molecule_top1",
}

COMPARISONS = [
    ("fixed_vs_dense", "fixed_assignment", "dense_base"),
    ("fixed_vs_concat", "fixed_assignment", "descriptor_concatenation"),
    ("fixed_vs_auxiliary", "fixed_assignment", "auxiliary_descriptor_prediction"),
    ("fixed_vs_learned_molecule", "fixed_assignment", "learned_molecule_top1"),
    ("fixed_vs_quantile", "fixed_assignment", "quantile_assignment"),
    ("fixed_vs_stable_random", "fixed_assignment", "stable_random_assignment"),
    ("fixed_conloss_vs_fixed", "fixed_assignment_conloss", "fixed_assignment"),
    ("fixed_conloss_vs_random_conloss", "fixed_assignment_conloss",
     "stable_random_assignment_conloss"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--result-root", type=Path, default=CORE_RESULT_ROOT)
    parser.add_argument("--cpu-folds", type=Path, default=CPU_FOLDS)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def load_lock(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Frozen P0 panel is absent: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "frozen":
        raise SystemExit(f"P0 panel is not frozen: {path}")
    if payload.get("test_results_used_for_selection") is not False:
        raise SystemExit("P0 panel does not preserve test isolation")
    if payload.get("final_test_release") is not True:
        raise SystemExit("P0 panel has not released final-test evaluation")
    if payload.get("final_matrix_models") != 16:
        raise SystemExit("P0 panel must contain exactly 16 model configurations")
    if payload.get("expected_final_test_folds") != 640:
        raise SystemExit("P0 panel must declare exactly 640 final-test folds")
    return payload


def model_sources(lock: dict) -> list[dict]:
    rows: list[dict] = []
    for spec in lock["model_specs"]:
        control = CORE_CONTROL.get(spec["role"])
        if control is None:
            raise ValueError(f"Unsupported core role: {spec['role']}")
        rows.append({
            "model": spec["checkpoint_model"],
            "result_exp": spec["result_exp"],
            "partition": "shared" if spec["role"] == "base" else spec["partition"],
            "control": control,
            "source_group": "corrected_core",
            "result_root": CORE_RESULT_ROOT,
        })
    for spec in lock["reused_p6_model_specs"]:
        rows.append({
            "model": spec["checkpoint_model"],
            "result_exp": spec["result_exp"],
            "partition": spec["partition"],
            "control": P6_CONTROL[spec["control"]],
            "source_group": "reused_corrected_p6",
            "result_root": P6_RESULT_ROOT,
        })
    for spec in lock["p8_model_specs"]:
        rows.append({
            "model": spec["checkpoint_model"],
            "result_exp": spec["result_exp"],
            "partition": spec["partition"],
            "control": P8_CONTROL[spec["control"]],
            "source_group": "corrected_p8",
            "result_root": P8_RESULT_ROOT,
        })
    result_names = [row["result_exp"] for row in rows]
    if len(rows) != 16 or len(result_names) != len(set(result_names)):
        raise ValueError("Frozen P0 model list is not a unique 16-model panel")
    if set(result_names) != set(lock["allowed_source_experiments"]):
        raise ValueError("Frozen P0 model list differs from its allowed experiments")
    return rows


def empty_record(spec: dict, dataset: str, fold: int, path: Path) -> dict:
    return {
        "model": spec["model"],
        "result_exp": spec["result_exp"],
        "partition": spec["partition"],
        "control": spec["control"],
        "source_group": spec["source_group"],
        "dataset": dataset,
        "fold": fold,
        "seed": 41 + fold,
        "roc_auc": np.nan,
        "completion": "missing",
        "evaluated_samples": 0,
        "split_samples": 0,
        "unique_samples": 0,
        "duplicate_samples": 0,
        "evaluation_coverage_ok": False,
        "source_csv": str(path),
        "source_metadata": str(path.with_suffix(".json")),
        "sample_id_manifest": "",
        "checkpoint": "",
        "validation_best_epoch": np.nan,
        "validation_best_auc": np.nan,
        "valid_endpoints": 0,
    }


def read_fold(spec: dict, dataset: str, fold: int) -> tuple[dict, str | None]:
    path = spec["result_root"] / dataset / (
        f"test_result_{spec['result_exp']}_{dataset}_fold_{fold}.csv")
    metadata_path = path.with_suffix(".json")
    record = empty_record(spec, dataset, fold, path)
    if not path.is_file() or not metadata_path.is_file():
        return record, f"missing output: {path}"
    try:
        metadata = json.loads(metadata_path.read_text())
        expected = {
            "protocol": "evalfix_full_coverage_v1",
            "mode": "final_test",
            "split": "test",
            "dataset": dataset,
            "fold": fold,
            "experiment_name": spec["result_exp"],
            "source_experiment_name": spec["result_exp"],
            "checkpoint_load_status": "FINETUNED_CHECKPOINT_LOAD_OK",
            "legacy_best_checkpoint": False,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"metadata {key}={metadata.get(key)!r}, expected {value!r}")
        if spec["source_group"] == "corrected_p8":
            if metadata.get("num_experts") != 8:
                raise ValueError("P8 control does not contain eight branches")
            if metadata.get("moe_layer_mode") != "last":
                raise ValueError("P8 control is not restricted to the last layer")
            if spec["control"] == "learned_molecule_top1":
                expected_p8 = {
                    "assignment_scheme": "learned",
                    "use_prior_moe": False,
                    "use_standard_moe": True,
                    "router_top_k": 1,
                    "router_granularity": "molecule",
                }
            else:
                expected_p8 = {
                    "assignment_scheme": "stable_random",
                    "descriptor_property": spec["partition"],
                    "use_prior_moe": True,
                    "use_standard_moe": False,
                    "expert_type": "ffn",
                }
            for key, value in expected_p8.items():
                if metadata.get(key) != value:
                    raise ValueError(
                        f"P8 metadata {key}={metadata.get(key)!r}, "
                        f"expected {value!r}")
        sample_manifest = Path(metadata["sample_id_manifest"])
        verify_sample_manifest(path, sample_manifest)
        frame = pd.read_csv(path, usecols=["sample_id"])
        duplicates = int(frame.sample_id.duplicated().sum())
        auc, endpoints, samples = roc_auc(path)
        coverage = (
            metadata.get("evaluation_coverage_ok") is True
            and metadata.get("evaluated_samples") == metadata.get("split_samples")
            and samples == metadata.get("split_samples")
            and duplicates == 0
        )
        if not coverage:
            raise ValueError("sample-ID coverage mismatch")
        if spec["control"] == "learned_molecule_top1":
            routed = pd.read_csv(path, usecols=["router_expert_id"])
            route_ids = routed.router_expert_id.to_numpy(dtype=int)
            occupancy = np.bincount(route_ids, minlength=8).tolist()
            if (
                metadata.get("router_granularity") != "molecule"
                or metadata.get("router_routed_molecules") != samples
                or metadata.get("router_occupancy") != occupancy
                or len(route_ids) != samples
                or (route_ids < 0).any()
                or (route_ids > 7).any()
            ):
                raise ValueError("molecule-router occupancy metadata mismatch")
        recorded_auc = metadata.get("metric", {}).get("auc")
        if recorded_auc is None or not np.isclose(
                auc, float(recorded_auc), rtol=0.0, atol=1e-12):
            raise ValueError("recomputed ROC-AUC differs from saved metadata")
        record.update({
            "roc_auc": auc,
            "completion": "complete",
            "evaluated_samples": samples,
            "split_samples": int(metadata["split_samples"]),
            "unique_samples": int(frame.sample_id.nunique()),
            "duplicate_samples": duplicates,
            "evaluation_coverage_ok": True,
            "sample_id_manifest": str(sample_manifest),
            "checkpoint": metadata.get("checkpoint", ""),
            "validation_best_epoch": metadata.get("validation_best_epoch", np.nan),
            "validation_best_auc": metadata.get("validation_best_auc", np.nan),
            "valid_endpoints": endpoints,
        })
        return record, None
    except Exception as exc:
        record["completion"] = "invalid"
        return record, f"{path}: {exc}"


def paired_contrasts(canonical: pd.DataFrame) -> pd.DataFrame:
    complete = canonical[canonical.completion == "complete"]
    output: list[dict] = []
    for partition in ("tpsa", "logp"):
        scoped = pd.concat([
            complete[complete.partition == partition],
            complete[complete.partition == "shared"],
        ], ignore_index=True)
        for name, left_control, right_control in COMPARISONS:
            left = scoped[scoped.control == left_control][
                ["dataset", "fold", "roc_auc"]].rename(
                    columns={"roc_auc": "left_auc"})
            right = scoped[scoped.control == right_control][
                ["dataset", "fold", "roc_auc"]].rename(
                    columns={"roc_auc": "right_auc"})
            if len(left) != 40 or len(right) != 40:
                continue
            paired = left.merge(
                right, on=["dataset", "fold"], validate="one_to_one")
            paired["delta"] = paired.left_auc - paired.right_auc
            dataset_delta = paired.groupby("dataset", as_index=False).delta.mean()
            for row in dataset_delta.itertuples(index=False):
                output.append({
                    "partition": partition,
                    "comparison": name,
                    "left_control": left_control,
                    "right_control": right_control,
                    "level": "dataset",
                    "dataset": row.dataset,
                    "delta": float(row.delta),
                    "wins": np.nan,
                    "datasets": 1,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                })
            ci_low, ci_high = hierarchical_ci(paired)
            output.append({
                "partition": partition,
                "comparison": name,
                "left_control": left_control,
                "right_control": right_control,
                "level": "macro",
                "dataset": "macro",
                "delta": float(dataset_delta.delta.mean()),
                "wins": int((dataset_delta.delta > 0).sum()),
                "datasets": len(dataset_delta),
                "ci_low": ci_low,
                "ci_high": ci_high,
            })
    return pd.DataFrame(output)


def write_cpu_tables(output_dir: Path) -> dict:
    if not CPU_FOLDS.is_file():
        return {"available": False, "folds": 0}
    folds = pd.read_csv(CPU_FOLDS)
    expected = 5 * len(DATASETS) * 5
    complete = (
        len(folds) == expected
        and folds.evaluation_coverage_ok.astype(bool).all()
        and folds.groupby(["family", "dataset"]).size().eq(5).all()
    )
    folds.to_csv(output_dir / "cpu_baseline_fold_auc.csv", index=False)
    dataset = (
        folds.groupby(["family", "dataset"])
        .agg(mean_roc_auc=("roc_auc", "mean"),
             sd_roc_auc=("roc_auc", "std"), folds=("roc_auc", "size"))
        .reset_index()
    )
    model = (
        dataset.groupby("family")
        .agg(macro_roc_auc=("mean_roc_auc", "mean"),
             datasets=("dataset", "size"), folds=("folds", "sum"))
        .reset_index()
    )
    dataset.to_csv(output_dir / "cpu_baseline_dataset_summary.csv", index=False)
    model.to_csv(output_dir / "cpu_baseline_model_summary.csv", index=False)
    return {"available": True, "folds": int(len(folds)), "complete": bool(complete)}


def main() -> int:
    global CORE_RESULT_ROOT, P6_RESULT_ROOT, P8_RESULT_ROOT, CPU_FOLDS
    args = parse_args()
    CORE_RESULT_ROOT = P6_RESULT_ROOT = P8_RESULT_ROOT = args.result_root.resolve()
    CPU_FOLDS = args.cpu_folds.resolve()
    lock_path = Path(args.lock).resolve()
    lock = load_lock(lock_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    errors: list[str] = []
    specs = model_sources(lock)
    for spec in specs:
        for dataset in DATASETS:
            for fold in range(1, 6):
                record, error = read_fold(spec, dataset, fold)
                records.append(record)
                if error:
                    errors.append(error)
    canonical = pd.DataFrame(records)
    canonical.to_csv(output_dir / "corrected_fold_auc.csv", index=False)

    coverage_columns = [
        "model", "result_exp", "source_group", "dataset", "fold", "seed",
        "completion", "evaluated_samples", "split_samples", "unique_samples",
        "duplicate_samples", "evaluation_coverage_ok", "sample_id_manifest",
        "source_csv",
    ]
    canonical[coverage_columns].to_csv(
        output_dir / "coverage_audit.csv", index=False)
    checkpoint_columns = [
        "model", "result_exp", "dataset", "fold", "seed", "checkpoint",
        "validation_best_epoch", "validation_best_auc", "completion",
    ]
    canonical[checkpoint_columns].to_csv(
        output_dir / "checkpoint_selection.csv", index=False)

    complete_rows = canonical[canonical.completion == "complete"]
    dataset_summary = (
        complete_rows.groupby([
            "model", "result_exp", "partition", "control", "source_group",
            "dataset",
        ])
        .agg(mean_roc_auc=("roc_auc", "mean"),
             sd_roc_auc=("roc_auc", "std"), folds=("roc_auc", "size"))
        .reset_index()
    )
    dataset_summary.to_csv(
        output_dir / "corrected_dataset_summary.csv", index=False)
    model_summary = (
        dataset_summary.groupby([
            "model", "result_exp", "partition", "control", "source_group",
        ])
        .agg(macro_roc_auc=("mean_roc_auc", "mean"),
             datasets=("dataset", "size"), folds=("folds", "sum"))
        .reset_index()
    )
    model_summary.to_csv(
        output_dir / "corrected_model_summary.csv", index=False)

    contrasts = paired_contrasts(canonical)
    contrasts.to_csv(
        output_dir / "corrected_paired_contrasts.csv", index=False)
    cpu_status = write_cpu_tables(output_dir)

    primary = contrasts[
        (contrasts.level == "macro")
        & (contrasts.comparison == "fixed_vs_stable_random")
    ]
    primary_decisions = []
    for row in primary.itertuples(index=False):
        supported = (
            row.delta > 0 and int(row.wins) >= 5 and row.ci_low > 0)
        primary_decisions.append({
            "partition": row.partition,
            "macro_delta": float(row.delta),
            "wins": int(row.wins),
            "ci_low": float(row.ci_low),
            "ci_high": float(row.ci_high),
            "descriptor_assignment_supported": bool(supported),
        })

    expected_folds = 640
    complete_folds = int((canonical.completion == "complete").sum())
    status = {
        "model_definition_lock": str(lock_path),
        "models": len(specs),
        "expected_folds": expected_folds,
        "complete_folds": complete_folds,
        "invalid_or_missing_folds": expected_folds - complete_folds,
        "errors": errors,
        "cpu_baselines": cpu_status,
        "primary_fixed_vs_stable_random": primary_decisions,
        "ready_for_manuscript": complete_folds == expected_folds and not errors,
    }
    (output_dir / "completion_audit.json").write_text(
        json.dumps(status, indent=2) + "\n")
    (output_dir / "completion_audit.md").write_text(
        "# P0 Corrected Panel Completion\n\n"
        f"Models: `{len(specs)}/16`\n\n"
        f"Complete final-test folds: `{complete_folds}/{expected_folds}`\n\n"
        f"CPU baseline folds: `{cpu_status.get('valid_folds', 0)}/200`\n\n"
        f"Ready for manuscript: `{status['ready_for_manuscript']}`\n"
    )
    print(json.dumps({
        key: value for key, value in status.items() if key != "errors"
    }, indent=2))
    if status["ready_for_manuscript"] or args.allow_incomplete:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
