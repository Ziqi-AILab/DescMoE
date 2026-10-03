#!/usr/bin/env python3
"""Check that the compact reviewer repository is internally complete."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "EXPERIMENT_PARAMETERS.md",
    "environment.yml",
    "configs/final_model_matrix.tsv",
    "configs/downstream_tasks.tsv",
    "MoleSG/pretrain/train_total.py",
    "MoleSG/pretrain/prior_moe.py",
    "MoleSG/pretrain/transformer_graph.py",
    "MoleSG/Downstream/train_graph.py",
    "MoleSG/Downstream/train_graph_evalfix.py",
    "MoleSG/Downstream/dataset_graph.py",
    "MoleSG/Data_process/stable_random_assignment.py",
    "scripts/run_model.py",
    "scripts/freeze_panel.py",
    "scripts/summarize_reported_results.py",
    "results/corrected/corrected_fold_auc.csv",
    "results/corrected/cpu_baseline_fold_auc.csv",
    "docs/METHOD_CODE_MAP.md",
]
EXCLUDED_SUFFIXES = {".pth", ".pt", ".ckpt", ".pickle", ".pkl", ".npz",
                     ".npy", ".safetensors", ".png", ".jpg", ".pdf", ".pptx", ".svg"}
PERSONAL_PATH = re.compile("/gpfs/work/che/" + "ziqiwang21|/home/" + "sa/")


def main() -> None:
    missing = [relative for relative in REQUIRED if not (ROOT / relative).is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")

    oversized = []
    excluded = []
    personal_paths = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "runs" in path.relative_to(ROOT).parts:
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            excluded.append(str(relative))
        if path.stat().st_size > 50 * 1024 * 1024:
            oversized.append(str(relative))
        if path.suffix.lower() in {
            ".py",
            ".md",
            ".yml",
            ".yaml",
            ".tsv",
            ".csv",
            ".json",
        }:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if PERSONAL_PATH.search(text):
                personal_paths.append(str(relative))

    if excluded:
        raise SystemExit(f"excluded binary artifacts are tracked locally: {excluded}")
    if oversized:
        raise SystemExit(f"files exceed 50 MiB: {oversized}")
    if personal_paths:
        raise SystemExit(f"personal absolute paths remain: {personal_paths}")

    tex = (ROOT / "paper_v3.0_all_in_one.tex").read_text(encoding="utf-8")
    bib = (ROOT / "references.bib").read_text(encoding="utf-8")
    citation_keys = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", tex)
        for key in group.split(",")
    }
    bibliography_keys = set(re.findall(r"^@[A-Za-z]+\{([^,]+),", bib, re.MULTILINE))
    missing_citations = sorted(citation_keys - bibliography_keys)
    if missing_citations:
        raise SystemExit(f"citation keys missing from references.bib: {missing_citations}")

    print(
        "repository check OK "
        f"required={len(REQUIRED)} citations={len(citation_keys)}"
    )


if __name__ == "__main__":
    main()
