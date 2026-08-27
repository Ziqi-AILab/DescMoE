#!/usr/bin/env python3
"""Audit the descriptor assignments used by MoleSG pretraining.

This is a local CPU audit. It recomputes TPSA and Wildman-Crippen MolLogP for
all ZINC 250K rows with the RDKit version used by the project, applies the
fixed project boundaries, and compares the results with ``expert_ids.npz``.
It also checks that the preprocessed molecule filenames map to the same source
row indices used by the assignment arrays.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]
MOLESG_ROOT = Path(os.environ.get("MOLESG_ROOT", ROOT / "MoleSG"))
DATA_ROOT = MOLESG_ROOT / "Data" / "zinc15"
SMILES_PATH = DATA_ROOT / "zinc15_250K.csv"
ASSIGNMENT_PATH = DATA_ROOT / "expert_ids.npz"
PREPROCESSED_ROOT = DATA_ROOT / "zinc15_0.25_geo" / "preprocess"

sys.path.insert(0, str(MOLESG_ROOT / "pretrain"))
from prior_moe import _DEFAULT_CHEM_EDGES  # noqa: E402


TARGET_NAME = "Ibuprofen"
TARGET_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
FILE_INDEX_RE = re.compile(r"mol(?P<index>\d+)\.npz$")


def fixed_region(value: float, edges: list[float]) -> int:
    return int(np.digitize(value, edges, right=False))


def main() -> None:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    stored = np.load(ASSIGNMENT_PATH)
    stored_tpsa = np.asarray(stored["tpsa"], dtype=np.int64)
    stored_logp = np.asarray(stored["logp"], dtype=np.int64)
    expected_rows = len(stored_tpsa)
    if len(stored_logp) != expected_rows:
        raise ValueError("TPSA and MolLogP assignment arrays differ in length")

    target = Chem.MolToSmiles(Chem.MolFromSmiles(TARGET_SMILES), canonical=True)
    target_source_rows: list[int] = []
    mismatch_rows = {"tpsa": [], "logp": []}
    invalid_rows: list[int] = []
    counts = {"tpsa": Counter(), "logp": Counter()}
    descriptor_ranges = {
        "tpsa": [float("inf"), float("-inf")],
        "logp": [float("inf"), float("-inf")],
    }

    with SMILES_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for source_row_index, row in enumerate(reader):
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                invalid_rows.append(source_row_index)
                continue

            canonical = Chem.MolToSmiles(mol, canonical=True)
            if canonical == target:
                target_source_rows.append(source_row_index)

            values = {
                "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
                "logp": float(Crippen.MolLogP(mol)),
            }
            expected = {
                key: fixed_region(value, list(_DEFAULT_CHEM_EDGES[key]))
                for key, value in values.items()
            }
            observed = {
                "tpsa": int(stored_tpsa[source_row_index]),
                "logp": int(stored_logp[source_row_index]),
            }
            for key in ("tpsa", "logp"):
                counts[key][observed[key]] += 1
                descriptor_ranges[key][0] = min(descriptor_ranges[key][0], values[key])
                descriptor_ranges[key][1] = max(descriptor_ranges[key][1], values[key])
                if observed[key] != expected[key]:
                    mismatch_rows[key].append(source_row_index)

    source_rows = sum(counts["tpsa"].values())
    if source_rows + len(invalid_rows) != expected_rows:
        raise ValueError(
            f"CSV rows ({source_rows + len(invalid_rows)}) do not match assignment "
            f"rows ({expected_rows})"
        )

    processed_indices: list[int] = []
    malformed_processed_names: list[str] = []
    for path in PREPROCESSED_ROOT.iterdir():
        match = FILE_INDEX_RE.search(path.name)
        if match:
            processed_indices.append(int(match.group("index")))
        else:
            malformed_processed_names.append(path.name)
    processed_set = set(processed_indices)
    expected_set = set(range(expected_rows))
    missing_processed = sorted(expected_set - processed_set)
    unexpected_processed = sorted(processed_set - expected_set)

    record = {
        "rdkit_version": rdBase.rdkitVersion,
        "source_csv": str(SMILES_PATH),
        "assignment_file": str(ASSIGNMENT_PATH),
        "preprocessed_directory": str(PREPROCESSED_ROOT),
        "source_rows": expected_rows,
        "valid_molecules": source_rows,
        "invalid_molecules": len(invalid_rows),
        "tpsa_edges": list(_DEFAULT_CHEM_EDGES["tpsa"]),
        "logp_edges": list(_DEFAULT_CHEM_EDGES["logp"]),
        "tpsa_assignment_mismatches": len(mismatch_rows["tpsa"]),
        "logp_assignment_mismatches": len(mismatch_rows["logp"]),
        "tpsa_counts": [int(counts["tpsa"][i]) for i in range(8)],
        "logp_counts": [int(counts["logp"][i]) for i in range(8)],
        "tpsa_range": descriptor_ranges["tpsa"],
        "logp_range": descriptor_ranges["logp"],
        "preprocessed_files": len(processed_indices),
        "unique_preprocessed_indices": len(processed_set),
        "missing_preprocessed_indices": len(missing_processed),
        "unexpected_preprocessed_indices": len(unexpected_processed),
        "malformed_preprocessed_filenames": len(malformed_processed_names),
        "illustrative_molecule": TARGET_NAME,
        "illustrative_molecule_in_zinc250k": bool(target_source_rows),
        "illustrative_molecule_source_rows": target_source_rows,
    }

    # The manuscript and overview are allowed to proceed only when the model's
    # stored assignments reproduce the documented fixed-boundary rule exactly.
    if rdBase.rdkitVersion != "2022.09.5":
        print(
            f"Warning: manuscript values used RDKit 2022.09.5, current version is "
            f"{rdBase.rdkitVersion}",
            file=sys.stderr,
        )
    assert not invalid_rows
    assert not mismatch_rows["tpsa"]
    assert not mismatch_rows["logp"]
    assert len(processed_indices) == expected_rows
    assert len(processed_set) == expected_rows
    assert not missing_processed
    assert not unexpected_processed
    assert not malformed_processed_names

    (reports / "pretraining_descriptor_assignment_audit.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    with (reports / "pretraining_descriptor_assignment_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["descriptor", "region_zero_based", "count", "fraction"])
        for descriptor in ("tpsa", "logp"):
            for region in range(8):
                count = int(counts[descriptor][region])
                writer.writerow([descriptor, region, count, count / expected_rows])

    report = f"""# Pretraining Descriptor Assignment Audit

The audit recomputed TPSA and Wildman-Crippen MolLogP for all {expected_rows:,}
ZINC rows using RDKit {rdBase.rdkitVersion}. The fixed project boundaries were
then compared row by row with the arrays loaded by pretraining.

| Check | Result |
| --- | ---: |
| Valid molecules | {source_rows:,}/{expected_rows:,} |
| TPSA assignment mismatches | {len(mismatch_rows['tpsa']):,} |
| MolLogP assignment mismatches | {len(mismatch_rows['logp']):,} |
| Indexed preprocessed files | {len(processed_indices):,}/{expected_rows:,} |
| Missing or unexpected preprocessing indices | {len(missing_processed) + len(unexpected_processed):,} |

The arrays used by pretraining therefore implement the documented fixed
boundaries exactly. The separate quantile-bucketing code in
`descriptor_analysis.py` belongs to the earlier descriptor-screening analysis
and does not generate `expert_ids.npz`.

{TARGET_NAME} is {'present in' if target_source_rows else 'not present in'} the
ZINC 250K source rows. Figure 1 consequently labels it as an illustrative
molecule rather than a sampled pretraining record.
"""
    (reports / "pretraining_descriptor_assignment_audit.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
