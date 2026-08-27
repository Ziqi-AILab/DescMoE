#!/usr/bin/env python3
"""Lightweight checks for the fixed descriptor assignment used in the paper."""

from __future__ import annotations

import sys
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MoleSG" / "pretrain"))

from prior_moe import _DEFAULT_CHEM_EDGES, assign_expert_ids  # noqa: E402


def main() -> None:
    smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None

    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    logp = float(Crippen.MolLogP(mol))
    assert abs(tpsa - 37.30) < 1.0e-6
    assert abs(logp - 3.0732) < 1.0e-4
    assert assign_expert_ids([smiles], "tpsa") == [1]
    assert assign_expert_ids([smiles], "logp") == [5]
    assert _DEFAULT_CHEM_EDGES["tpsa"] == [20, 40, 60, 80, 100, 120, 140]
    assert _DEFAULT_CHEM_EDGES["logp"] == [-1, 0, 1, 2, 3, 4, 5]
    print(
        "descriptor assignment OK "
        f"rdkit={rdBase.rdkitVersion} tpsa={tpsa:.2f} logp={logp:.4f}"
    )


if __name__ == "__main__":
    main()
