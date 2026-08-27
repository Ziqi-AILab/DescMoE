#!/usr/bin/env python3
"""Build and verify the chemistry-specific assets used in Figure 1.

Run with the supplied MoleSG environment, which contains RDKit 2022.09.5:

    conda run -n MoleSG python analysis/build_verified_overview_chemistry.py

The script calls the same ``assign_expert_ids`` function used by the model. It
does not infer a region from a label embedded in the figure.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("MOLESG_ROOT", ROOT / "MoleSG"))
sys.path.insert(0, str(PROJECT_ROOT / "pretrain"))

from prior_moe import _DEFAULT_CHEM_EDGES, assign_expert_ids  # noqa: E402


NAME = "Ibuprofen"
INPUT_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
TPSA_KAN_BRANCHES = {0, 1, 2, 3, 5}
LOGP_KAN_BRANCHES = {1, 2, 3, 4}


def branch_family(region: int, selected: set[int]) -> str:
    return "KAN" if region in selected else "MLP"


def draw_molecule(mol: Chem.Mol, svg_path: Path, png_path: Path) -> None:
    rdDepictor.Compute2DCoords(mol)

    svg = rdMolDraw2D.MolDraw2DSVG(720, 340)
    svg.drawOptions().padding = 0.08
    svg.drawOptions().bondLineWidth = 2.2
    svg.DrawMolecule(mol)
    svg.FinishDrawing()
    svg_path.write_text(svg.GetDrawingText(), encoding="utf-8")

    png = rdMolDraw2D.MolDraw2DCairo(1440, 680)
    png.drawOptions().padding = 0.08
    png.drawOptions().bondLineWidth = 3.2
    png.DrawMolecule(mol)
    png.FinishDrawing()
    png_path.write_bytes(png.GetDrawingText())


def main() -> None:
    asset_dir = ROOT / "figures" / "assets"
    report_dir = ROOT / "reports"
    asset_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    mol = Chem.MolFromSmiles(INPUT_SMILES)
    if mol is None:
        raise ValueError(f"RDKit could not parse {INPUT_SMILES}")

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    mol_logp = float(Crippen.MolLogP(mol))
    tpsa_region = int(assign_expert_ids([canonical_smiles], "tpsa")[0])
    logp_region = int(assign_expert_ids([canonical_smiles], "logp")[0])

    # These assertions deliberately fail if the project implementation,
    # structure, or descriptor version changes.
    assert rdBase.rdkitVersion == "2022.09.5"
    assert abs(tpsa - 37.30) < 1.0e-6
    assert abs(mol_logp - 3.0732) < 1.0e-4
    assert tpsa_region == 1
    assert logp_region == 5
    assert branch_family(tpsa_region, TPSA_KAN_BRANCHES) == "KAN"
    assert branch_family(logp_region, LOGP_KAN_BRANCHES) == "MLP"

    record = {
        "molecule": NAME,
        "example_status": "illustrative molecule; not claimed as a ZINC 250K row",
        "input_smiles": INPUT_SMILES,
        "canonical_smiles": canonical_smiles,
        "molecular_formula": formula,
        "rdkit_version": rdBase.rdkitVersion,
        "tpsa_angstrom_squared": tpsa,
        "tpsa_edges": _DEFAULT_CHEM_EDGES["tpsa"],
        "tpsa_region_zero_based": tpsa_region,
        "tpsa_branch_family": branch_family(tpsa_region, TPSA_KAN_BRANCHES),
        "mol_logp": mol_logp,
        "mol_logp_edges": _DEFAULT_CHEM_EDGES["logp"],
        "mol_logp_region_zero_based": logp_region,
        "mol_logp_branch_family": branch_family(logp_region, LOGP_KAN_BRANCHES),
        "tpsa_selected_kan_regions": sorted(TPSA_KAN_BRANCHES),
        "mol_logp_selected_kan_regions": sorted(LOGP_KAN_BRANCHES),
        "assignment_code": str(PROJECT_ROOT / "pretrain" / "prior_moe.py"),
    }

    (report_dir / "figure1_chemistry_audit.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    with (report_dir / "figure1_chemistry_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)

    draw_molecule(
        mol,
        asset_dir / "ibuprofen_rdkit_2022_09_5.svg",
        asset_dir / "ibuprofen_rdkit_2022_09_5.png",
    )

    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
