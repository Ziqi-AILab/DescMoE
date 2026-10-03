"""Cross-stage deterministic random assignments for matched MoFE controls."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from rdkit import Chem


DESCRIPTOR_CODES = {
    "tpsa": 0x54505341,
    "logp": 0x4C4F4750,
}


def canonicalize_smiles(smiles: str) -> str:
    """Return the RDKit canonical SMILES used as the molecule-level key."""
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True)


def stable_random_score(canonical_smiles: str, descriptor: str,
                        seed: int = 42) -> float:
    """Map a canonical molecule key to a reproducible PCG64 score in [0, 1)."""
    if descriptor not in DESCRIPTOR_CODES:
        raise ValueError(f"Unsupported descriptor: {descriptor}")
    key_bytes = canonical_smiles.encode("utf-8")
    entropy = [int(seed), DESCRIPTOR_CODES[descriptor], *key_bytes]
    sequence = np.random.SeedSequence(entropy)
    return float(np.random.Generator(np.random.PCG64(sequence)).random())


def stable_random_scores(canonical_smiles: Iterable[str], descriptor: str,
                         seed: int = 42) -> np.ndarray:
    """Vector wrapper that reuses scores for duplicate canonical structures."""
    cache: dict[str, float] = {}
    scores = []
    for smiles in canonical_smiles:
        if smiles not in cache:
            cache[smiles] = stable_random_score(smiles, descriptor, seed)
        scores.append(cache[smiles])
    return np.asarray(scores, dtype=np.float64)


def occupancy_matched_score_edges(scores: np.ndarray,
                                  target_counts: Iterable[int]) -> list[float]:
    """Choose score cut points nearest to target cumulative occupancies.

    Identical canonical structures receive identical scores and are never split.
    Consequently, achieved occupancy can differ slightly from the target when a
    duplicated structure straddles a requested boundary.
    """
    values = np.asarray(scores, dtype=np.float64)
    counts = np.asarray(list(target_counts), dtype=np.int64)
    if values.ndim != 1 or counts.shape != (8,):
        raise ValueError("Expected one-dimensional scores and eight target counts")
    if int(counts.sum()) != len(values):
        raise ValueError("Target occupancy does not sum to the score count")
    if len(values) == 0:
        raise ValueError("Cannot derive boundaries from an empty score array")

    unique_scores, unique_counts = np.unique(values, return_counts=True)
    cumulative = np.cumsum(unique_counts)
    desired = np.cumsum(counts)[:-1]
    edges: list[float] = []
    previous_split = 0
    for target in desired:
        candidate_splits = np.arange(len(unique_scores) + 1)
        candidate_counts = np.concatenate(([0], cumulative))
        distances = np.abs(candidate_counts - target)
        distances[candidate_splits < previous_split] = len(values) + 1
        split = int(candidate_splits[np.argmin(distances)])
        previous_split = split
        if split == 0:
            edge = float(np.nextafter(unique_scores[0], -np.inf))
        elif split == len(unique_scores):
            edge = float(np.nextafter(unique_scores[-1], np.inf))
        else:
            edge = float((unique_scores[split - 1] + unique_scores[split]) / 2.0)
        edges.append(edge)
    return edges


def assign_scores(scores: np.ndarray, edges: Iterable[float]) -> np.ndarray:
    """Assign random scores to one of eight frozen regions."""
    return np.searchsorted(
        np.asarray(list(edges), dtype=np.float64),
        np.asarray(scores, dtype=np.float64),
        side="right",
    ).astype(np.int64)


__all__ = [
    "assign_scores",
    "canonicalize_smiles",
    "occupancy_matched_score_edges",
    "stable_random_score",
    "stable_random_scores",
]
