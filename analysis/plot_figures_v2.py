#!/usr/bin/env python3
"""Render deterministic JCIM v2 main and Supporting Information figures."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT_ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = OUT_ROOT / "tables_v2"
FIGURE_ROOT = OUT_ROOT / "figures"
FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = ["bbbp", "tox21", "toxcast", "sider", "clintox", "bace", "hiv", "muv"]
DATASET_LABELS = {
    "bbbp": "BBBP",
    "tox21": "Tox21",
    "toxcast": "ToxCast",
    "sider": "SIDER",
    "clintox": "ClinTox",
    "bace": "BACE",
    "hiv": "HIV",
    "muv": "MUV",
}
REGION_COLORS = [
    "#355070",
    "#6d597a",
    "#b56576",
    "#e56b6f",
    "#d99b45",
    "#2a9d8f",
    "#457b9d",
    "#8d99ae",
]
MODEL_COLORS = {
    "Base": "#333333",
    "I1": "#4477AA",
    "I1+I2": "#EE7733",
    "I1+I3": "#228833",
    "Full": "#AA3377",
    "all-KAN": "#CCBB44",
}
PARTITION_COLORS = {"TPSA": "#4477AA", "MolLogP": "#CC6677"}
ROLE_ORDER = ["Base", "I1", "I1+I2", "I1+I3", "Full"]
MODEL_SETS = {
    "TPSA": {
        "Base": "p3_base_vanilla_n8aligned",
        "I1": "p3_tpsa_mofe_last_ffn_n8aligned",
        "I1+I2": "p5_tpsa_mofe_last_selected_kan_n8aligned",
        "I1+I3": "p3_tpsa_mofe_last_ffn_conloss_n8aligned",
        "Full": "p5_tpsa_mofe_last_selected_kan_conloss_n8aligned",
        "all-KAN": "p3_tpsa_mofe_last_all_kan_n8aligned",
    },
    "MolLogP": {
        "Base": "p3_base_vanilla_n8aligned",
        "I1": "p3_logp_mofe_last_ffn_n8aligned",
        "I1+I2": "p5_logp_mofe_last_selected_kan_n8aligned",
        "I1+I3": "p3_logp_mofe_last_ffn_conloss_n8aligned",
        "Full": "p5_logp_mofe_last_selected_kan_conloss_n8aligned",
        "all-KAN": "p3_logp_mofe_last_all_kan_n8aligned",
    },
}


def rows(path: str) -> list[dict]:
    with (TABLE_ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def maybe_float(value: str | None) -> float:
    return float(value) if value not in ("", None) else math.nan


def set_style() -> None:
    plt.rcParams.update(
        {
        "font.family": "Nimbus Sans",
            "font.size": 8.3,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURE_ROOT / f"{stem}.pdf")
    fig.savefig(FIGURE_ROOT / f"{stem}.png", dpi=600)
    plt.close(fig)


def heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    center: float | None = None,
    fmt: str = ".3f",
    annotate: bool = True,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    if center is not None and finite.size:
        bound = max(abs(float(finite.min()) - center), abs(float(finite.max()) - center))
        norm = TwoSlopeNorm(vmin=center - bound, vcenter=center, vmax=center + bound)
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
    else:
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=35, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_title(title)
    if annotate:
        threshold = np.nanmedian(finite) if finite.size else 0
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                if not np.isfinite(value):
                    text = ""
                else:
                    text = format(value, fmt)
                color = "white" if np.isfinite(value) and value < threshold else "black"
                ax.text(
                    col_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color=color,
                )
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)


def rolling_epoch_quantiles(
    epoch: np.ndarray,
    values: np.ndarray,
    half_window: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.empty_like(values, dtype=float)
    q1 = np.empty_like(values, dtype=float)
    q3 = np.empty_like(values, dtype=float)
    for index, current_epoch in enumerate(epoch):
        mask = np.abs(epoch - current_epoch) <= half_window
        q1[index], median[index], q3[index] = np.quantile(
            values[mask],
            [0.25, 0.50, 0.75],
        )
    return median, q1, q3


def workflow_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.3, 3.7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    asset_root = FIGURE_ROOT / "assets"
    chemical_space = plt.imread(asset_root / "image2_chemical_space.png")
    branch_visual = plt.imread(asset_root / "image2_conditional_branches.png")
    ax.imshow(chemical_space, extent=(0.05, 2.65, 2.72, 5.55), aspect="auto")
    ax.imshow(branch_visual, extent=(5.65, 8.75, 3.08, 5.12), aspect="auto")

    boxes = [
        (0.25, 0.72, 2.25, 1.15, "Graph and SMILES\npretraining", "#d9e8f5"),
        (3.05, 3.42, 2.15, 1.22, "MolLogP or TPSA\nfixed intervals", "#f7e1b5"),
        (5.95, 0.72, 2.35, 1.15, "Partial KAN\nbranch functions", "#f4c7b9"),
        (8.65, 0.72, 2.35, 1.15, "Descriptor-aware\nConLoss", "#dfd3ee"),
        (9.35, 3.42, 2.35, 1.22, "Scaffold-split\nproperty prediction", "#d4e4f7"),
    ]
    for x, y, width, height, label, color in boxes:
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor=color,
                edgecolor="#333333",
                linewidth=0.9,
            )
        )
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center")
    for start, end in [
        ((2.65, 4.03), (3.05, 4.03)),
        ((5.20, 4.03), (5.67, 4.03)),
        ((8.76, 4.03), (9.35, 4.03)),
        ((7.18, 3.02), (7.18, 1.87)),
        ((8.30, 1.30), (8.65, 1.30)),
        ((1.38, 2.70), (1.38, 1.87)),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.0,
                color="#444444",
            )
        )
    ax.text(
        6.0,
        5.65,
        "Descriptor-guided chemical-space partition",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(1.35, 2.55, "Heterogeneous chemical space", ha="center", fontsize=7.5)
    ax.text(7.20, 2.73, "Descriptor-conditioned branches", ha="center", fontsize=7.5)
    ax.text(
        5.95,
        0.22,
        "One fixed descriptor region selects one parameterized computation branch.",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    save(fig, "fig01_workflow")


def descriptor_figure() -> None:
    profile = rows("descriptor_region_profiles.csv")
    joint = rows("descriptor_joint_sample.csv")
    fig = plt.figure(figsize=(7.4, 5.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], wspace=0.40, hspace=0.42)
    profile_axes = []
    for col, partition in enumerate(("tpsa", "logp")):
        subset = sorted(
            [row for row in profile if row["partition"] == partition],
            key=lambda row: int(row["region"]),
        )
        counts = np.asarray([int(row["count"]) for row in subset])
        ax = fig.add_subplot(grid[0, col])
        ax.bar(range(8), counts / counts.sum() * 100, color=REGION_COLORS)
        ax.set_xticks(range(8), [f"R{i}" for i in range(8)])
        ax.set_ylabel("Molecules (%)")
        ax.set_title("TPSA occupancy" if partition == "tpsa" else "MolLogP occupancy")

        metrics = [
            "mol_wt_mean",
            "hbd_mean",
            "hba_mean",
            "rotatable_bonds_mean",
            "ring_count_mean",
            "aromatic_atom_fraction_mean",
            "heteroatom_fraction_mean",
            "formal_charge_mean",
        ]
        labels = ["MW", "HBD", "HBA", "RotB", "Rings", "Arom.", "Hetero", "Charge"]
        matrix = np.asarray(
            [[maybe_float(row[metric]) for row in subset] for metric in metrics]
        )
        matrix[:, counts < 10] = np.nan
        row_mean = np.nanmean(matrix, axis=1, keepdims=True)
        row_sd = np.nanstd(matrix, axis=1, keepdims=True)
        row_sd[row_sd == 0] = 1
        standardized = (matrix - row_mean) / row_sd
        ax_profile = fig.add_subplot(grid[1, col])
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#e5e5e5")
        image = ax_profile.imshow(
            np.ma.masked_invalid(standardized),
            aspect="auto",
            cmap=cmap,
            vmin=-2,
            vmax=2,
        )
        ax_profile.set_xticks(range(8), [f"R{i}" for i in range(8)])
        ax_profile.set_yticks(range(len(labels)), labels)
        ax_profile.set_title("Standardized region profile")
        profile_axes.append((ax_profile, image))

    joint_ax = fig.add_subplot(grid[0, 2])
    tpsa = np.asarray([float(row["tpsa"]) for row in joint])
    logp = np.asarray([float(row["logp"]) for row in joint])
    density = joint_ax.hexbin(
        logp,
        tpsa,
        gridsize=42,
        mincnt=1,
        cmap="cividis",
        bins="log",
    )
    joint_ax.set_xlabel("MolLogP")
    joint_ax.set_ylabel(r"TPSA ($\mathrm{\AA^2}$)")
    joint_ax.set_title("Joint descriptor space")
    fig.colorbar(density, ax=joint_ax, fraction=0.046, pad=0.03, label="log count")

    relation_ax = fig.add_subplot(grid[1, 2])
    relation_metrics = ["hbd", "hba", "aromatic_atom_fraction", "heteroatom_fraction"]
    relation_labels = ["HBD", "HBA", "Arom.", "Hetero"]
    values = np.asarray(
        [
            [float(row[key]) for row in joint]
            for key in relation_metrics
        ]
    )
    descriptors = np.vstack([tpsa, logp])
    corr = np.corrcoef(np.vstack([descriptors, values]))[:2, 2:]
    image = relation_ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    relation_ax.set_xticks(range(4), relation_labels, rotation=30, ha="right")
    relation_ax.set_yticks([0, 1], ["TPSA", "MolLogP"])
    relation_ax.set_title("Descriptor-property correlation")
    for i in range(2):
        for j in range(4):
            relation_ax.text(j, i, f"{corr[i, j]:+.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=relation_ax, fraction=0.046, pad=0.03)
    fig.colorbar(
        profile_axes[-1][1],
        ax=[item[0] for item in profile_axes],
        orientation="horizontal",
        fraction=0.055,
        pad=0.17,
        aspect=35,
        label="Region-profile z score",
    )
    save(fig, "fig02_descriptor_space_profiles")


def layer_figure() -> None:
    data = rows("dataset_5fold_summary.csv")
    macro = rows("layer_screen_summary.csv")
    models = [
        ("Last", "p2_mofe_ffn_logp_last", "#228833", "o"),
        ("Odd", "p2_mofe_ffn_logp_odd", "#4477AA", "s"),
        ("Even", "p2_mofe_ffn_logp_even", "#EE7733", "^"),
    ]
    lookup = {
        (row["model"], row["dataset"]): row
        for row in data
        if row["complete_5fold"] == "1"
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.45), gridspec_kw={"width_ratios": [3.2, 1]})
    x = np.arange(len(DATASETS))
    for offset, (label, model, color, marker) in enumerate(models):
        means = [float(lookup[(model, dataset)]["auc_mean"]) for dataset in DATASETS]
        errors = [float(lookup[(model, dataset)]["auc_sd"]) for dataset in DATASETS]
        axes[0].errorbar(
            x + (offset - 1) * 0.08,
            means,
            yerr=errors,
            marker=marker,
            markersize=3.6,
            linestyle="none",
            capsize=2,
            color=color,
            label=label,
        )
    axes[0].set_ylim(0.55, 1.0)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_xticks(x, [DATASET_LABELS[item] for item in DATASETS], rotation=28, ha="right")
    axes[0].set_title("Per-dataset five-fold results")
    axes[0].legend(frameon=False, ncol=3)

    macro_lookup = {row["placement"]: row for row in macro}
    placements = ["last", "odd", "even"]
    bars = axes[1].bar(
        range(3),
        [float(macro_lookup[item]["macro_auc"]) for item in placements],
        color=["#228833", "#4477AA", "#EE7733"],
    )
    axes[1].set_xticks(range(3), ["Last", "Odd", "Even"], rotation=25, ha="right")
    axes[1].set_ylim(0.75, 0.79)
    axes[1].set_ylabel("Eight-task macro AUC")
    axes[1].set_title("Architecture selection")
    for bar, placement in zip(bars, placements):
        delta = 100 * float(macro_lookup[placement]["delta_vs_last"])
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0007,
            f"{delta:+.2f} pp",
            ha="center",
            fontsize=6.5,
        )
    fig.tight_layout()
    save(fig, "fig03_layer_placement")


def kan_figure() -> None:
    summary = rows("kan_screen_summary.csv")
    profile = rows("descriptor_region_profiles.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.45))
    for ax, partition in zip(axes, ("tpsa", "logp")):
        subset = sorted(
            [row for row in summary if row["partition"] == partition],
            key=lambda row: int(row["k"]),
        )
        profile_subset = sorted(
            [row for row in profile if row["partition"] == partition],
            key=lambda row: int(row["region"]),
        )
        deltas = np.asarray([100 * float(row["delta"]) for row in subset])
        selected = deltas > 0
        ax.bar(
            range(8),
            deltas,
            color=["#228833" if flag else "#aaaaaa" for flag in selected],
        )
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(range(8), [f"k{i}" for i in range(8)])
        ax.set_ylabel("Macro AUC delta (pp)")
        ax.set_xlabel("Single KAN branch")
        ax.set_title("TPSA" if partition == "tpsa" else "MolLogP")
        occupancy_ax = ax.twinx()
        occupancy = [100 * float(row["fraction"]) for row in profile_subset]
        occupancy_ax.plot(
            range(8),
            occupancy,
            color="#555555",
            marker="o",
            markersize=3,
            linewidth=0.8,
            alpha=0.75,
        )
        occupancy_ax.set_ylabel("Pretraining occupancy (%)", color="#555555")
        occupancy_ax.tick_params(axis="y", labelcolor="#555555")
    fig.suptitle("Branch-wise nonlinear function screening", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig04_kan_selection")


def final_performance_figure() -> None:
    data = rows("dataset_5fold_summary.csv")
    lookup = {(row["model"], row["dataset"]): row for row in data}
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.1), sharex="col")
    for col, (partition, models) in enumerate(MODEL_SETS.items()):
        x = np.arange(len(DATASETS))
        for role_index, role in enumerate(ROLE_ORDER):
            model = models[role]
            means = [maybe_float(lookup[(model, dataset)]["auc_mean"]) for dataset in DATASETS]
            errors = [maybe_float(lookup[(model, dataset)]["auc_sd"]) for dataset in DATASETS]
            offset = (role_index - 2) * 0.055
            axes[0, col].errorbar(
                x + offset,
                means,
                yerr=errors,
                linestyle="none",
                marker="o",
                markersize=3,
                capsize=1.5,
                color=MODEL_COLORS[role],
                label=role,
            )
        axes[0, col].set_title(partition)
        axes[0, col].set_ylabel("ROC-AUC")
        axes[0, col].set_ylim(0.55, 1.0)

        base_model = models["Base"]
        for role_index, role in enumerate(ROLE_ORDER[1:]):
            model = models[role]
            delta = []
            for dataset in DATASETS:
                left = maybe_float(lookup[(model, dataset)]["auc_mean"])
                base = maybe_float(lookup[(base_model, dataset)]["auc_mean"])
                delta.append(100 * (left - base) if np.isfinite(left + base) else math.nan)
            axes[1, col].scatter(
                x + (role_index - 1.5) * 0.07,
                delta,
                s=15,
                color=MODEL_COLORS[role],
                label=role,
            )
        axes[1, col].axhline(0, color="#333333", linewidth=0.8)
        axes[1, col].set_ylabel("Delta vs local Base (pp)")
        axes[1, col].set_xticks(
            x,
            [DATASET_LABELS[item] for item in DATASETS],
            rotation=28,
            ha="right",
        )
    axes[0, 0].legend(frameon=False, ncol=3, loc="lower left")
    axes[1, 0].legend(frameon=False, ncol=2, loc="lower left")
    fig.suptitle("Legacy eight-layer-aligned downstream snapshot", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig05_final_n8_performance")


def conditional_ablation_figure() -> None:
    paired = [
        row
        for row in rows("paired_comparisons.csv")
        if "::" not in row["comparison"]
        and row["comparison"]
        in {
            "I1 vs Base",
            "I2 conditional effect",
            "I3 conditional effect",
            "Full vs Base",
        }
    ]
    interaction = {
        row["partition"]: row
        for row in rows("conditional_interaction.csv")
        if row["scope"] == "macro"
    }
    effects = ["I1 vs Base", "I2 effect", "I3 effect", "Interaction", "Full vs Base"]
    source_labels = {
        "I1 vs Base": "I1 vs Base",
        "I2 effect": "I2 conditional effect",
        "I3 effect": "I3 conditional effect",
        "Full vs Base": "Full vs Base",
    }
    lookup = {(row["partition"], row["comparison"]): row for row in paired}
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for offset, partition in zip((-0.10, 0.10), ("TPSA", "MolLogP")):
        points, lows, highs = [], [], []
        for effect in effects:
            if effect == "Interaction":
                row = interaction[partition]
                point = float(row["interaction"])
                low = float(row["ci95_low"])
                high = float(row["ci95_high"])
            else:
                row = lookup[(partition, source_labels[effect])]
                point = float(row["macro_delta"])
                low = float(row["ci95_low"])
                high = float(row["ci95_high"])
            points.append(100 * point)
            lows.append(100 * (point - low))
            highs.append(100 * (high - point))
        y = np.arange(len(effects)) + offset
        ax.errorbar(
            points,
            y,
            xerr=np.vstack([lows, highs]),
            fmt="o",
            capsize=2.5,
            markersize=4,
            color=PARTITION_COLORS[partition],
            label=partition,
        )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(range(len(effects)), effects)
    ax.invert_yaxis()
    ax.set_xlabel("Matched macro AUC delta (percentage points)")
    ax.set_title("Conditional component effects with hierarchical bootstrap 95% CI")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig06_conditional_ablation")


def tsne_panel(
    ax: plt.Axes,
    subset: list[dict],
    title: str,
    point_size: float,
) -> None:
    for region in range(8):
        region_rows = [row for row in subset if int(row["region"]) == region]
        if not region_rows:
            continue
        ax.scatter(
            [float(row["tsne_x"]) for row in region_rows],
            [float(row["tsne_y"]) for row in region_rows],
            s=point_size,
            alpha=0.68,
            color=REGION_COLORS[region],
            linewidths=0,
        )
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb")
        spine.set_linewidth(0.5)


def tsne_main_figure() -> None:
    data = rows("embedding_tsne_coordinates.csv")
    panels = [
        ("tpsa", "I1"),
        ("tpsa", "I1+I2"),
        ("tpsa", "I1+I3"),
        ("tpsa", "Full"),
        ("logp", "I1"),
        ("logp", "I1+I2"),
        ("logp", "I1+I3"),
        ("logp", "Full"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(7.5, 4.0))
    for ax, (partition, role) in zip(axes.ravel(), panels):
        subset = [
            row
            for row in data
            if row["dataset"] == "tox21"
            and row["partition"] == partition
            and row["model_role"] == role
        ]
        label = ("TPSA " if partition == "tpsa" else "MolLogP ") + role
        tsne_panel(ax, subset, label, 4.0)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=4,
            color=REGION_COLORS[index],
            label=f"R{index}",
        )
        for index in range(8)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False)
    fig.suptitle("Tox21 fold-1 embeddings colored by descriptor-defined region", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    save(fig, "fig07_embedding_tsne_tox21")


def compute_pareto_figure() -> None:
    macro = {row["model"]: row for row in rows("matched_model_macro_summary.csv")}
    complexity = {row["configuration"]: row for row in rows("architecture_complexity.csv")}
    runtime = {row["model"]: row for row in rows("pretraining_runtime_summary.csv")}
    points = [
        ("Base", "p3_base_vanilla_n8aligned", "Vanilla", "p3_base_vanilla"),
        ("TPSA I1", "p3_tpsa_mofe_last_ffn_n8aligned", "MoFE-last all-MLP", "p3_tpsa_mofe_last_ffn"),
        ("MolLogP I1", "p3_logp_mofe_last_ffn_n8aligned", "MoFE-last all-MLP", "p3_logp_mofe_last_ffn"),
        ("TPSA selected", "p5_tpsa_mofe_last_selected_kan_n8aligned", "TPSA selected-KAN", "p5_tpsa_mofe_last_selected_kan"),
        ("MolLogP selected", "p5_logp_mofe_last_selected_kan_n8aligned", "MolLogP selected-KAN", "p5_logp_mofe_last_selected_kan"),
        ("TPSA Full", "p5_tpsa_mofe_last_selected_kan_conloss_n8aligned", "TPSA selected-KAN", "p5_tpsa_mofe_last_selected_kan_conloss"),
        ("MolLogP Full", "p5_logp_mofe_last_selected_kan_conloss_n8aligned", "MolLogP selected-KAN", "p5_logp_mofe_last_selected_kan_conloss"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.45))
    for index, (label, result_model, config, pretrain_model) in enumerate(points):
        auc = float(macro[result_model]["matched_macro_auc"])
        flops = int(complexity[config]["active_ffn_flops_per_token"]) / 1e6
        minutes = float(runtime[pretrain_model]["median_active_seconds_per_epoch"]) / 60
        stored = int(complexity[config]["stored_total_parameters_analytical"]) / 1e6
        color = REGION_COLORS[index]
        size = 18 + stored * 4
        axes[0].scatter(flops, auc, s=size, color=color, alpha=0.85)
        axes[1].scatter(minutes, auc, s=size, color=color, alpha=0.85)
        axes[0].annotate(label, (flops, auc), xytext=(3, 2), textcoords="offset points", fontsize=6)
        axes[1].annotate(label, (minutes, auc), xytext=(3, 2), textcoords="offset points", fontsize=6)
    axes[0].set_xlabel("Analytical active FFN FLOPs/token (M)")
    axes[1].set_xlabel("Median active pretraining min/epoch")
    axes[0].set_ylabel("Matched macro ROC-AUC")
    axes[1].set_ylabel("Matched macro ROC-AUC")
    axes[0].set_title("Arithmetic cost")
    axes[1].set_title("Observed active epoch time")
    fig.suptitle("Predictive performance and computation", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig08_performance_compute")


def supplementary_descriptor_figure() -> None:
    joint = rows("descriptor_joint_sample.csv")
    tpsa = np.asarray([float(row["tpsa"]) for row in joint])
    logp = np.asarray([float(row["logp"]) for row in joint])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    axes[0].hist(tpsa, bins=70, color="#4477AA", alpha=0.85)
    for edge in [20, 40, 60, 80, 100, 120, 140]:
        axes[0].axvline(edge, color="#444444", linewidth=0.5)
    axes[0].set_xlabel(r"TPSA ($\mathrm{\AA^2}$)")
    axes[0].set_ylabel("Sampled molecules")
    axes[0].set_title("TPSA fixed boundaries")
    axes[1].hist(logp, bins=70, color="#CC6677", alpha=0.85)
    for edge in [-1, 0, 1, 2, 3, 4, 5]:
        axes[1].axvline(edge, color="#444444", linewidth=0.5)
    axes[1].set_xlabel("MolLogP")
    axes[1].set_title("MolLogP fixed boundaries")
    density = axes[2].hexbin(logp, tpsa, gridsize=45, mincnt=1, bins="log", cmap="cividis")
    axes[2].set_xlabel("MolLogP")
    axes[2].set_ylabel(r"TPSA ($\mathrm{\AA^2}$)")
    axes[2].set_title("Joint descriptor distribution")
    fig.colorbar(density, ax=axes[2], fraction=0.046, pad=0.03)
    fig.tight_layout()
    save(fig, "figS01_descriptor_joint_distribution")


def supplementary_layer_heatmap() -> None:
    data = rows("dataset_5fold_summary.csv")
    models = [
        ("Last", "p2_mofe_ffn_logp_last"),
        ("Odd", "p2_mofe_ffn_logp_odd"),
        ("Even", "p2_mofe_ffn_logp_even"),
    ]
    lookup = {(row["model"], row["dataset"]): row for row in data}
    absolute = np.asarray(
        [
            [maybe_float(lookup[(model, dataset)]["auc_mean"]) for dataset in DATASETS]
            for _, model in models
        ]
    )
    delta = absolute - absolute[0:1, :]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.8))
    heatmap(
        axes[0],
        absolute,
        [DATASET_LABELS[item] for item in DATASETS],
        [label for label, _ in models],
        "Absolute ROC-AUC",
        fmt=".3f",
    )
    heatmap(
        axes[1],
        delta * 100,
        [DATASET_LABELS[item] for item in DATASETS],
        [label for label, _ in models],
        "Delta versus last (pp)",
        center=0,
        fmt="+.2f",
    )
    fig.tight_layout()
    save(fig, "figS02_layer_placement_heatmap")


def supplementary_kan_heatmaps() -> None:
    detail = rows("kan_screen_detail.csv")
    specifications = [
        ("tpsa", "mean", "figS03_tpsa_kan_absolute", "TPSA single-branch KAN AUC", None),
        ("tpsa", "delta_dataset", "figS04_tpsa_kan_delta", "TPSA delta versus all-MLP", 0),
        ("logp", "mean", "figS05_logp_kan_absolute", "MolLogP single-branch KAN AUC", None),
        ("logp", "delta_dataset", "figS06_logp_kan_delta", "MolLogP delta versus all-MLP", 0),
    ]
    for partition, field, stem, title, center in specifications:
        matrix = np.full((8, len(DATASETS)), np.nan)
        for row in detail:
            if row["partition"] != partition:
                continue
            branch = int(row["expert"].replace("k", ""))
            dataset_index = DATASETS.index(row["dataset"])
            value = float(row[field])
            matrix[branch, dataset_index] = 100 * value if center == 0 else value
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        heatmap(
            ax,
            matrix,
            [DATASET_LABELS[item] for item in DATASETS],
            [f"k{i}" for i in range(8)],
            title + (" (percentage points)" if center == 0 else ""),
            center=center,
            fmt="+.2f" if center == 0 else ".3f",
        )
        ax.set_ylabel("Single KAN branch")
        fig.tight_layout()
        save(fig, stem)


def supplementary_fold_distribution(partition: str, stem: str) -> None:
    data = rows("final_fold_wide.csv")
    group = "TPSA" if partition == "tpsa" else "MolLogP"
    models = MODEL_SETS[group]
    lookup = {(row["model"], row["dataset"]): row for row in data}
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.7), sharey=True)
    for ax, dataset in zip(axes.ravel(), DATASETS):
        for role_index, role in enumerate(ROLE_ORDER):
            row = lookup[(models[role], dataset)]
            values = [
                maybe_float(row[f"fold_{fold}"])
                for fold in range(1, 6)
            ]
            values = [value for value in values if np.isfinite(value)]
            if not values:
                continue
            x = role_index + np.linspace(-0.08, 0.08, len(values))
            ax.scatter(x, values, s=12, color=MODEL_COLORS[role], alpha=0.8)
            ax.plot(
                [role_index - 0.18, role_index + 0.18],
                [np.mean(values), np.mean(values)],
                color=MODEL_COLORS[role],
                linewidth=1.6,
            )
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xticks(range(5), ROLE_ORDER, rotation=45, ha="right", fontsize=6)
    axes[0, 0].set_ylabel("Fold ROC-AUC")
    axes[1, 0].set_ylabel("Fold ROC-AUC")
    fig.suptitle(f"{group} legacy eight-layer fold-level results", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, stem)


def supplementary_control_figure() -> None:
    data = rows("dataset_5fold_summary.csv")
    controls = [
        ("N4 Base", "p3_base_vanilla_n4aligned"),
        ("Routed top1", "pt_smoe_top1"),
        ("Routed top2", "pt_smoe_top2"),
        ("Routed top3", "pt_smoe_top3"),
        ("Routed top4", "pt_smoe_top4"),
        ("TPSA all-KAN", "p3_tpsa_mofe_last_all_kan_n8aligned"),
        ("MolLogP all-KAN", "p3_logp_mofe_last_all_kan_n8aligned"),
    ]
    lookup = {(row["model"], row["dataset"]): row for row in data}
    matrix = np.asarray(
        [
            [maybe_float(lookup[(model, dataset)]["auc_mean"]) for dataset in DATASETS]
            for _, model in controls
        ]
    )
    fig, ax = plt.subplots(figsize=(7.3, 4.0))
    heatmap(
        ax,
        matrix,
        [DATASET_LABELS[item] for item in DATASETS],
        [label for label, _ in controls],
        "Auxiliary controls under their recorded protocols",
        fmt=".3f",
    )
    fig.tight_layout()
    save(fig, "figS09_auxiliary_controls")


def pretraining_figure(partition: str, stem: str) -> None:
    data = rows("pretraining_curves.csv")
    timing = rows("pretraining_epoch_timing_audit.csv")
    prefix = "TPSA" if partition == "tpsa" else "MolLogP"
    models = {
        "Base": "p3_base_vanilla",
        f"{prefix} I1": f"p3_{partition}_mofe_last_ffn",
        f"{prefix} I1+I2": f"p5_{partition}_mofe_last_selected_kan",
        f"{prefix} I1+I3": f"p3_{partition}_mofe_last_ffn_conloss",
        f"{prefix} Full": f"p5_{partition}_mofe_last_selected_kan_conloss",
    }
    colors = ["#333333", "#4477AA", "#EE7733", "#228833", "#AA3377"]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.2))
    for (label, model), color in zip(models.items(), colors):
        subset = sorted(
            [row for row in data if row["model"] == model],
            key=lambda row: int(row["epoch"]),
        )
        if subset:
            epoch = np.asarray([int(row["epoch"]) for row in subset])
            loss = np.asarray([float(row["train_total_loss"]) for row in subset])
            axes[0].plot(epoch, loss, label=label, color=color, linewidth=0.95)
        timing_subset = sorted(
            [
                row
                for row in timing
                if row["model"] == model and int(row["included"])
            ],
            key=lambda row: int(row["epoch"]),
        )
        if timing_subset:
            timing_epoch = np.asarray([int(row["epoch"]) for row in timing_subset])
            active_minutes = np.asarray(
                [float(row["seconds_per_epoch_active"]) / 60 for row in timing_subset]
            )
            rolling, q1, q3 = rolling_epoch_quantiles(timing_epoch, active_minutes)
            axes[1].fill_between(timing_epoch, q1, q3, color=color, alpha=0.08, linewidth=0)
            axes[1].plot(timing_epoch, rolling, label=label, color=color, linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Logged terminal-batch composite loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Active minutes per epoch")
    axes[0].set_title(f"{prefix} optimization trace")
    axes[1].set_title("11-epoch rolling median and IQR")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, stem)


def supplementary_runtime_figure() -> None:
    data = [
        row
        for row in rows("finetune_runtime_jobs.csv")
        if row["state"] == "COMPLETED" and row["protocol"] == "final_n8"
    ]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in data:
        grouped[row["dataset"]].append(float(row["elapsed_seconds"]) / 3600)
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    values = [grouped[dataset] for dataset in DATASETS]
    ax.boxplot(
        values,
        tick_labels=[DATASET_LABELS[item] for item in DATASETS],
        showfliers=True,
    )
    for index, dataset in enumerate(DATASETS, start=1):
        y = np.asarray(grouped[dataset])
        x = index + np.linspace(-0.08, 0.08, len(y))
        ax.scatter(x, y, s=8, color="#4477AA", alpha=0.55)
    ax.set_ylabel("Completed Slurm job time (h)")
    ax.set_title("Observed legacy eight-layer finetuning runtime by dataset")
    ax.tick_params(axis="x", rotation=28)
    fig.tight_layout()
    save(fig, "figS12_finetuning_runtime")


def supplementary_embedding_metric_figure() -> None:
    data = rows("embedding_metric_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    for row_index, partition in enumerate(("tpsa", "logp")):
        for col_index, (metric, label) in enumerate(
            (
                ("silhouette_mean", "Silhouette"),
                ("inter_intra_ratio_mean", "Inter/intra ratio"),
            )
        ):
            matrix = np.full((5, len(DATASETS)), np.nan)
            for role_index, role in enumerate(ROLE_ORDER):
                for dataset_index, dataset in enumerate(DATASETS):
                    selected = [
                        row
                        for row in data
                        if row["partition"] == partition
                        and row["model_role"] == role
                        and row["dataset"] == dataset
                    ][0]
                    matrix[role_index, dataset_index] = maybe_float(selected[metric])
            heatmap(
                axes[row_index, col_index],
                matrix,
                [DATASET_LABELS[item] for item in DATASETS],
                ROLE_ORDER,
                f"{'TPSA' if partition == 'tpsa' else 'MolLogP'} {label}",
                fmt=".2f",
            )
    fig.tight_layout()
    save(fig, "figS13_embedding_metrics_all_datasets")


def supplementary_tsne_figures() -> None:
    data = rows("embedding_tsne_coordinates.csv")
    number = 14
    for dataset in DATASETS:
        fig, axes = plt.subplots(2, 5, figsize=(7.8, 3.5))
        for row_index, partition in enumerate(("tpsa", "logp")):
            for col_index, role in enumerate(ROLE_ORDER):
                subset = [
                    row
                    for row in data
                    if row["dataset"] == dataset
                    and row["partition"] == partition
                    and row["model_role"] == role
                ]
                label = ("TPSA " if partition == "tpsa" else "MolLogP ") + role
                point_size = 3.0 if dataset in {"hiv", "muv"} else 4.0
                tsne_panel(axes[row_index, col_index], subset, label, point_size)
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=3.5,
                color=REGION_COLORS[index],
                label=f"R{index}",
            )
            for index in range(8)
        ]
        sample_rows = [
            row
            for row in data
            if row["dataset"] == dataset and row["partition"] == "tpsa"
        ]
        common_count = len({row["smiles"] for row in sample_rows})
        fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False)
        fig.suptitle(
            f"{DATASET_LABELS[dataset]} fold-1 embeddings, n={common_count}",
            fontsize=9.5,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        save(fig, f"figS{number:02d}_tsne_{dataset}")
        number += 1


def main() -> None:
    set_style()
    workflow_assets = [
        FIGURE_ROOT / "assets" / "image2_chemical_space.png",
        FIGURE_ROOT / "assets" / "image2_conditional_branches.png",
    ]
    if all(path.is_file() for path in workflow_assets):
        workflow_figure()
    else:
        print("Skipping optional workflow figure because visual assets are absent")
    descriptor_figure()
    layer_figure()
    kan_figure()
    final_performance_figure()
    conditional_ablation_figure()
    tsne_main_figure()
    compute_pareto_figure()

    supplementary_descriptor_figure()
    supplementary_layer_heatmap()
    supplementary_kan_heatmaps()
    supplementary_fold_distribution("tpsa", "figS07_tpsa_fold_distributions")
    supplementary_fold_distribution("logp", "figS08_logp_fold_distributions")
    supplementary_control_figure()
    pretraining_figure("tpsa", "figS10_pretraining_dynamics_tpsa")
    pretraining_figure("logp", "figS11_pretraining_dynamics_logp")
    supplementary_runtime_figure()
    supplementary_embedding_metric_figure()
    supplementary_tsne_figures()
    print(f"Wrote main and Supporting Information figures to {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
