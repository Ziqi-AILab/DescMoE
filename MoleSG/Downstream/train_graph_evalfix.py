#!/usr/bin/env python3
"""Full-coverage scaffold finetuning and checkpoint evaluation.

This entrypoint isolates the corrected JCIM evaluation protocol from legacy
outputs. Validation and test loaders are deterministic and never drop samples.
Test evaluation is guarded by a frozen model-definition file.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_graph import (
    construct_dataset,
    mol_collate_func,
    mol_collate_with_ids_func,
)
from train_graph import (
    _apply_backbone_profile,
    _freeze_inactive,
    _load_pretrained_checkpoint,
    _standard_moe_aux_loss,
    _unfreeze_all,
)
from transformer_graph_finetune import make_model
from utils import ScheduledOptim, cal_loss, evaluate, get_loss, get_options, scaffold_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pretrain"))
from prior_moe import assign_expert_ids  # noqa: E402


CLASSIFICATION_DATASETS = {
    "bbbp", "tox21", "toxcast", "sider", "clintox", "bace", "hiv", "muv"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["finetune", "validation_audit", "final_test"],
                        default="validation_audit")
    parser.add_argument("--dataset", required=True, choices=sorted(CLASSIFICATION_DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--fold_indices", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size_override", type=int, default=0)

    parser.add_argument("--experiment_name", required=True,
                        help="New output namespace. It must end in _evalfix.")
    parser.add_argument("--source_experiment_name", default=None,
                        help="Experiment name used by an existing finetuned checkpoint.")
    parser.add_argument("--pretrained_ckpt_path", default=None)
    parser.add_argument("--finetuned_checkpoint_root", default="./Model")
    parser.add_argument("--model_output_root", default="./Model_evalfix")
    parser.add_argument("--result_output_root", default="./Result_evalfix")
    parser.add_argument("--model_definition_lock", default=None)
    parser.add_argument("--require_model_definition_lock", action="store_true")
    parser.add_argument("--save_embeddings", action="store_true")

    parser.add_argument("--use_prior_moe", action="store_true")
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--expert_property", default="tpsa",
                        choices=["tpsa", "logp", "mw", "n_heavy", "n_rings"])
    parser.add_argument("--expert_type", default="ffn", choices=["ffn", "kan", "mixed"])
    parser.add_argument("--kan_expert_indices", default=None)
    parser.add_argument("--kan_grid_size", type=int, default=5)
    parser.add_argument("--kan_spline_order", type=int, default=3)
    parser.add_argument("--moe_layer_mode", default="none",
                        choices=["all", "last", "odd", "even", "none"])
    parser.add_argument("--moe_layer_indices", default=None)
    parser.add_argument("--moe_d_ff", type=int, default=0)
    parser.add_argument("--use_standard_moe", action="store_true")
    parser.add_argument("--moe_top_k", type=int, default=2)
    parser.add_argument("--moe_aux_loss_coeff", type=float, default=0.01)
    parser.add_argument("--freeze_inactive_experts", action="store_true")
    parser.add_argument("--descriptor_use", choices=["none", "concat", "auxiliary"],
                        default="none")
    parser.add_argument("--descriptor_stats", default=None)
    parser.add_argument("--descriptor_aux_coeff", type=float, default=0.1)
    parser.add_argument(
        "--assignment_scheme",
        choices=["fixed", "quantile", "occupancy_random", "merge_tail",
                 "min_occupancy", "learned"], default="fixed")
    parser.add_argument("--assignment_manifest", default=None)
    parser.add_argument("--bucket_edges_file", default=None)

    parser.add_argument("--backbone_profile", default="dataset_default",
                        choices=["dataset_default", "pretrain_aligned"])
    parser.add_argument("--encoder_layers_override", type=int, default=0)
    parser.add_argument("--strict_backbone_load", action="store_true")
    return parser.parse_args()


def fold_indices(args: argparse.Namespace) -> list[int]:
    if not args.fold_indices:
        return list(range(args.fold))
    values = [int(value.strip()) - 1 for value in args.fold_indices.split(",") if value.strip()]
    if any(value < 0 or value >= args.fold for value in values):
        raise ValueError(f"fold_indices must be between 1 and {args.fold}: {args.fold_indices}")
    return values


def set_seed(seed: int, device: torch.device) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def parse_kan_indices(value: str | None) -> list[int] | None:
    if not value:
        return None
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(indices) != len(set(indices)) or any(index < 0 or index > 7 for index in indices):
        raise ValueError(f"Invalid KAN indices: {value}")
    return indices


def model_kwargs(args: argparse.Namespace) -> dict:
    return {
        "use_prior_moe": args.use_prior_moe,
        "num_experts": args.num_experts,
        "expert_type": args.expert_type,
        "kan_expert_indices": parse_kan_indices(args.kan_expert_indices),
        "kan_grid_size": args.kan_grid_size,
        "kan_spline_order": args.kan_spline_order,
        "moe_layer_mode": args.moe_layer_mode,
        "moe_layer_indices": args.moe_layer_indices,
        "moe_d_ff": args.moe_d_ff,
        "use_standard_moe": args.use_standard_moe,
        "moe_top_k": args.moe_top_k,
        "moe_aux_loss_coeff": args.moe_aux_loss_coeff,
        "descriptor_use": args.descriptor_use,
    }


def load_control_inputs(args: argparse.Namespace) -> None:
    args._bucket_edges = None
    args._assignment_map = None
    args._descriptor_mean = None
    args._descriptor_std = None
    if args.bucket_edges_file:
        payload = json.loads(Path(args.bucket_edges_file).read_text())
        edge_keys = {
            "fixed": "fixed_edges",
            "quantile": "quantile_edges",
            "merge_tail": "merge_tail_edges",
            "min_occupancy": "min_occupancy_edges",
        }
        if args.assignment_scheme in edge_keys:
            args._bucket_edges = payload["partitions"][args.expert_property][
                edge_keys[args.assignment_scheme]]
    if args.assignment_scheme == "occupancy_random":
        if not args.assignment_manifest:
            raise ValueError("occupancy_random requires --assignment_manifest")
        frame = pd.read_csv(args.assignment_manifest)
        if frame.sample_id.duplicated().any():
            raise ValueError(f"Duplicate sample_id in {args.assignment_manifest}")
        args._assignment_map = dict(zip(
            frame.sample_id.astype(str), frame.random_occ_expert_id.astype(int)))
    if args.descriptor_use != "none":
        if not args.descriptor_stats:
            raise ValueError(f"{args.descriptor_use} requires --descriptor_stats")
        payload = json.loads(Path(args.descriptor_stats).read_text())
        stats = payload["partitions"][args.expert_property]
        args._descriptor_mean = float(stats["mean"])
        args._descriptor_std = float(stats["std"])
        if args._descriptor_std <= 0:
            raise ValueError("Descriptor standard deviation must be positive")


def active_expert_ids(args: argparse.Namespace, smiles: list[str], device: torch.device,
                      sample_ids: list[str] | None = None):
    if not args.use_prior_moe:
        return None
    if args.assignment_scheme == "occupancy_random":
        if sample_ids is None:
            raise ValueError("Random assignment requires sample IDs")
        missing = [sample_id for sample_id in sample_ids
                   if sample_id not in args._assignment_map]
        if missing:
            raise ValueError(f"Assignment manifest lacks sample IDs: {missing[:5]}")
        values = [args._assignment_map[sample_id] for sample_id in sample_ids]
    else:
        values = assign_expert_ids(
            smiles, property_name=args.expert_property,
            num_experts=args.num_experts, bucket_edges=args._bucket_edges)
    return torch.tensor(values, dtype=torch.long, device=device)


def standardized_descriptor_values(args: argparse.Namespace, smiles: list[str],
                                   device: torch.device):
    if args.descriptor_use == "none":
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors
    except ImportError as exc:
        raise ImportError("RDKit is required for descriptor controls") from exc
    values = []
    for smile in smiles:
        molecule = Chem.MolFromSmiles(smile)
        if molecule is None:
            raise ValueError(f"Invalid SMILES: {smile}")
        value = (Descriptors.TPSA(molecule) if args.expert_property == "tpsa"
                 else Crippen.MolLogP(molecule))
        values.append((float(value) - args._descriptor_mean) / args._descriptor_std)
    return torch.tensor(values, dtype=torch.float32, device=device)


def valid_endpoint_count(labels: np.ndarray) -> int:
    count = 0
    for endpoint in range(labels.shape[1]):
        values = labels[:, endpoint]
        values = values[values >= 0]
        if len(np.unique(values)) >= 2:
            count += 1
    return count


def expected_sample_ids(dataset) -> list[str]:
    return [molecule.sample_id for molecule in dataset]


def verify_coverage(expected: list[str], observed: list[str], split_name: str) -> None:
    if len(observed) != len(set(observed)):
        raise RuntimeError(f"{split_name} contains duplicate sample_id values")
    if set(expected) != set(observed) or len(expected) != len(observed):
        missing = sorted(set(expected) - set(observed))[:10]
        unexpected = sorted(set(observed) - set(expected))[:10]
        raise RuntimeError(
            f"{split_name} coverage mismatch expected={len(expected)} observed={len(observed)} "
            f"missing={missing} unexpected={unexpected}")
    print(f"EVALUATION_COVERAGE_OK: split={split_name} samples={len(observed)}")


def evaluate_model(model, dataset, args, train_params, split_name: str) -> tuple[dict, np.ndarray]:
    loader = DataLoader(
        dataset=dataset,
        batch_size=train_params["batch_size"],
        collate_fn=mol_collate_with_ids_func,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model.eval()
    sample_ids: list[str] = []
    source_indices: list[int] = []
    smiles: list[str] = []
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            (batch_ids, batch_indices, batch_smiles, adjacency_matrix,
             node_features, edge_features, y_true) = batch
            adjacency_matrix = adjacency_matrix.to(train_params["device"])
            node_features = node_features.to(train_params["device"])
            edge_features = edge_features.to(train_params["device"])
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            expert_ids = active_expert_ids(
                args, batch_smiles, train_params["device"], batch_ids)
            descriptor_values = standardized_descriptor_values(
                args, batch_smiles, train_params["device"])
            y_pred, y_embedding = model(
                node_features, batch_mask, adjacency_matrix, edge_features,
                expert_ids=expert_ids, descriptor_values=descriptor_values)
            sample_ids.extend(batch_ids)
            source_indices.extend(int(value) for value in batch_indices)
            smiles.extend(batch_smiles)
            labels.append(y_true.numpy())
            predictions.append(y_pred.detach().cpu().numpy())
            embeddings.append(y_embedding.detach().cpu().numpy())

    label_array = np.concatenate(labels, axis=0)
    prediction_array = np.concatenate(predictions, axis=0)
    embedding_array = np.concatenate(embeddings, axis=0)
    verify_coverage(expected_sample_ids(dataset), sample_ids, split_name)
    result = evaluate(
        label_array, prediction_array, np.asarray(smiles),
        requirement=["sample", train_params["loss_function"], train_params["metric"]],
        data_mean=0, data_std=1, data_task=train_params["task"])
    result["sample_id"] = sample_ids
    result["source_row_index"] = source_indices
    result["canonical_smiles"] = smiles
    result["valid_endpoints"] = valid_endpoint_count(label_array)
    return result, embedding_array


def prediction_frame(result: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": result["sample_id"],
        "source_row_index": result["source_row_index"],
        "canonical_smiles": result["canonical_smiles"],
        "actual": [json.dumps(value) for value in result["label"]],
        "prediction": [json.dumps(value) for value in result["prediction"]],
    })


def write_evaluation(result: dict, embeddings: np.ndarray, dataset, args,
                     split_name: str, fold_number: int, checkpoint_path: Path,
                     legacy_best: bool, checkpoint_metadata: dict | None = None) -> None:
    result_dir = Path(args.result_output_root) / args.dataset
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{split_name}_result_{args.experiment_name}_{args.dataset}_fold_{fold_number}"
    prediction_path = result_dir / f"{stem}.csv"
    sample_manifest_path = result_dir / (
        f"{split_name}_sample_manifest_{args.experiment_name}_"
        f"{args.dataset}_fold_{fold_number}.csv")
    frame = prediction_frame(result)
    frame.to_csv(prediction_path, index=False)
    frame[["sample_id", "source_row_index", "canonical_smiles"]].to_csv(
        sample_manifest_path, index=False)
    metadata = {
        "protocol": "evalfix_full_coverage_v1",
        "mode": args.mode,
        "split": split_name,
        "dataset": args.dataset,
        "fold": fold_number,
        "seed": args.seed + fold_number - 1,
        "experiment_name": args.experiment_name,
        "source_experiment_name": args.source_experiment_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_load_status": "FINETUNED_CHECKPOINT_LOAD_OK",
        "legacy_best_checkpoint": legacy_best,
        "prediction_file": str(prediction_path),
        "sample_id_manifest": str(sample_manifest_path),
        "split_samples": len(dataset),
        "evaluated_samples": len(result["sample_id"]),
        "evaluation_coverage_ok": len(dataset) == len(result["sample_id"]),
        "valid_endpoints": result["valid_endpoints"],
        "descriptor_use": args.descriptor_use,
        "descriptor_property": args.expert_property,
        "assignment_scheme": args.assignment_scheme,
        "assignment_manifest": args.assignment_manifest,
        "bucket_edges_file": args.bucket_edges_file,
        "metric": train_params_metric(result),
        "validation_best_epoch": (
            checkpoint_metadata.get("best_epoch") if checkpoint_metadata else None),
        "validation_best_auc": (
            checkpoint_metadata.get("best_valid_auc") if checkpoint_metadata else None),
    }
    (result_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.save_embeddings:
        payload = {
            sample_id: {
                "source_row_index": source_index,
                "canonical_smiles": smiles,
                "embedding": embedding,
            }
            for sample_id, source_index, smiles, embedding in zip(
                result["sample_id"], result["source_row_index"],
                result["canonical_smiles"], embeddings)
        }
        with (result_dir / f"{split_name}_embedding_{args.experiment_name}_{args.dataset}_fold_{fold_number}.pickle").open("wb") as handle:
            pickle.dump(payload, handle)


def train_params_metric(result: dict) -> dict:
    metrics = {}
    for key in ("auc", "rmse", "mae"):
        if key in result:
            metrics[key] = float(result[key])
    return metrics


def load_finetuned_model(checkpoint_path: Path, model_params: dict, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = make_model(**model_params).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    print(
        f"FINETUNED_CHECKPOINT_LOAD_OK: path={checkpoint_path} "
        f"best_epoch={checkpoint.get('best_epoch', 'unknown')}")
    return model, checkpoint


def require_frozen_lock(path: str | None, args: argparse.Namespace) -> dict:
    if not path:
        raise RuntimeError("Final test requires --model_definition_lock")
    lock_path = Path(path)
    payload = json.loads(lock_path.read_text())
    if payload.get("status") != "frozen":
        raise RuntimeError(f"Model definition is not frozen: {lock_path}")
    allowed = payload.get("allowed_source_experiments", [])
    source = args.source_experiment_name or args.experiment_name
    if source not in allowed:
        raise RuntimeError(f"Source experiment {source} is absent from {lock_path}")
    print(f"MODEL_DEFINITION_LOCK_OK: {lock_path}")
    return payload


def train_one_fold(model, train_dataset, valid_dataset, args, model_params,
                   train_params, fold_number: int) -> tuple[Path, dict, np.ndarray]:
    train_loader = DataLoader(
        train_dataset, batch_size=train_params["batch_size"],
        collate_fn=mol_collate_with_ids_func, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True)
    criterion = get_loss(train_params["loss_function"])
    optimizer = ScheduledOptim(
        torch.optim.Adam(model.parameters(), lr=0),
        train_params["warmup_factor"], model_params["d_model"],
        train_params["total_warmup_steps"])
    best_metric = float("-inf")
    best_epoch = -1
    best_result = None
    best_embeddings = None
    history = []
    model_dir = Path(args.model_output_root) / args.dataset
    result_dir = Path(args.result_output_root) / args.dataset
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / (
        f"best_model_{args.experiment_name}_{args.dataset}_fold_{fold_number}.pt")
    history_path = result_dir / (
        f"training_history_{args.experiment_name}_{args.dataset}_fold_{fold_number}.csv")

    for epoch_index in range(train_params["total_epochs"]):
        started = time.time()
        model.train()
        losses = []
        for batch in train_loader:
            (batch_ids, _, smiles, adjacency_matrix,
             node_features, edge_features, y_true) = batch
            adjacency_matrix = adjacency_matrix.to(train_params["device"])
            node_features = node_features.to(train_params["device"])
            edge_features = edge_features.to(train_params["device"])
            y_true = y_true.to(train_params["device"])
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            expert_ids = active_expert_ids(
                args, smiles, train_params["device"], batch_ids)
            descriptor_values = standardized_descriptor_values(
                args, smiles, train_params["device"])
            if args.use_prior_moe and args.freeze_inactive_experts:
                _freeze_inactive(model, expert_ids)
            y_pred, _ = model(
                node_features, batch_mask, adjacency_matrix, edge_features,
                expert_ids=expert_ids, descriptor_values=descriptor_values)
            loss = cal_loss(
                y_true, y_pred, train_params["loss_function"], criterion,
                0, 1, train_params["device"])
            if args.use_standard_moe:
                loss = loss + _standard_moe_aux_loss(model, train_params["device"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
            if args.use_prior_moe and args.freeze_inactive_experts:
                _unfreeze_all(model)
            losses.append(float(loss.detach().item()))

        valid_result, valid_embeddings = evaluate_model(
            model, valid_dataset, args, train_params, "validation")
        metric = float(valid_result[train_params["metric"]])
        epoch_number = epoch_index + 1
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch_number
            best_result = valid_result
            best_embeddings = valid_embeddings
            torch.save({
                "state_dict": model.state_dict(),
                "best_epoch": best_epoch,
                f"best_valid_{train_params['metric']}": best_metric,
                "evaluation_protocol": "evalfix_full_coverage_v1",
                "control_metadata": {
                    "descriptor_use": args.descriptor_use,
                    "descriptor_property": args.expert_property,
                    "descriptor_aux_coeff": args.descriptor_aux_coeff,
                    "assignment_scheme": args.assignment_scheme,
                    "assignment_manifest": args.assignment_manifest,
                    "bucket_edges_file": args.bucket_edges_file,
                },
            }, checkpoint_path)
        history.append({
            "epoch": epoch_number,
            "train_loss": float(np.mean(losses)),
            "validation_auc": metric,
            "best_epoch": best_epoch,
            "best_validation_auc": best_metric,
            "learning_rate": optimizer.view_lr(),
            "epoch_seconds": time.time() - started,
            "validation_samples": len(valid_result["sample_id"]),
        })
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(
            f"EVALFIX_EPOCH: epoch={epoch_number} train_loss={np.mean(losses):.6f} "
            f"validation_auc={metric:.6f} best_epoch={best_epoch} "
            f"best_validation_auc={best_metric:.6f}")
        if epoch_number - best_epoch >= 20:
            print("EVALFIX_EARLY_STOP")
            break

    if best_result is None or best_embeddings is None:
        raise RuntimeError("No valid checkpoint was selected")
    return checkpoint_path, best_result, best_embeddings


def main() -> int:
    args = parse_args()
    if not args.experiment_name.endswith("_evalfix"):
        raise ValueError("experiment_name must end in _evalfix")
    if args.mode != "finetune" and not args.source_experiment_name:
        raise ValueError(f"{args.mode} requires --source_experiment_name")
    if args.mode == "finetune" and not args.pretrained_ckpt_path:
        raise ValueError("finetune requires --pretrained_ckpt_path")
    if args.mode == "final_test" or args.require_model_definition_lock:
        require_frozen_lock(args.model_definition_lock, args)
    load_control_inputs(args)

    model_params, train_params = get_options(args.dataset)
    _apply_backbone_profile(
        model_params, args.backbone_profile, args.encoder_layers_override)
    if args.backbone_profile == "pretrain_aligned" and args.dataset == "hiv" and not args.batch_size_override:
        args.batch_size_override = 12
    if args.batch_size_override:
        train_params["batch_size"] = args.batch_size_override
    model_params.update(model_kwargs(args))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_params["device"] = device
    print(
        f"EVALUATION_PROTOCOL: corrected_full_coverage validation_shuffle=false "
        f"validation_drop_last=false test_shuffle=false test_drop_last=false")

    data_path = Path("Data") / args.dataset / "preprocess" / f"{args.dataset}.pickle"
    with data_path.open("rb") as handle:
        data_mol, data_label = pickle.load(handle)
    model_params["max_length"] = max(mol.GetNumAtoms() for mol in data_mol)
    dataset = construct_dataset(
        data_mol, data_label, model_params["d_atom"], model_params["d_edge"],
        model_params["max_length"], dataset_name=args.dataset)

    for fold_index in fold_indices(args):
        fold_number = fold_index + 1
        seed = args.seed + fold_index
        set_seed(seed, device)
        train_index, valid_index, test_index = scaffold_split(
            data_mol, frac=[0.8, 0.1, 0.1], balanced=True,
            include_chirality=False, ramdom_state=seed)
        train_dataset = dataset[train_index]
        valid_dataset = dataset[valid_index]
        test_dataset = dataset[test_index] if args.mode == "final_test" else None
        print(
            f"SPLIT_SIZES: dataset={args.dataset} fold={fold_number} seed={seed} "
            f"train={len(train_dataset)} validation={len(valid_dataset)} "
            f"test={len(test_index)}")

        if args.mode == "validation_audit":
            checkpoint_path = Path(args.finetuned_checkpoint_root) / args.dataset / (
                f"best_model_{args.source_experiment_name}_{args.dataset}_fold_{fold_number}.pt")
            model, checkpoint = load_finetuned_model(checkpoint_path, model_params, device)
            result, embeddings = evaluate_model(
                model, valid_dataset, args, train_params, "validation")
            write_evaluation(
                result, embeddings, valid_dataset, args, "validation", fold_number,
                checkpoint_path, legacy_best=True,
                checkpoint_metadata=checkpoint)
            continue

        if args.mode == "final_test":
            checkpoint_path = Path(args.finetuned_checkpoint_root) / args.dataset / (
                f"best_model_{args.source_experiment_name}_{args.dataset}_fold_{fold_number}.pt")
            model, checkpoint = load_finetuned_model(checkpoint_path, model_params, device)
            if checkpoint.get("evaluation_protocol") != "evalfix_full_coverage_v1":
                raise RuntimeError(f"Final test refuses non-evalfix checkpoint: {checkpoint_path}")
            result, embeddings = evaluate_model(
                model, test_dataset, args, train_params, "test")
            write_evaluation(
                result, embeddings, test_dataset, args, "test", fold_number,
                checkpoint_path, legacy_best=False,
                checkpoint_metadata=checkpoint)
            continue

        train_params["total_warmup_steps"] = (
            int(len(train_dataset) / train_params["batch_size"])
            * train_params["total_warmup_epochs"])
        model = make_model(**model_params).to(device)
        _load_pretrained_checkpoint(
            model, args.pretrained_ckpt_path, device,
            strict_backbone_load=args.strict_backbone_load)
        checkpoint_path, valid_result, valid_embeddings = train_one_fold(
            model, train_dataset, valid_dataset, args, model_params,
            train_params, fold_number)
        write_evaluation(
            valid_result, valid_embeddings, valid_dataset, args, "validation",
            fold_number, checkpoint_path, legacy_best=False,
            checkpoint_metadata=torch.load(checkpoint_path, map_location="cpu"))
        print(
            f"EVALFIX_FINETUNE_DONE: experiment={args.experiment_name} "
            f"dataset={args.dataset} fold={fold_number}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
