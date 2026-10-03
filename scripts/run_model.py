#!/usr/bin/env python3
"""Print one paper-model command, or execute it inside a GPU Slurm job."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'bace', 'hiv', 'muv']


def model_matrix():
    with (ROOT / 'configs/final_model_matrix.tsv').open() as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def control_args(spec, phase, control_dir, dataset=None):
    pretrain = phase == 'pretrain'
    values = ['--moe_layer_mode', spec['moe_layer_mode']]
    partition, use = spec['partition'], spec['descriptor_use']
    scheme = spec['assignment_scheme']
    metadata = control_dir / 'descriptor_control_metadata.json'
    arrays = control_dir / 'descriptor_controls.npz'
    if partition != 'shared':
        values += ['--expert_property', partition]
    if use != 'none':
        values += ['--descriptor_use', use, '--descriptor_aux_coeff', '0.1']
        values += (['--descriptor_values_path', str(arrays)] if pretrain else
                   ['--descriptor_stats', str(metadata)])
    elif scheme == 'learned':
        values += ['--use_standard_moe', '--num_experts', '8', '--moe_top_k', '1',
                   '--moe_aux_loss_coeff', '0.01', '--router_granularity', 'molecule',
                   '--assignment_scheme', 'learned']
    elif spec['control'] != 'dense_base':
        values += ['--use_prior_moe', '--num_experts', '8', '--expert_type', 'ffn',
                   '--assignment_scheme', scheme, '--bucket_edges_file', str(metadata)]
        if pretrain:
            key = f'{partition}_{scheme}'
            values += ['--expert_ids_path', str(arrays), '--expert_assignment_key', key]
        elif scheme == 'stable_random':
            values += ['--assignment_manifest', str(
                control_dir / 'downstream_random_assignments' / f'{dataset}_{partition}.csv')]
    if pretrain and float(spec['conloss_coeff']):
        values += ['--use_contrastive_expert_loss', '--contrastive_coff', '0.1',
                   '--contrastive_temperature', '0.1']
    return values


def build_command(args, spec):
    run = args.run_root.resolve()
    controls = (args.control_dir or run / 'control_data').resolve()
    weights = (args.pretrain_root or run / 'pretrain').resolve()
    if args.phase == 'pretrain':
        if args.zinc_cache is None:
            raise ValueError('--zinc-cache is required for pretraining')
        cwd = ROOT / 'MoleSG/pretrain'
        command = [sys.executable, str(cwd / 'train_total.py'), '--gpu', '0',
                   '--seed', '42', '--epochs', '300', '--periodic_checkpoint_every', '10',
                   '--overwrite_best_checkpoint', '--experiment_name', spec['model'],
                   '--data_root', str(args.zinc_cache.resolve()),
                   '--model_output_root', str(weights)]
    else:
        if args.downstream_data is None or args.dataset is None:
            raise ValueError('--downstream-data and --dataset are required')
        cwd = ROOT / 'MoleSG/Downstream'
        command = [sys.executable, str(cwd / 'train_graph_evalfix.py'),
                   '--mode', args.phase, '--gpu', '0', '--seed', '42', '--fold', '5',
                   '--dataset', args.dataset, '--experiment_name', spec['result_exp'],
                   '--data_root', str(args.downstream_data.resolve()), '--num_workers', '0',
                   '--result_output_root', str(run / 'results'), '--save_embeddings',
                   '--backbone_profile', 'pretrain_aligned', '--encoder_layers_override', '8',
                   '--strict_backbone_load']
        if args.dataset == 'hiv':
            command += ['--batch_size_override', '12']
        if args.phase == 'finetune':
            command += ['--pretrained_ckpt_path', str(
                weights / spec['model'] / 'compt/periodic_latest.pth'),
                '--model_output_root', str(run / 'finetuned')]
        else:
            command += ['--source_experiment_name', spec['result_exp'],
                        '--finetuned_checkpoint_root', str(run / 'finetuned'),
                        '--model_definition_lock', str(run / 'model_definition_frozen.json')]
    command += control_args(spec, args.phase, controls, args.dataset)
    return command, cwd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['pretrain', 'finetune', 'final_test'])
    parser.add_argument('--model', required=True, choices=[s['model'] for s in model_matrix()])
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs')
    parser.add_argument('--pretrain-root', type=Path)
    parser.add_argument('--control-dir', type=Path)
    parser.add_argument('--zinc-cache', type=Path)
    parser.add_argument('--downstream-data', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    spec = next(row for row in model_matrix() if row['model'] == args.model)
    command, cwd = build_command(args, spec)
    print(f'Working directory: {cwd}')
    print(shlex.join(command))
    if not args.execute:
        return 0
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('--execute must run inside an allocated Slurm GPU job')
    for flag in ('--data_root', '--expert_ids_path', '--descriptor_values_path',
                 '--bucket_edges_file', '--descriptor_stats', '--assignment_manifest',
                 '--pretrained_ckpt_path'):
        if flag in command:
            path = Path(command[command.index(flag) + 1])
            if not path.exists():
                raise FileNotFoundError(f'{flag}: {path}')
    if args.phase == 'finetune':
        import torch
        path = command[command.index('--pretrained_ckpt_path') + 1]
        checkpoint = torch.load(path, map_location='cpu')
        if checkpoint.get('epoch', checkpoint.get('best_epoch', -1)) != 299:
            raise ValueError('Finetuning requires the completed 300-epoch periodic checkpoint')
        del checkpoint
    if args.phase == 'final_test':
        existing = list((args.run_root / 'results' / args.dataset).glob(
            f'test_result_{spec["result_exp"]}_{args.dataset}_fold_*.*'))
        if existing:
            raise SystemExit('Test output already exists. Do not overwrite a completed test.')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'MoleSG') + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run(command, cwd=cwd, env=env, check=True)
    if args.phase == 'pretrain':
        print(f'PRETRAIN_DONE: {spec["model"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
