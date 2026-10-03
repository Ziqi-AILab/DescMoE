#!/usr/bin/env python3
"""Rebuild paper statistics from small fold tables without models or datasets."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from build_p0_corrected_panel import paired_contrasts


def check_folds(frame, key, configurations):
    if len(frame) != configurations * 8 * 5 or frame[key].nunique() != configurations:
        raise ValueError('Unexpected fold/configuration count')
    if frame.duplicated([key, 'dataset', 'fold']).any():
        raise ValueError('Duplicate model-dataset-fold')
    if not np.isfinite(frame.roc_auc).all() or not frame.roc_auc.between(0, 1).all():
        raise ValueError('Invalid ROC-AUC')
    if not (frame.seed == 41 + frame.fold).all():
        raise ValueError('Seed/fold mismatch')
    for _, group in frame.groupby([key, 'dataset']):
        if set(group.fold) != {1, 2, 3, 4, 5}:
            raise ValueError('Incomplete five-seed group')
    if not frame.evaluation_coverage_ok.eq(True).all():
        raise ValueError('Coverage failure')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'results/corrected')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/rebuilt_tables')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    neural = pd.read_csv(args.input_dir / 'corrected_fold_auc.csv', float_precision='round_trip')
    cpu = pd.read_csv(args.input_dir / 'cpu_baseline_fold_auc.csv', float_precision='round_trip')
    check_folds(neural, 'model', 16)
    check_folds(cpu, 'family', 5)
    if not neural.completion.eq('complete').all():
        raise ValueError('Incomplete neural panel')
    groups = ['model', 'result_exp', 'partition', 'control', 'source_group']
    dataset = neural.groupby(groups + ['dataset']).agg(
        mean_roc_auc=('roc_auc', 'mean'), sd_roc_auc=('roc_auc', 'std'),
        folds=('roc_auc', 'size')).reset_index()
    model = dataset.groupby(groups).agg(macro_roc_auc=('mean_roc_auc', 'mean'),
        datasets=('dataset', 'size'), folds=('folds', 'sum')).reset_index()
    cpu_dataset = cpu.groupby(['family', 'dataset']).agg(
        mean_roc_auc=('roc_auc', 'mean'), sd_roc_auc=('roc_auc', 'std'),
        folds=('roc_auc', 'size')).reset_index()
    cpu_model = cpu_dataset.groupby('family').agg(macro_roc_auc=('mean_roc_auc', 'mean'),
        datasets=('dataset', 'size'), folds=('folds', 'sum')).reset_index()
    tables = {'corrected_dataset_summary.csv': dataset, 'corrected_model_summary.csv': model,
              'corrected_paired_contrasts.csv': paired_contrasts(neural),
              'cpu_baseline_dataset_summary.csv': cpu_dataset,
              'cpu_baseline_model_summary.csv': cpu_model}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if args.check:
            expected = pd.read_csv(args.input_dir / name, float_precision='round_trip')
            pd.testing.assert_frame_equal(table, expected, check_dtype=False,
                                          atol=1e-12, rtol=1e-12)
        table.to_csv(args.output_dir / name, index=False)
    print('OK: 640 neural folds, 200 CPU folds, 16 paired macro contrasts.')
    print(f'Tables written to {args.output_dir}')


if __name__ == '__main__':
    main()
