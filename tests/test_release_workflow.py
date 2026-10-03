#!/usr/bin/env python3
"""CPU checks for the portable commands and actual model components."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'MoleSG/pretrain'))
from run_model import build_command, model_matrix, DATASETS
from prepare_matched_control_assignments import assign, quantile_edges
from prior_moe import expert_contrastive_loss


def test_commands():
    specs = model_matrix()
    assert len(specs) == 16
    for spec in specs:
        for phase in ('pretrain', 'finetune', 'final_test'):
            args = argparse.Namespace(run_root=Path('/tmp/new_run'),
                control_dir=None, pretrain_root=None, phase=phase,
                zinc_cache=Path('/tmp/external/zinc_cache'),
                downstream_data=Path('/tmp/external/Data'), dataset='hiv')
            command, cwd = build_command(args, spec)
            assert cwd.is_dir() and Path(command[1]).is_file()
            assert '--kan_expert_indices' not in command
            if phase != 'pretrain':
                assert '--contrastive_coff' not in command
                assert command[command.index('--encoder_layers_override') + 1] == '8'
                assert command[command.index('--batch_size_override') + 1] == '12'
            if spec['assignment_scheme'] == 'learned':
                assert command[command.index('--router_granularity') + 1] == 'molecule'
            if phase == 'final_test':
                assert '--model_definition_lock' in command
    print('48 phase/configuration commands checked')


def test_boundaries_and_conloss():
    assert assign(np.array([19.9, 20, 40, 140]), [20, 40, 60, 80, 100, 120, 140]).tolist() == [0, 1, 2, 7]
    assert len(quantile_edges(np.arange(250, dtype=float))) == 7
    torch.manual_seed(42)
    features = torch.randn(5, 7, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2])
    normalized = F.normalize(features, dim=1)
    logits = normalized @ normalized.T / 0.1
    expected = []
    for anchor in range(5):
        positives = [j for j in range(5) if j != anchor and labels[j] == labels[anchor]]
        if positives:
            others = [j for j in range(5) if j != anchor]
            expected.append(torch.logsumexp(logits[anchor, others], 0)
                            - logits[anchor, positives].mean())
    actual = expert_contrastive_loss(features, labels, temperature=0.1)
    assert torch.allclose(actual, torch.stack(expected).mean(), atol=1e-5)
    assert expert_contrastive_loss(features, torch.arange(5)).item() == 0
    actual.backward()
    assert torch.isfinite(features.grad).all()
    print('Fixed boundaries, quantiles, and ConLoss formula checked')


def test_freeze_refuses_missing_validation():
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory)
        # Even complete-looking test outputs cannot release a panel without validation.
        (run / 'results').mkdir()
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/freeze_panel.py'),
            '--run-root', str(run)], capture_output=True, text=True)
        assert result.returncode != 0
        assert not (run / 'model_definition_frozen.json').exists()
    print('Missing validation does not release final test')


def test_freeze_complete_validation():
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory)
        checkpoint = run / 'checkpoint_fixture.pth'
        checkpoint.touch()
        first_manifest = None
        for spec in model_matrix():
            for dataset in DATASETS:
                folder = run / 'results' / dataset
                folder.mkdir(parents=True, exist_ok=True)
                for fold in range(1, 6):
                    stem = f'validation_result_{spec["result_exp"]}_{dataset}_fold_{fold}'
                    prediction = folder / f'{stem}.csv'
                    manifest = folder / f'{stem}_samples.csv'
                    sample_ids = [f'{dataset}:0', f'{dataset}:1']
                    with prediction.open('w', newline='') as handle:
                        writer = csv.writer(handle)
                        writer.writerow(['sample_id', 'actual', 'prediction'])
                        writer.writerows([[sample_ids[0], '[0]', '[0.1]'],
                                          [sample_ids[1], '[1]', '[0.9]']])
                    manifest.write_text('sample_id\n' + '\n'.join(sample_ids) + '\n')
                    meta = dict(mode='finetune', split='validation',
                        protocol='evalfix_full_coverage_v1', legacy_best_checkpoint=False,
                        evaluation_coverage_ok=True, dataset=dataset, fold=fold,
                        seed=41 + fold, experiment_name=spec['result_exp'],
                        sample_id_manifest=str(manifest), split_samples=2,
                        evaluated_samples=2, validation_best_auc=1.0,
                        checkpoint=str(checkpoint))
                    prediction.with_suffix('.json').write_text(json.dumps(meta))
                    if first_manifest is None:
                        first_manifest = manifest
        command = [sys.executable, str(ROOT / 'scripts/freeze_panel.py'),
                   '--run-root', str(run)]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        output = run / 'model_definition_frozen.json'
        frozen = json.loads(output.read_text())
        assert frozen['expected_final_test_folds'] == 640
        assert len(frozen['allowed_source_experiments']) == 16
        assert [len(frozen[key]) for key in
                ('model_specs', 'reused_p6_model_specs', 'p8_model_specs')] == [5, 6, 5]
        output.unlink()
        first_manifest.write_text('sample_id\nwrong:0\nwrong:1\n')
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0
        assert not output.exists()
    print('Complete validation releases the panel, mismatched sample IDs do not')


if __name__ == '__main__':
    test_commands()
    test_boundaries_and_conloss()
    test_freeze_refuses_missing_validation()
    test_freeze_complete_validation()
    print('RELEASE_WORKFLOW_OK')
