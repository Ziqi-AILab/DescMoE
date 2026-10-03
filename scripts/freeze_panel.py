#!/usr/bin/env python3
"""Release the declared 16-model test panel after complete validation outputs."""
import argparse
import json
from pathlib import Path
import sys

from run_model import DATASETS, ROOT, model_matrix

sys.path.insert(0, str(ROOT / 'analysis'))
from build_evalfix_canonical import roc_auc, verify_sample_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()
    run = args.run_root.resolve()
    output = run / 'model_definition_frozen.json'
    if output.exists():
        raise SystemExit('A frozen definition already exists. Preserve it for this run.')
    specs = model_matrix()
    for spec in specs:
        for dataset in DATASETS:
            for fold in range(1, 6):
                stem = f'validation_result_{spec["result_exp"]}_{dataset}_fold_{fold}'
                path = run / 'results' / dataset / f'{stem}.csv'
                meta = json.loads(path.with_suffix('.json').read_text())
                expected = {'mode': 'finetune', 'split': 'validation',
                            'protocol': 'evalfix_full_coverage_v1',
                            'legacy_best_checkpoint': False, 'evaluation_coverage_ok': True,
                            'dataset': dataset, 'fold': fold, 'seed': 41 + fold,
                            'experiment_name': spec['result_exp']}
                if any(meta.get(key) != value for key, value in expected.items()):
                    raise ValueError(f'Validation metadata mismatch: {path}')
                verify_sample_manifest(path, Path(meta['sample_id_manifest']))
                auc, _, count = roc_auc(path)
                if count != meta['split_samples'] or count != meta['evaluated_samples']:
                    raise ValueError(f'Incomplete validation: {path}')
                if abs(auc - meta['validation_best_auc']) > 1e-8:
                    raise ValueError(f'Validation metric mismatch: {path}')
                if not Path(meta['checkpoint']).is_file():
                    raise FileNotFoundError(meta['checkpoint'])
    groups = {'model_specs': [], 'reused_p6_model_specs': [], 'p8_model_specs': []}
    reverse = {'dense_base': 'base', 'fixed_assignment': 'I1',
               'fixed_assignment_conloss': 'I1+I3',
               'descriptor_concatenation': 'descriptor_concat',
               'auxiliary_descriptor_prediction': 'descriptor_auxiliary',
               'quantile_assignment': 'quantile_routing',
               'stable_random_assignment': 'stable_random',
               'stable_random_assignment_conloss': 'stable_random_conloss',
               'learned_molecule_top1': 'learned_molecule_top1'}
    for spec in specs:
        item = dict(checkpoint_model=spec['model'], result_exp=spec['result_exp'],
                    partition=spec['partition'])
        field = 'role' if spec['source_group'] == 'model_specs' else 'control'
        item[field] = reverse[spec['control']]
        groups[spec['source_group']].append(item)
    payload = dict(status='frozen', final_test_release=True,
                   test_results_used_for_selection=False, final_matrix_models=16,
                   expected_final_test_folds=640,
                   allowed_source_experiments=[s['result_exp'] for s in specs], **groups)
    output.write_text(json.dumps(payload, indent=2) + '\n')
    print(f'640 validation folds checked. Final test released: {output}')


if __name__ == '__main__':
    main()
