#!/usr/bin/env python3
"""Check public command imports from outside the checkout without loading data."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = [
    'scripts/run_model.py',
    'scripts/freeze_panel.py',
    'scripts/prepare_matched_control_assignments.py',
    'scripts/run_cpu_descriptor_ecfp_baselines.py',
    'scripts/summarize_reported_results.py',
    'analysis/build_p0_corrected_panel.py',
    'MoleSG/pretrain/train_total.py',
    'MoleSG/Downstream/train_graph_evalfix.py',
    'MoleSG/Data_process/data_geometirc_maeratio.py',
]


def main():
    for folder in ('MoleSG', 'scripts', 'analysis', 'tests'):
        for source in (ROOT / folder).rglob('*.py'):
            ast.parse(source.read_text(), filename=str(source))
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['PYTHONPATH'] = str(ROOT / 'MoleSG')
    with tempfile.TemporaryDirectory() as directory:
        for relative in COMMANDS:
            result = subprocess.run([sys.executable, str(ROOT / relative), '--help'],
                cwd=directory, env=env, capture_output=True, text=True)
            assert result.returncode == 0, f'{relative}\n{result.stderr}'
            print(f'OK {relative}')
    print('ENTRYPOINTS_OK: CPU import/help checks, no training or inference')


if __name__ == '__main__':
    main()
