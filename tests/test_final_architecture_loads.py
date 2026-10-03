#!/usr/bin/env python3
"""CPU synthetic checkpoint transfer for the five architectures in the final panel."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ['dense', 'concat', 'auxiliary', 'assigned', 'learned']
COMMON = """
import sys
import torch
torch.set_num_threads(1)
variant = sys.argv[2]
kwargs = dict(d_atom=115, d_edge=13, N=8, d_model=256, h=8,
              dropout=0., N_dense=2, distance_matrix_kernel='exp',
              scale_norm=True, dense_output_nonlinearity='mish', moe_layer_mode='none')
if variant in ('concat', 'auxiliary'):
    kwargs['descriptor_use'] = variant
elif variant == 'assigned':
    kwargs.update(use_prior_moe=True, num_experts=8, expert_type='ffn', moe_layer_mode='last')
elif variant == 'learned':
    kwargs.update(use_standard_moe=True, num_experts=8, moe_top_k=1,
                  router_granularity='molecule', moe_layer_mode='last')
"""


def main():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(ROOT / 'MoleSG'))
    with tempfile.TemporaryDirectory(prefix='paper101_load_') as tmp:
        path = str(Path(tmp) / 'synthetic.pt')
        for variant in VARIANTS:
            build = COMMON + """
from transformer_graph import make_model
model = make_model(**kwargs)
torch.save({'state_dict': model.state_dict()}, sys.argv[1])
"""
            load = COMMON + """
from transformer_graph_finetune import make_model
from train_graph import _load_pretrained_checkpoint
model = make_model(**kwargs)
_load_pretrained_checkpoint(model, sys.argv[1], torch.device('cpu'), strict_backbone_load=True)
assert len(model.encoder.layers) == 8
"""
            for code, directory in ((build, 'pretrain'), (load, 'Downstream')):
                subprocess.run([sys.executable, '-c', code, path, variant],
                    cwd=ROOT / 'MoleSG' / directory, env=env, check=True)
    print('FINAL_ARCHITECTURE_LOADS_OK: five architectures covering 16 configurations')


if __name__ == '__main__':
    main()
