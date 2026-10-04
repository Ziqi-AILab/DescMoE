import argparse
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
from dataset_graph import construct_dataset, mol_collate_func
from transformer_graph_finetune import make_model
from utils import ScheduledOptim, get_options, get_loss, cal_loss, evaluate, scaffold_split
from collections import defaultdict
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pretrain'))
from prior_moe import assign_expert_ids, PriorMoEFFN


PRETRAIN_ALIGNED_BACKBONE = {
    'N': 8,
    'h': 8,
    'N_dense': 2,
    'd_model': 256,
    'scale_norm': True,
    'distance_matrix_kernel': 'exp',
    'dropout': 0.0,
}


def _apply_backbone_profile(model_params, profile, encoder_layers_override):
    if profile == 'pretrain_aligned':
        model_params.update(PRETRAIN_ALIGNED_BACKBONE)
    if encoder_layers_override:
        if encoder_layers_override <= 0:
            raise ValueError('--encoder_layers_override must be > 0')
        model_params['N'] = encoder_layers_override

    resolved = {key: model_params[key] for key in PRETRAIN_ALIGNED_BACKBONE}
    print('RESOLVED_BACKBONE_CONFIG: profile={} {}'.format(profile, resolved))


def _is_truncated_encoder_key(key, encoder_layers):
    prefix = 'encoder.layers.'
    if not key.startswith(prefix):
        return False
    try:
        layer_idx = int(key[len(prefix):].split('.', 1)[0])
    except (TypeError, ValueError):
        return False
    return layer_idx >= encoder_layers


def _load_pretrained_checkpoint(model, ckpt_path, device, strict_backbone_load=False):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = dict(checkpoint['state_dict'])
    expert_layers = sorted({
        int(key.split('.')[2])
        for key in state_dict
        if key.startswith('encoder.layers.') and '.feed_forward.experts.' in key
    })
    print('CHECKPOINT_EXPERT_LAYERS: {}'.format(expert_layers))
    state_dict.pop('pos_embed.pe.weight', None)
    incompatible = model.load_state_dict(state_dict, strict=False)

    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    print('CHECKPOINT_MISSING_KEYS: {}'.format(missing))
    print('CHECKPOINT_UNEXPECTED_KEYS: {}'.format(unexpected))

    if strict_backbone_load:
        allowed_missing = [
            key for key in missing
            if key == 'pos_embed.pe.weight' or key.startswith('generator.')
        ]
        bad_missing = [key for key in missing if key not in allowed_missing]
        encoder_layers = len(model.encoder.layers)
        allowed_unexpected = [
            key for key in unexpected
            if _is_truncated_encoder_key(key, encoder_layers)
        ]
        bad_unexpected = [key for key in unexpected if key not in allowed_unexpected]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                'Strict backbone checkpoint validation failed. '
                'bad_missing={}; bad_unexpected={}'.format(
                    bad_missing, bad_unexpected))
        print(
            'BACKBONE_LOAD_OK: checkpoint={} encoder_layers={} '
            'allowed_missing={} allowed_truncated={}'.format(
                ckpt_path, encoder_layers, len(allowed_missing),
                len(allowed_unexpected)))
    else:
        print('Successful Loading the ckpt')

    return incompatible


def _freeze_inactive(model, expert_ids):
    """Freeze experts not in the current batch's active set."""
    for m in model.modules():
        if isinstance(m, PriorMoEFFN):
            m.freeze_inactive_experts(expert_ids)


def _unfreeze_all(model):
    """Restore requires_grad on all experts."""
    for m in model.modules():
        if isinstance(m, PriorMoEFFN):
            m.unfreeze_all_experts()


def _standard_moe_aux_loss(model, device):
    """Collect load-balancing losses from StandardMoEFFN layers."""
    aux = None
    for m in model.modules():
        if hasattr(m, 'aux_loss'):
            value = m.aux_loss
            if torch.is_tensor(value):
                aux = value if aux is None else aux + value
    if aux is None:
        return torch.tensor(0.0, device=device)
    return aux


def model_train(model, train_dataset, valid_dataset, model_params, train_params, dataset_name, experiment_name,fold,
                # ---- DescMoE flags ----
                use_prior_moe=False, num_experts=8, expert_property='tpsa',                freeze_inactive_experts=False,
                use_standard_moe=False,
                num_workers=4):
    # build data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,
                              shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)

    valid_loader = DataLoader(dataset=valid_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,
                              shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

    # build loss function
    criterion = get_loss(train_params['loss_function'])

    # build optimizer
    optimizer = ScheduledOptim(torch.optim.Adam(model.parameters(), lr=0),
                               train_params['warmup_factor'], model_params['d_model'],
                               train_params['total_warmup_steps'])

    best_valid_metric = float('inf') if train_params['task'] == 'regression' else float('-inf')
    best_epoch = -1
    best_valid_result, best_valid_bedding = None, None

    for epoch in range(train_params['total_epochs']):
        # train
        train_loss = list()
        model.train()
        for batch in tqdm(train_loader):
            smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
            adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch, max_length, max_length)
            node_features = node_features.to(train_params['device'])  # (batch, max_length, d_node)
            edge_features = edge_features.to(train_params['device'])  # (batch, max_length, max_length, d_edge)
            y_true = y_true.to(train_params['device'])  # (batch, task_numbers)
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
            # ---- DescMoE: compute expert_ids from SMILES ----
            mol_expert_ids = None
            if use_prior_moe:
                eid_list = assign_expert_ids(smile_list, property_name=expert_property, num_experts=num_experts)
                mol_expert_ids = torch.tensor(eid_list, dtype=torch.long, device=train_params['device'])
                if freeze_inactive_experts:
                    _freeze_inactive(model, mol_expert_ids)
            # (batch, task_numbers)
            y_pred, _ = model(node_features, batch_mask, adjacency_matrix, edge_features, expert_ids=mol_expert_ids)
            loss = cal_loss(y_true, y_pred, train_params['loss_function'], criterion,
                            0, 1, train_params['device'])
            if use_standard_moe:
                loss = loss + _standard_moe_aux_loss(model, train_params['device'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
            if use_prior_moe and freeze_inactive_experts:
                _unfreeze_all(model)
            train_loss.append(loss.detach().item())

        # valid
        model.eval()
        with torch.no_grad():
            valid_true, valid_pred, valid_smile, valid_embedding = list(), list(), list(), list()
            for batch in tqdm(valid_loader):
                smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
                adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch, max_length, max_length)
                node_features = node_features.to(train_params['device'])  # (batch, max_length, d_node)
                edge_features = edge_features.to(train_params['device'])  # (batch, max_length, max_length, d_edge)
                batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
                # ---- DescMoE: compute expert_ids ----
                mol_expert_ids = None
                if use_prior_moe:
                    eid_list = assign_expert_ids(smile_list, property_name=expert_property, num_experts=num_experts)
                    mol_expert_ids = torch.tensor(eid_list, dtype=torch.long, device=train_params['device'])
                # (batch, task_numbers)
                y_pred, y_embedding = model(node_features, batch_mask, adjacency_matrix, edge_features, expert_ids=mol_expert_ids)

                y_true = y_true.numpy()  # (batch, task_numbers)
                y_pred = y_pred.detach().cpu().numpy()  # (batch, task_numbers)
                y_embedding = y_embedding.detach().cpu().numpy()

                valid_true.append(y_true)
                valid_pred.append(y_pred)
                valid_smile.append(smile_list)
                valid_embedding.append(y_embedding)

            valid_true, valid_pred = np.concatenate(valid_true, axis=0), np.concatenate(valid_pred, axis=0)
            valid_smile, valid_embedding = np.concatenate(valid_smile, axis=0), np.concatenate(valid_embedding, axis=0)

        valid_result = evaluate(valid_true, valid_pred, valid_smile,
                                requirement=['sample', train_params['loss_function'], train_params['metric']],
                                data_mean=0, data_std=1, data_task=train_params['task'])

        # save and print message in graph regression
        if train_params['task'] == 'regression':
            if valid_result[train_params['metric']] < best_valid_metric:
                best_valid_metric = valid_result[train_params['metric']]
                best_epoch = epoch + 1
                best_valid_result = valid_result
                best_valid_bedding = valid_embedding
                torch.save({'state_dict': model.state_dict(),
                            'best_epoch': best_epoch,
                            f'best_valid_{train_params["metric"]}': best_valid_metric},
                           f'./Model/{dataset_name}/best_model_{experiment_name}_{dataset_name}_fold_{fold}.pt')

            print("Epoch {}, learning rate {:.6f}, "
                  "train {}: {:.4f}, "
                  "valid {}: {:.4f}, "
                  "best epoch {}, best valid {}: {:.4f}"
                  .format(epoch + 1, optimizer.view_lr(),
                          train_params['loss_function'], np.mean(train_loss),
                          train_params['loss_function'], valid_result[train_params['loss_function']],
                          best_epoch, train_params['metric'], best_valid_metric
                          ))

        # save and print message in graph classification
        else:
            if valid_result[train_params['metric']] > best_valid_metric:
                best_valid_metric = valid_result[train_params['metric']]
                best_epoch = epoch + 1
                best_valid_result = valid_result
                best_valid_bedding = valid_embedding
                torch.save({'state_dict': model.state_dict(),
                            'best_epoch': best_epoch,
                            f'best_valid_{train_params["metric"]}': best_valid_metric},
                           f'./Model/{dataset_name}/best_model_{experiment_name}_{dataset_name}_fold_{fold}.pt')

            print("Epoch {}, learning rate {:.6f}, "
                  "train {}: {:.4f}, "
                  "valid {}: {:.4f}, "
                  "valid {}: {:.4f}, "
                  "best epoch {}, best valid {}: {:.4f}"
                  .format(epoch + 1, optimizer.view_lr(),
                          train_params['loss_function'], np.mean(train_loss),
                          train_params['loss_function'], valid_result[train_params['loss_function']],
                          train_params['metric'], valid_result[train_params['metric']],
                          best_epoch, train_params['metric'], best_valid_metric
                          ))

        # early stop
        if abs(best_epoch - epoch) >= 20:
            print("=" * 20 + ' early stop ' + "=" * 20)
            break

    return best_valid_result, best_valid_bedding


def model_test(checkpoint, test_dataset, model_params, train_params,
               use_prior_moe=False, num_experts=8, expert_property='tpsa',
               freeze_inactive_experts=False,
               use_standard_moe=False,
               num_workers=4):
    # build loader
    test_loader = DataLoader(dataset=test_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,
                             shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

    # build model
    model = make_model(**model_params)
    model.to(train_params['device'])
    model.load_state_dict(checkpoint['state_dict'])

    # test
    model.eval()
    with torch.no_grad():
        test_true, test_pred, test_smile, test_embedding = list(), list(), list(), list()
        for batch in tqdm(test_loader):
            smile_list, adjacency_matrix, node_features, edge_features, y_true = batch
            adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch, max_length, max_length)
            node_features = node_features.to(train_params['device'])  # (batch, max_length, d_node)
            edge_features = edge_features.to(train_params['device'])  # (batch, max_length, max_length, d_edge)
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
            # ---- DescMoE: compute expert_ids ----
            mol_expert_ids = None
            if use_prior_moe:
                eid_list = assign_expert_ids(smile_list, property_name=expert_property, num_experts=num_experts)
                mol_expert_ids = torch.tensor(eid_list, dtype=torch.long, device=train_params['device'])
            # (batch, task_numbers)
            y_pred, y_embedding = model(node_features, batch_mask, adjacency_matrix, edge_features, expert_ids=mol_expert_ids)

            y_true = y_true.numpy()  # (batch, task_numbers)
            y_pred = y_pred.detach().cpu().numpy()  # (batch, task_numbers)
            y_embedding = y_embedding.detach().cpu().numpy()

            test_true.append(y_true)
            test_pred.append(y_pred)
            test_smile.append(smile_list)
            test_embedding.append(y_embedding)
        test_true, test_pred = np.concatenate(test_true, axis=0), np.concatenate(test_pred, axis=0)
        test_smile, test_embedding = np.concatenate(test_smile, axis=0), np.concatenate(test_embedding, axis=0)
    test_result = evaluate(test_true, test_pred, test_smile,
                           requirement=['sample', train_params['loss_function'], train_params['metric']],
                           data_mean=0, data_std=1, data_task=train_params['task'])

    print("test {}: {:.4f}".format(train_params['metric'], test_result[train_params['metric']]))

    return test_result, test_embedding


if __name__ == '__main__':
    # init args
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seeds", default=np.random.randint(10000))
    parser.add_argument("--gpu", type=str, help='gpu', default=-1)
    parser.add_argument("--fold", type=int, help='the number of k-fold', default=5)
    parser.add_argument("--fold_indices", type=str, default=None,
                        help='Optional comma-separated 1-based fold indices to run, e.g. 3,4,5.')
    parser.add_argument("--dataset", type=str, help='choose a dataset', default='esol')
    parser.add_argument("--split", type=str, help="choose the split type", default='random',
                        choices=['random', 'scaffold', 'cv'])
    parser.add_argument("--ckpt_path", type=str, help="ckpt-path", default=None)
    parser.add_argument("--experiment_name", type=str, help="experiment_nam", default=None)
    # ---- DescMoE flags ----
    parser.add_argument("--use_prior_moe", action='store_true', default=False)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--expert_property", type=str, default='tpsa',
                        choices=['tpsa', 'logp', 'mw', 'n_heavy', 'n_rings'])
    parser.add_argument("--expert_type", type=str, default='ffn',
                        choices=['ffn', 'kan', 'mixed'])
    parser.add_argument("--kan_expert_indices", type=str, default=None)
    parser.add_argument("--kan_grid_size", type=int, default=5)
    parser.add_argument("--kan_spline_order", type=int, default=3)
    parser.add_argument("--moe_layer_mode", type=str, default='all',
                        choices=['all', 'last', 'odd', 'even', 'none'],
                        help='Which transformer FFN layers use learned MoE or DescMoE. odd/even use 1-based layer numbers.')
    parser.add_argument("--moe_layer_indices", type=str, default=None,
                        help='Optional comma-separated zero-based layer indices overriding --moe_layer_mode.')
    parser.add_argument("--moe_d_ff", type=int, default=0)
    parser.add_argument("--use_standard_moe", action='store_true', default=False,
                        help='Use learnable routed StandardMoEFFN instead of vanilla/prior FFN.')
    parser.add_argument("--moe_top_k", type=int, default=2,
                        help='Number of experts activated per token for standard MoE.')
    parser.add_argument("--moe_aux_loss_coeff", type=float, default=0.01,
                        help='Load-balancing auxiliary loss coefficient for standard MoE.')
    parser.add_argument("--freeze_inactive_experts", action='store_true', default=False,
                        help='During finetune, freeze experts not assigned to the current batch.')
    parser.add_argument("--num_workers", type=int, default=4,
                        help='DataLoader num_workers (set 0 for large datasets to avoid fork OOM)')
    parser.add_argument("--batch_size_override", type=int, default=0,
                        help='Override dataset default batch size when > 0.')
    parser.add_argument('--backbone_profile', type=str, default='dataset_default',
                        choices=['dataset_default', 'pretrain_aligned'],
                        help='Use dataset defaults or the encoder configuration used in pretraining.')
    parser.add_argument('--encoder_layers_override', type=int, default=0,
                        help='Override encoder depth after applying --backbone_profile; 0 keeps the profile value.')
    parser.add_argument('--strict_backbone_load', action='store_true', default=False,
                        help='Fail if pretrained backbone keys do not match the resolved encoder.')
    parser.add_argument('--validate_checkpoint_only', action='store_true', default=False,
                        help='Build the resolved model, validate its checkpoint, and exit before training.')
    args = parser.parse_args()

    # parse DescMoE kwargs
    kan_expert_indices = None
    if args.kan_expert_indices:
        kan_expert_indices = [int(x) for x in args.kan_expert_indices.split(',')]
    moe_kwargs = dict(
        use_prior_moe=args.use_prior_moe,
        num_experts=args.num_experts,
        expert_type=args.expert_type,
        kan_expert_indices=kan_expert_indices,
        kan_grid_size=args.kan_grid_size,
        kan_spline_order=args.kan_spline_order,
        moe_layer_mode=args.moe_layer_mode,
        moe_layer_indices=args.moe_layer_indices,
        moe_d_ff=args.moe_d_ff,
        use_standard_moe=args.use_standard_moe,
        moe_top_k=args.moe_top_k,
        moe_aux_loss_coeff=args.moe_aux_loss_coeff,
    )
    moe_train_kwargs = dict(
        use_prior_moe=args.use_prior_moe,
        num_experts=args.num_experts,
        expert_property=args.expert_property,
        freeze_inactive_experts=args.freeze_inactive_experts,
        use_standard_moe=args.use_standard_moe,
        num_workers=args.num_workers,
    )

    # load options
    model_params, train_params = get_options(args.dataset)
    _apply_backbone_profile(
        model_params, args.backbone_profile, args.encoder_layers_override)
    if (args.backbone_profile == 'pretrain_aligned'
            and args.dataset == 'hiv'
            and args.batch_size_override == 0):
        args.batch_size_override = 12
        print('Aligned HIV safety override: batch_size=12')
    if args.batch_size_override < 0:
        raise ValueError('--batch_size_override must be >= 0')
    if args.batch_size_override > 0:
        old_batch_size = train_params['batch_size']
        train_params['batch_size'] = args.batch_size_override
        print(f"Batch size override: {old_batch_size} -> {train_params['batch_size']}")
    # ---- inject DescMoE flags into model_params so all make_model calls get them ----
    model_params.update(moe_kwargs)

    # init device and seed
    print(f"Seed: {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        train_params['device'] = torch.device(f'cuda:{args.gpu}')
        torch.cuda.manual_seed(args.seed)
    else:
        train_params['device'] = torch.device('cpu')

    # load data
    if train_params['task'] == 'regression':
        with open(f'./Data/{args.dataset}/preprocess/{args.dataset}.pickle', 'rb') as f:
            [data_mol, data_label, data_mean, data_std] = pkl.load(f)
    else:
        with open(f'./Data/{args.dataset}/preprocess/{args.dataset}.pickle', 'rb') as f:
            [data_mol, data_label] = pkl.load(f)

    # calculate the padding
    model_params['max_length'] = max([data.GetNumAtoms() for data in data_mol])
    print(f"Max padding length is: {model_params['max_length']}")

    if args.validate_checkpoint_only:
        if args.ckpt_path is None:
            raise ValueError('--validate_checkpoint_only requires --ckpt_path')
        validation_model = make_model(**model_params).to(train_params['device'])
        _load_pretrained_checkpoint(
            validation_model, args.ckpt_path, train_params['device'],
            strict_backbone_load=args.strict_backbone_load)
        print('CHECKPOINT_VALIDATION_DONE: {}'.format(args.experiment_name))
        raise SystemExit(0)

    # construct dataset
    print('=' * 20 + ' construct dataset ' + '=' * 20)
    dataset = construct_dataset(data_mol, data_label, model_params['d_atom'], model_params['d_edge'], model_params['max_length'])
    total_metrics = defaultdict(list)
    if args.fold_indices:
        run_fold_indices = [int(x.strip()) - 1 for x in args.fold_indices.split(',') if x.strip()]
        bad_folds = [idx + 1 for idx in run_fold_indices if idx < 0 or idx >= args.fold]
        if bad_folds:
            raise ValueError(f'fold_indices out of range for fold={args.fold}: {bad_folds}')
    else:
        run_fold_indices = list(range(args.fold))

    # split dataset
    if args.split == 'scaffold':
        # we run the scaffold split 5 times for different random seed, which means different train/valid/test
        for idx in run_fold_indices:
            print('=' * 20 + f' train on fold {idx + 1} ' + '=' * 20)
            print(f"Seed: {args.seed+ idx}")
            np.random.seed(args.seed+idx)
            torch.manual_seed(args.seed+idx)
            if torch.cuda.is_available():
                train_params['device'] = torch.device(f'cuda:{args.gpu}')
                torch.cuda.manual_seed(args.seed+idx)
            # get dataset
            train_index, valid_index, test_index = scaffold_split(data_mol, frac=[0.8, 0.1, 0.1], balanced=True,
                                                                  include_chirality=False, ramdom_state=args.seed + idx)
            train_dataset, valid_dataset, test_dataset = dataset[train_index], dataset[valid_index], dataset[test_index]

            # calculate total warmup steps
            train_params['total_warmup_steps'] = \
                int(len(train_dataset) / train_params['batch_size']) * train_params['total_warmup_epochs']
            print('train warmup step is: {}'.format(train_params['total_warmup_steps']))

            if train_params['task'] == 'regression':
                train_params['mean'] = np.mean(np.array(data_label)[train_index])
                train_params['std'] = np.std(np.array(data_label)[train_index])
            else:
                train_params['mean'], train_params['std'] = 0, 1

            # define a model
            model = make_model(**model_params)
            model = model.to(train_params['device'])

            if args.ckpt_path is not None:
                _load_pretrained_checkpoint(
                    model, args.ckpt_path, train_params['device'],
                    strict_backbone_load=args.strict_backbone_load)


            # train and valid
            print(f"train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}")
            best_valid_result, _ = model_train(model, train_dataset, valid_dataset, model_params, train_params, args.dataset, args.experiment_name,idx + 1, **moe_train_kwargs)
            best_valid_csv = pd.DataFrame.from_dict({'smile': best_valid_result['smile'], 'actual': best_valid_result['label'], 'predict': best_valid_result['prediction']})
            best_valid_csv.to_csv(f'./Result/{args.dataset}/best_valid_result_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['valid'].append(best_valid_result[train_params['metric']])

            # test
            print('=' * 20 + f' test on fold {idx + 1} ' + '=' * 20)
            checkpoint = torch.load(f'./Model/{args.dataset}/best_model_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.pt', map_location=train_params['device'])
            test_result, test_embedding = model_test(checkpoint, test_dataset, model_params, train_params, **moe_train_kwargs)
            test_csv = pd.DataFrame.from_dict({'smile': test_result['smile'], 'actual': test_result['label'], 'predict': test_result['prediction']})
            test_csv.to_csv(f'./Result/{args.dataset}/best_test_result_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['test'].append(test_result[train_params['metric']])

            total_embedding = dict()
            for smile, embedding in zip(test_result['smile'], test_embedding):
                total_embedding[smile] = embedding

            embedding_path = (
                f'./Result/{args.dataset}/total_test_embedding_'
                f'{args.experiment_name}_{args.dataset}_fold_{idx + 1}.pickle')
            with open(embedding_path, 'wb') as fw:
                pkl.dump(total_embedding, fw)

        print('=' * 20 + ' summary ' + '=' * 20)
        print('Seed: {}'.format(args.seed))
        for local_idx, idx in enumerate(run_fold_indices):
            print('fold {}, valid {} = {:.4f}, test {} = {:.4f}'
                  .format(idx + 1,
                          train_params['metric'], total_metrics['valid'][local_idx],
                          train_params['metric'], total_metrics['test'][local_idx]))

        print('{} selected folds valid average {} = {:.4f} ± {:.4f}, test average {} = {:.4f} ± {:.4f}'
              .format(len(run_fold_indices),
                      train_params['metric'], np.nanmean(total_metrics['valid']), np.nanstd(total_metrics['valid']),
                      train_params['metric'], np.nanmean(total_metrics['test']), np.nanstd(total_metrics['test']),
                      ))
        print('=' * 20 + " finished! " + '=' * 20)

    elif args.split == 'random':
        # we run the random split 5 times for different random seed, which means different train/valid/test
        for idx in run_fold_indices:
            print('=' * 20 + f' train on fold {idx + 1} ' + '=' * 20)
            # print('=' * 20 + f' train on fold {idx + 1} ' + '=' * 20)
            print(f"Seed: {args.seed + idx}")
            np.random.seed(args.seed + idx)
            torch.manual_seed(args.seed + idx)
            if torch.cuda.is_available():
                train_params['device'] = torch.device(f'cuda:{args.gpu}')
                torch.cuda.manual_seed(args.seed + idx)
            # get dataset
            train_valid_dataset, test_dataset = train_test_split(dataset, test_size=0.1, random_state=args.seed)
            train_dataset, valid_dataset = train_test_split(train_valid_dataset, test_size=len(test_dataset),
                                                            random_state=args.seed)
            # calculate total warmup steps
            train_params['total_warmup_steps'] = \
                int(len(train_dataset) / train_params['batch_size']) * train_params['total_warmup_epochs']
            print('train warmup step is: {}'.format(train_params['total_warmup_steps']))

            # if train_params['task'] == 'regression':
            #     train_params['mean'] = np.mean(np.array(data_label)[train_index])
            #     train_params['std'] = np.std(np.array(data_label)[train_index])
            # else:
            #     train_params['mean'], train_params['std'] = 0, 1

            # define a model
            model = make_model(**model_params)
            model = model.to(train_params['device'])
            if args.ckpt_path is not None:
                _load_pretrained_checkpoint(
                    model, args.ckpt_path, train_params['device'],
                    strict_backbone_load=args.strict_backbone_load)

            # train and valid
            print(f"train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}")
            best_valid_result, _ = model_train(model, train_dataset, valid_dataset, model_params, train_params, args.dataset, args.experiment_name,idx + 1, **moe_train_kwargs)
            best_valid_csv = pd.DataFrame.from_dict({'actual': best_valid_result['label'], 'predict': best_valid_result['prediction']})
            best_valid_csv.to_csv(f'./Result/{args.dataset}/best_valid_result_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['valid'].append(best_valid_result[train_params['metric']])

            # test
            print('=' * 20 + f' test on fold {idx + 1} ' + '=' * 20)
            checkpoint = torch.load(f'./Model/{args.dataset}/best_model_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.pt')
            test_result, _ = model_test(checkpoint, test_dataset, model_params, train_params, **moe_train_kwargs)
            test_csv = pd.DataFrame.from_dict({'actual': test_result['label'], 'predict': test_result['prediction']})
            test_csv.to_csv(f'./Result/{args.dataset}/best_test_result_{args.experiment_name}_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['test'].append(test_result[train_params['metric']])

        print('=' * 20 + ' summary ' + '=' * 20)
        print('Seed: {}'.format(args.seed))
        for local_idx, idx in enumerate(run_fold_indices):
            print('fold {}, valid {} = {:.4f}, test {} = {:.4f}'
                  .format(idx + 1,
                          train_params['metric'], total_metrics['valid'][local_idx],
                          train_params['metric'], total_metrics['test'][local_idx]))

        print('{} selected folds valid average {} = {:.4f} ± {:.4f}, test average {} = {:.4f} ± {:.4f}'
              .format(len(run_fold_indices),
                      train_params['metric'], np.nanmean(total_metrics['valid']), np.nanstd(total_metrics['valid']),
                      train_params['metric'], np.nanmean(total_metrics['test']), np.nanstd(total_metrics['test']),
                      ))
        print('=' * 20 + " finished! " + '=' * 20)

    elif args.split == 'cv':
        # k-fold
        kf = KFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
        for idx, (train_index, valid_index) in enumerate(kf.split(X=dataset)):
            print('=' * 20 + f' train on fold {idx + 1} ' + '=' * 20)
            # get dataset
            train_dataset, valid_dataset = dataset[train_index], dataset[valid_index]

            # calculate total warmup steps
            train_params['total_warmup_steps'] = \
                int(len(train_dataset) / train_params['batch_size']) * train_params['total_warmup_epochs']
            print('train warmup step is: {}'.format(train_params['total_warmup_steps']))

            if train_params['task'] == 'regression':
                train_params['mean'] = np.mean(np.array(data_label)[train_index])
                train_params['std'] = np.std(np.array(data_label)[train_index])
            else:
                train_params['mean'], train_params['std'] = 0, 1

            # define a model
            model = make_model(**model_params)
            model = model.to(train_params['device'])

            if args.ckpt_path is not None:
                _load_pretrained_checkpoint(
                    model, args.ckpt_path, train_params['device'],
                    strict_backbone_load=args.strict_backbone_load)

            # train and valid
            print(f"train size: {len(train_dataset)}, valid size: {len(valid_dataset)}")
            best_valid_result, best_valid_embedding = model_train(model, train_dataset, valid_dataset, model_params, train_params, args.dataset, idx + 1, **moe_train_kwargs)
            best_valid_csv = pd.DataFrame.from_dict({'smile': best_valid_result['smile'], 'actual': best_valid_result['label'], 'predict': best_valid_result['prediction']})
            best_valid_csv.to_csv(f'./Result/{args.dataset}/best_valid_result_{args.dataset}_fold_{idx + 1}.csv', sep=',', index=False, encoding='UTF-8')
            total_metrics['cv'].append(best_valid_result[train_params['metric']])

            total_embedding = dict()
            for smile, embedding in zip(best_valid_result['smile'], best_valid_embedding):
                total_embedding[smile] = embedding

            with open(f'./Result/{args.dataset}/total_valid_embedding_fold_{idx + 1}.pickle', 'wb') as fw:
                pkl.dump(total_embedding, fw)

        print('=' * 20 + ' summary ' + '=' * 20)
        print('Seed: {}'.format(args.seed))
        for idx in range(args.fold):
            print('fold {} {} = {:.4f}'
                  .format(idx + 1,
                          train_params['metric'], total_metrics['cv'][idx]))
        print('{} folds {} = {:.4f} ± {:.4f}'
              .format(args.fold,
                      train_params['metric'], np.nanmean(total_metrics['cv']), np.nanstd(total_metrics['cv'])))
        print('=' * 20 + " finished! " + '=' * 20)
    else:
        raise Exception('We only support random split, scaffold split and cv!')
