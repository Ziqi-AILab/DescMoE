import argparse
import os
import sys
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import pickle as pkl
import numpy
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.model_selection import train_test_split
# from dataset_node_zinc import construct_dataset_mask, mol_collate_func_mask
MOLESG_ROOT = Path(__file__).resolve().parents[1]
if str(MOLESG_ROOT) not in sys.path:
    sys.path.insert(0, str(MOLESG_ROOT))

from transformer_graph import make_model
from utils import ScheduledOptim, get_options
# from graph_mae_model import GNNDecoder
from GNN_model import GNNDecoder
from Data_process.zinc_dataset_pretrain import Zinc_data, mol_collate_func_mask
from smiles_model import Smiles_encoder_model, encoder_model, smiles_decoder_model
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import RobertaConfig
from prior_moe import expert_contrastive_loss


def loss_function(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    y_mask = torch.where(y_true != 0., torch.full_like(y_true, 1), torch.full_like(y_true, 0))
    loss = torch.sum(torch.abs(y_true - y_pred * y_mask)) / torch.sum(y_mask)
    return loss
def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def model_train(model, atom_pred_decoder, smiles_encoder_model,encoder_model ,smiles_decoder_model, train_dataset, model_params, train_params, epochs, experiment_name,
                # ---- MoFE flags ----
                use_prior_moe=False, num_experts=8, expert_property='n_heavy',
                use_contrastive_expert_loss=False, contrastive_coff=0.1, contrastive_temperature=0.1,
                start_epoch=0,
                # ---- Standard MoE flag ----
                use_standard_moe=False):

    num_workers = int(os.environ.get("PRETRAIN_NUM_WORKERS", "4"))
    pin_memory = os.environ.get("PRETRAIN_PIN_MEMORY", "1") != "0"
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func_mask,
                              shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin_memory)


    # build optimizer
    optimizer = ScheduledOptim(torch.optim.Adam(model.parameters(), lr=0),
                               train_params['warmup_factor'], model_params['d_model'],
                               train_params['total_warmup_steps'])
    optimizer_dec_pred_atoms = ScheduledOptim(torch.optim.Adam(atom_pred_decoder.parameters(), lr=0),
                               train_params['warmup_factor'], model_params['d_model'],
                               train_params['total_warmup_steps'])
    optimizer_smiles_encoder_model = ScheduledOptim(torch.optim.Adam(smiles_encoder_model.parameters(), lr=0),
                                              train_params['warmup_factor'], model_params['d_model'],
                                              train_params['total_warmup_steps'])
    optimizer_encoder_model = ScheduledOptim(torch.optim.Adam(encoder_model.parameters(), lr=0),
                                              train_params['warmup_factor'], model_params['d_model'],
                                              train_params['total_warmup_steps'])
    optimizer_smiles_decoder_model = ScheduledOptim(torch.optim.Adam(smiles_decoder_model.parameters(), lr=0),
                                              train_params['warmup_factor'], model_params['d_model'],
                                              train_params['total_warmup_steps'])


    best_valid_loss = float('inf')
    loss_accum=0

    if not Path("./Model/" + experiment_name + "/compt/").exists():
        os.makedirs("./Model/" + experiment_name + "/compt/")
    if not Path("./Model/" + experiment_name + "/smiles_encoder/").exists():
        os.makedirs("./Model/" + experiment_name + "/smiles_encoder/")
    if not Path("./Model/" + experiment_name + "/total_encoder/").exists():
        os.makedirs("./Model/" + experiment_name + "/total_encoder/")


    epoch_times = []
    train_start_time = time.time()

    for epoch in range(start_epoch, epochs):
        # train
        epoch_start = time.time()

        model.train()
        atom_pred_decoder.train()
        smiles_encoder_model.train()
        encoder_model.train()
        smiles_decoder_model.train()

        for node_features, bond_features, adjacency_matrix, mask_node_labels, masked_atom_indices, token_ids, labels, edge_attr, edge_index, x, xmasked_atom_indices, xmask_node_labels, num_nodes, batch_expert_ids in tqdm(train_loader):

            adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch_size, max_length, max_length)
            node_features = node_features.to(train_params['device'])  # (batch_size, max_length, d_node)
            bond_features = bond_features.to(train_params['device'])
            # mask_node_labels = mask_node_labels.to(train_params['device'])
            # masked_atom_indices = masked_atom_indices.to(train_params['device'])
            token_ids = token_ids.to(train_params['device'])
            labels = labels.to(train_params['device'])
            edge_attr = edge_attr.to(train_params['device'])
            edge_index = edge_index.to(train_params['device'])
            x = x.to(train_params['device'])
            xmasked_atom_indices = xmasked_atom_indices.to(train_params['device'])
            xmask_node_labels =xmask_node_labels.to(train_params['device'])
            num_nodes =num_nodes.to(train_params['device'])

            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0   # (batch_size, max_length)

            # ---- MoFE: use pre-computed expert_ids from dataloader ----
            mol_expert_ids = None
            if use_prior_moe:
                mol_expert_ids = batch_expert_ids.to(train_params['device'])

            # (batch_size, max_length, 1)
            node_new_feature = model(node_features, batch_mask, adjacency_matrix, bond_features, expert_ids=mol_expert_ids)
            graph_features = node_new_feature + nn.Parameter(torch.zeros(1, 1, 256)).cuda()
            token_ids = token_ids.squeeze()

            smiles_embedding = smiles_encoder_model(token_ids)
            smiles_features = smiles_embedding + nn.Parameter(torch.zeros(1, 1, 256)).cuda()
            new_embedding = torch.cat([graph_features,smiles_features],dim=1)
            total_embedding = encoder_model(new_embedding)
            total_list=[]
            for a in range(total_embedding.size()[0]):
                total_list.append(total_embedding[a,:num_nodes[a],:])
            total_embedding_new = torch.cat(total_list, 0)
            pred_node = atom_pred_decoder(total_embedding_new, edge_index, edge_attr, xmasked_atom_indices)
            loss_graph = sce_loss(xmask_node_labels, pred_node[xmasked_atom_indices])
            loss_fct = CrossEntropyLoss()
            prediction_scores = smiles_decoder_model(total_embedding[:,26:,:])
            masked_lm_loss = loss_fct(prediction_scores.view(-1, 700), labels.long().view(-1))
            loss = loss_graph + masked_lm_loss

            # ---- MoFE: contrastive expert loss ----
            if use_contrastive_expert_loss and mol_expert_ids is not None:
                # mean-pool graph branch node embeddings to get molecule-level repr
                node_mask_float = batch_mask.unsqueeze(-1).float()  # (B, L, 1)
                mol_emb = (node_new_feature * node_mask_float).sum(dim=1) / node_mask_float.sum(dim=1).clamp(min=1)
                cl_loss = expert_contrastive_loss(
                    mol_emb, mol_expert_ids,
                    temperature=contrastive_temperature)
                loss = loss + contrastive_coff * cl_loss

            # ---- Standard MoE: add load-balancing aux loss ----
            if use_standard_moe:
                for layer in model.encoder.layers:
                    ff = layer.feed_forward
                    if hasattr(ff, 'aux_loss'):
                        loss = loss + ff.aux_loss

            # b = node_new_feature[0].cpu().detach().numpy()
            # numpy.save("node_new_feature.npy", b)
            # loss = loss_function(y_true, y_pred)
            optimizer.zero_grad()
            optimizer_dec_pred_atoms.zero_grad()
            optimizer_smiles_encoder_model.zero_grad()
            optimizer_encoder_model.zero_grad()
            optimizer_smiles_decoder_model.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
            optimizer_dec_pred_atoms.step_and_update_lr()
            optimizer_smiles_encoder_model.step_and_update_lr()
            optimizer_encoder_model.step_and_update_lr()
            optimizer_smiles_decoder_model.step_and_update_lr()


        # valid
        # model.eval()
        # with torch.no_grad():
        #     valid_result = dict()
        #     valid_result['label'], valid_result['prediction'], valid_result['loss'] = list(), list(), list()
        #     for batch in tqdm(valid_loader):
        #         adjacency_matrix, node_features, edge_features, y_true = batch
        #         adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch_size, max_length, max_length)
        #         node_features = node_features.to(train_params['device'])  # (batch_size, max_length, d_node)
        #         edge_features = edge_features.to(train_params['device'])  # (batch_size, max_length, max_length, d_edge)
        #
        #         batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch_size, max_length)
        #         # (batch_size, max_length, 1)
        #         y_pred = model(node_features, batch_mask, adjacency_matrix, edge_features)
        #
        #         y_true = y_true.numpy().flatten()
        #         y_pred = y_pred.cpu().detach().numpy().flatten()
        #         y_mask = np.where(y_true != 0., 1, 0)
        #
        #         times = 0
        #         for true, pred in zip(y_true, y_pred):
        #             if true != 0.:
        #                 times += 1
        #                 valid_result['label'].append(true)
        #                 valid_result['prediction'].append(pred)
        #                 valid_result['loss'].append(np.abs(true - pred))
        #         assert times == np.sum(y_mask)
        #
        #     valid_result['r2'] = metrics.r2_score(valid_result['label'], valid_result['prediction'])

        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        total_elapsed = time.time() - train_start_time
        eta = avg_epoch * (epochs - epoch - 1)
        print('Epoch {}, learning rate {:.6f}, train loss: {:.4f} | {:.1f}s/epoch, total {:.1f}s ({:.1f}h), ETA {:.1f}h'.format(
            epoch + 1, optimizer.view_lr(), loss,
            epoch_elapsed, total_elapsed, total_elapsed / 3600, eta / 3600))



        # save the model and valid result
        if loss < best_valid_loss:
            torch.save({'state_dict': model.state_dict(),
                        'best_epoch': epoch, 'best_valid_loss': best_valid_loss},
                       "./Model/"+experiment_name+ "/compt/compt_epoch{}.pth".format(epoch),)
            torch.save({'state_dict': smiles_encoder_model.state_dict(),
                        'best_epoch': epoch, 'best_valid_loss': best_valid_loss},
                       "./Model/"+ experiment_name+ "/smiles_encoder/smiles_encoder_epoch{}.pth".format(epoch),)
            torch.save({'state_dict': encoder_model.state_dict(),
                        'best_epoch': epoch, 'best_valid_loss': best_valid_loss},
                       "./Model/" + experiment_name + "/total_encoder/total_encoder_epoch{}.pth".format(epoch),)
            best_valid_loss = loss

        # Periodic checkpoint every 50 epochs (overwrite to save space)
        if (epoch + 1) % 50 == 0:
            for subdir, obj in [("compt", model), ("smiles_encoder", smiles_encoder_model), ("total_encoder", encoder_model)]:
                torch.save({'state_dict': obj.state_dict(),
                            'epoch': epoch, 'loss': loss},
                           "./Model/{}/{}/periodic_latest.pth".format(experiment_name, subdir))
            print("  Periodic checkpoint saved at epoch {}".format(epoch + 1))

        # temp test
        # if (epoch + 1) % 10 == 0:
        #     checkpoint = torch.load(f'./Model/{dataset_name}/best_model_{dataset_name}_{element}.pt')
        #     print('=' * 20 + ' middle test ' + '=' * 20)
        #     test_result = model_test(checkpoint, test_dataset, model_params, train_params)
        #     print("best epoch: {}, best valid loss: {:.4f}, test loss: {:.4f}, test r2: {:.4f}".format(
        #         checkpoint['best_epoch'], checkpoint['best_valid_loss'], np.mean(test_result['loss']), test_result['r2']
        #     ))
        #     print('=' * 40)
        #
        # # early stop
        # if abs(best_epoch - epoch) >= 20:
        #     print("=" * 20 + ' early stop ' + "=" * 20)
        #     break
        loss_accum += float(loss.cpu().item())

    return loss_accum


# def model_test(checkpoint, test_dataset, model_params, train_params):
#     # build loader
#     test_loader = DataLoader(dataset=test_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func_mask,
#                              shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
#
#     # build model
#     model = make_model(**model_params)
#     model.to(train_params['device'])
#     model.load_state_dict(checkpoint['state_dict'])
#
#     # test
#     model.eval()
#     with torch.no_grad():
#         test_result = dict()
#         test_result['label'], test_result['prediction'], test_result['loss'] = list(), list(), list()
#         for batch in tqdm(test_loader):
#             adjacency_matrix, node_features, edge_features = batch
#             adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch_size, max_length, max_length)
#             node_features = node_features.to(train_params['device'])  # (batch_size, max_length, d_node)
#             edge_features = edge_features.to(train_params['device'])  # (batch_size, max_length, max_length, d_edge)
#
#             batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch_size, max_length)
#             # (batch_size, max_length, 1)
#             y_pred = model(node_features, batch_mask, adjacency_matrix, edge_features)
#
#
#             y_true = y_true.numpy().flatten()
#             y_pred = y_pred.cpu().detach().numpy().flatten()
#             y_mask = np.where(y_true != 0., 1, 0)
#
#             times = 0
#             for true, pred in zip(y_true, y_pred):
#                 if true != 0.:
#                     times += 1
#                     test_result['label'].append(true)
#                     test_result['prediction'].append(pred)
#                     test_result['loss'].append(np.abs(true - pred))
#             assert times == np.sum(y_mask)
#     test_result['r2'] = metrics.r2_score(test_result['label'], test_result['prediction'])
#     test_result['best_valid_loss'] = checkpoint['best_valid_loss']
#     return test_result


if __name__ == '__main__':
    # init args
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seeds", default=np.random.randint(10000))
    parser.add_argument("--gpu", type=str, help='gpu', default=0)
    parser.add_argument("--dataset", type=str, help='nmrshiftdb/DFT8K_DFT/DFT8K_FF/Exp5K_DFT/Exp5K_FF', default='nmrshiftdb')
    parser.add_argument("--element", type=str, help="1H/13C", default='1H')
    parser.add_argument("--epochs", type=int, help="epochs", default=300)
    parser.add_argument("--experiment_name", type=str, help="experiment_name", default='')
    # ---- MoFE flags ----
    parser.add_argument("--use_prior_moe", action='store_true', default=False,
                        help="Replace Graph Transformer FFN with PriorMoEFFN")
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--expert_property", type=str, default='tpsa',
                        choices=['tpsa', 'logp', 'mw', 'n_heavy', 'n_rings'])
    parser.add_argument("--expert_type", type=str, default='ffn',
                        choices=['ffn', 'kan', 'mixed'])
    parser.add_argument("--kan_expert_indices", type=str, default=None,
                        help="Comma-separated indices for KAN experts when expert_type=mixed")
    parser.add_argument("--kan_grid_size", type=int, default=5)
    parser.add_argument("--kan_spline_order", type=int, default=3)
    parser.add_argument("--moe_layer_mode", type=str, default='all',
                        choices=['all', 'last', 'odd', 'even', 'none'],
                        help="Which transformer FFN layers use MoE/MoFE. odd/even use 1-based layer numbers.")
    parser.add_argument("--moe_layer_indices", type=str, default=None,
                        help="Optional comma-separated zero-based layer indices overriding --moe_layer_mode")
    parser.add_argument("--use_contrastive_expert_loss", action='store_true', default=False)
    parser.add_argument("--contrastive_coff", type=float, default=0.1)
    parser.add_argument("--contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--moe_d_ff", type=int, default=0,
                        help="Expert hidden dim (0 = same as d_model)")
    # ---- Standard MoE flags ----
    parser.add_argument("--use_standard_moe", action='store_true', default=False,
                        help="Replace FFN with StandardMoEFFN (learnable gating, TopK routing)")
    parser.add_argument("--moe_top_k", type=int, default=2,
                        help="Number of experts activated per token in standard MoE")
    parser.add_argument("--moe_aux_loss_coeff", type=float, default=0.01,
                        help="Load-balancing auxiliary loss coefficient for standard MoE")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path prefix to resume from, e.g. ./Model/exp/  (loads compt, smiles_encoder, total_encoder)")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Epoch to resume from (skips first start_epoch epochs)")
    args = parser.parse_args()

    # load options
    model_params, train_params = get_options(args.dataset)

    # init device and seed
    print(f"Seed: {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        train_params['device'] = torch.device(f'cuda:{args.gpu}')
        torch.cuda.manual_seed(args.seed)
    else:
        train_params['device'] = torch.device('cpu')

    #     mol = pkl.load(f)

    # print('=' * 20 + ' begin train ' + '=' * 20)
    # model_params['max_length'] = max([data.GetNumAtoms() for data in mol])
    # print(f"Max padding length is: {model_params['max_length']}")

    # train_mol, test_mol = train_test_split(mol, test_size=0.05,random_state=np.random.randint(10000))

    atom_hidden = 115
    bond_hidden = 13
    train_dataset = Zinc_data(
        expert_ids_path='../Data/zinc15/expert_ids.npz',
        expert_property=args.expert_property,
    )

    # valid_dataset = construct_dataset_mask(test_mol, model_params['d_atom'], model_params['d_edge'], model_params['max_length'], mask_rate = 0.25)
    # test_dataset = construct_dataset_mask(test_mol, model_params['d_atom'], model_params['d_edge'], model_params['max_length'], mask_rate = 0.25)

    # calculate total warmup factor and step
    train_params['warmup_factor'] = 0.2 if args.element == '1H' else 1.0
    train_params['total_warmup_steps'] = \
        int(len(train_dataset) / train_params['batch_size']) * train_params['total_warmup_epochs']
    print('train warmup step is: {}'.format(train_params['total_warmup_steps']))

    # define a model
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
    model = make_model(**model_params, **moe_kwargs)
    atom_pred_decoder = GNNDecoder(hidden_dim=256, out_dim=116)
    config = RobertaConfig(
        vocab_size=700,
        max_position_embeddings=515,
        num_attention_heads=2,
        num_hidden_layers=2,
        type_vocab_size=1,
        is_gpu=torch.cuda.is_available(),
        hidden_size= 256
    )

    smiles_encoder_model = Smiles_encoder_model(config)
    encoder_model = encoder_model(config)
    smiles_decoder_model = smiles_decoder_model(config)

    model = model.to(train_params['device'])
    atom_pred_decoder =atom_pred_decoder.to(train_params['device'])
    smiles_encoder_model = smiles_encoder_model.to(train_params['device'])
    encoder_model = encoder_model.to(train_params['device'])
    smiles_decoder_model = smiles_decoder_model.to(train_params['device'])

    # train and valid
    print(f"train size: {len(train_dataset)}")
    # Resume from checkpoint if requested
    if args.resume_from:
        import glob
        resume_dir = args.resume_from.rstrip('/')
        for subdir, target_model in [('compt', model), ('smiles_encoder', smiles_encoder_model), ('total_encoder', encoder_model)]:
            periodic = os.path.join(resume_dir, subdir, 'periodic_latest.pth')
            pattern = os.path.join(resume_dir, subdir, '*epoch*.pth')
            best_ckpts = sorted(glob.glob(pattern), key=lambda x: int(x.split('epoch')[1].split('.')[0]))
            # Pick whichever has a higher epoch: periodic or best
            chosen = None
            if os.path.exists(periodic):
                p_ckpt = torch.load(periodic, map_location=train_params['device'])
                p_epoch = p_ckpt.get('epoch', p_ckpt.get('best_epoch', -1))
                chosen = (periodic, p_ckpt, p_epoch)
            if best_ckpts:
                b_ckpt = torch.load(best_ckpts[-1], map_location=train_params['device'])
                b_epoch = b_ckpt.get('best_epoch', b_ckpt.get('epoch', -1))
                if chosen is None or b_epoch > chosen[2]:
                    chosen = (best_ckpts[-1], b_ckpt, b_epoch)
            if chosen:
                target_model.load_state_dict(chosen[1]['state_dict'])
                print(f"Loaded {subdir} from {chosen[0]} (epoch {chosen[2]})")
                if subdir == 'total_encoder' and args.start_epoch <= 0:
                    args.start_epoch = chosen[2] + 1
                    print(f"  Auto-set start_epoch={args.start_epoch}")
            else:
                print(f"WARNING: no checkpoint found for {subdir} in {resume_dir}")

    train_loss=model_train(model, atom_pred_decoder, smiles_encoder_model,encoder_model ,smiles_decoder_model, train_dataset, model_params, train_params, args.epochs, args.experiment_name,
                           use_prior_moe=args.use_prior_moe,
                           num_experts=args.num_experts,
                           expert_property=args.expert_property,
                           use_contrastive_expert_loss=args.use_contrastive_expert_loss,
                           contrastive_coff=args.contrastive_coff,
                           contrastive_temperature=args.contrastive_temperature,
                           start_epoch=args.start_epoch,
                           use_standard_moe=args.use_standard_moe)
    print(train_loss)
