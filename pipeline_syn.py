import os
import glob
import time
from time import sleep

import torch
import pickle
import numpy as np
import random
import sys
import argparse
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data

from Explainer.model_syn.orexplainer import ORExplainer

from Explainer.utils.metrics import mask_ood_ratio
from Explainer.utils.metrics import roc_auc, pr_auc, ap, top_k_recall
from GNN.model.gcn_syn import GCN

from GNN.utils import preprocess_features
from Explainer.configs.selector import Selector



parser = argparse.ArgumentParser()
parser.add_argument('--hidden', default=20, type=int)

parser.add_argument('--device', default=0, type=int, help='CPU or GPU.')

parser.add_argument('--seed', default=42, type=int)

parser.add_argument('--dataset', default='syn1')
parser.add_argument('--date', default='0901')

parser.add_argument('--ood', default="0", type=str)

parser.add_argument('--gamma', default=5.0, type=float)
parser.add_argument('--temp', default=1.0, type=float)

parser.add_argument('--K', default=0, type=int)
parser.add_argument('--lamda', default=0.5, type=float)

parser.add_argument('--coff_size', default=0.05, type=float)
parser.add_argument('--coff_ent', default=1.0, type=float)

parser.add_argument('--sparsity', default=12.0, type=float)
parser.add_argument('--connectivity', default="topk", type=str)
args = parser.parse_args()

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def pipeline_NC(txt_dir):
    num_oods = [0, 120, 130, 50, 60]
    num_ood = num_oods[int(args.dataset[-1])]

    no_ood_nodes = [0, 700, 1400, 871, 1231]
    no_ood_nodes = no_ood_nodes[int(args.dataset[-1])]

    nclass = args.num_classes
    if args.ood == '0':
        with open(f'./Dataset/{args.dataset}/{args.dataset}.pkl', 'rb') as fin:
            data = pickle.load(fin).to(device)
    else:
        with open(f'./Dataset/{args.dataset}/{args.dataset}_str_{int(args.ood)*num_ood}.pkl', 'rb') as fin:
            data = pickle.load(fin).to(device)

    data.y[no_ood_nodes:] = -1

    if args.dataset in ['syn1', 'syn2']:
        node_indice = range(400, 700, 5)
    elif args.dataset in ['syn3']:
        node_indice = range(511, 871, 6)
    elif args.dataset in ['syn4']:
        node_indice = range(511, 800, 1)

    model = GCN(in_channels=data.x.shape[1], hidden_channels=args.hidden, out_channels=nclass)
    if 'syn2' not in args.dataset:
        data.x = preprocess_features(data.x)

    model_state_dict = torch.load(f"./GNN/check_point/{args.dataset}/best_model")

    model.load_state_dict(model_state_dict['model_state_dict'])

    data = data.to(device)
    model.to(device)
    model.eval()
    full_logits = model(data)

    temp = args.temp
    energy = temp * -torch.logsumexp(full_logits/temp, dim=-1).squeeze().detach()
    
    # Train Explainer
    explainer = ORExplainer(model, epochs=args.epochs, lr=args.lr, num_classes=nclass, args=args,node_indice=node_indice)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tic = time.perf_counter()
    # Train Explainer
    explainer.get_explanation_network(data, energy)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    toc = time.perf_counter()
    training_duration = toc - tic
    print(f"training time is {training_duration}s ")

    # test
    exc_wrong = 0
    exc_isolated = 0

    # OOD
    ood_edges = 0
    selected_edges = 0


    ground_truth = []
    concat_mask = []

    motif_edge_recall = []
    mask_sum = []
    ood_edge_precision = []

    explainer.eval()
    for ori_node_idx in tqdm(node_indice):

        data = data.to(device)
        x, edge_index, y, subset, kwargs = \
            explainer.get_subgraph(node_idx=ori_node_idx, x=data.x, edge_index=data.edge_index, y=data.y) # , edge_reals=None, edge_ood=None

        node_idx = int(torch.where(ori_node_idx == subset)[0])
        sub_data = Data(x=x, edge_index=edge_index).to(device)
        logits = model(sub_data)
        ori_label = logits.argmax(dim=1)

        if ori_label[node_idx].item() != data.y[ori_node_idx].item():
            exc_wrong += 1
            continue

        if edge_index.shape[1] == 0:
            exc_isolated += 1
            continue

        if args.sparsity > 1.0:
            args.num_edges = int(args.sparsity)
        else:
            args.num_edges = int((edge_index.shape[1] / 2) * args.sparsity) * 2

        sub_emb = model.embed(x, edge_index)
        edge_mask = explainer.explain_edge_mask(x, edge_index, sub_emb, node_idx)
        nodesize = x.shape[0]
        edge_index, edge_mask = to_undirected(edge_index, edge_attr=edge_mask, reduce='mean', num_nodes=nodesize)

        # Get ground truth of edges
        if args.dataset in ['syn1', 'syn2']:
            edge_reals = (subset[edge_index[0]] // 5 == ori_node_idx // 5) & (subset[edge_index[0]] // 5 == ori_node_idx // 5)
        elif args.dataset in ['syn3', 'syn4']:
            edge_reals = (((subset[edge_index[0]] - 511) // args.num_nodes == (ori_node_idx - 511) // args.num_nodes) &
                          ((subset[edge_index[1]] - 511) // args.num_nodes == (ori_node_idx - 511) // args.num_nodes))


        top_k = min(max(args.num_edges, 2), edge_mask.shape[0]-1)
        try:
            threshold = float(edge_mask.reshape(-1).sort(descending=True).values[top_k])
        except Exception as e:
            threshold = 1.0

        hard_mask = (edge_mask > threshold)
        motif_edge_recall.append(top_k_recall(edge_reals, hard_mask, args.num_edges).item())

        ood_edge, selected_edge = mask_ood_ratio(hard_mask, top_k, edge_index, y)
        ood_edges  += ood_edge

        selected_edges += selected_edge

        ground_truth.extend(edge_reals)
        concat_mask.extend(edge_mask)

        mask_sum.append(edge_mask.mean().item())

    try :
        roc_auc_value = roc_auc(torch.tensor(ground_truth), torch.tensor(concat_mask)).item()
        pr_auc_value = pr_auc(torch.tensor(ground_truth), torch.tensor(concat_mask)).item()
        ap_value = ap(torch.tensor(ground_truth), torch.tensor(concat_mask)).item()
        motif_edge_recall = sum(motif_edge_recall)/ len(motif_edge_recall)
        mask_mean = sum(mask_sum) / len(mask_sum)
        ood_edge_precision = ood_edges / selected_edges if selected_edges > 0 else 0.0

    except Exception as e:
        print(e)
        roc_auc_value = 0.0
        pr_auc_value = 0.0
        ap_value = 0.0
        mask_mean = 0.0
    print(
            f"AUC : {roc_auc_value:.4f}\n"
            f"PR : {pr_auc_value:.4f}\n"
            f"AP : {ap_value:.4f}\n"
            f"Recall : {motif_edge_recall:.4f}\n"
            f"OOD: {ood_edge_precision:.4f}\n"
        )

    with open(txt_dir, 'a+') as txt:
        txt.write(
          f"AUC : {roc_auc_value:.4f}\n"
          f"PR : {pr_auc_value:.4f}\n"
          f"AP : {ap_value:.4f}\n"
          f"Recall : {motif_edge_recall:.4f}\n"
          f"OOD: {ood_edge_precision:.4f}\n\n"
        )
    return 0

if __name__ == '__main__':
    print(f"python {' '.join(sys.argv)}")
    for key, value in Selector(args.dataset, 0).args.items():
        setattr(args, key, value)

    args_dict = vars(args)
    args_list = [f"--{k} {v}" for k, v in args_dict.items() if v is not None]
    print(f"{' '.join(args_list)}")

    date_version = (args.date).replace("_", "/")
    model_name = 'OR'
    txt_dir = f'./result/{date_version}/{args.dataset}/{args.ood}/{model_name}/gamma_{args.gamma}/ent_{args.coff_ent}/size_{args.coff_size}'

    if not os.path.isdir(txt_dir):
        os.makedirs(txt_dir)

    with open(f'{txt_dir}/seed_{args.seed}.txt', 'a+') as txt:
        txt.write(  f"python {' '.join(sys.argv)}\n"
                    f"{' '.join(args_list)}\n\n" )

    pipeline_NC(f'{txt_dir}/seed_{args.seed}.txt')
