import os
import glob
import time
import torch
import pickle
import numpy as np
import random
import sys
import argparse

from tqdm import tqdm

from configs.selector import Selector

from gnn.GNN_real import NodeGCN
from gnn.utils import preprocess_features, preprocess_model_keys

from utils.metrics import top_k_sparsity, mask_fidelity, mask_ood_ratio

from torch_geometric.data import Data
from explainer.ORExplainer import ORExplainer

parser = argparse.ArgumentParser()
parser.add_argument('--hidden', default=16, type=int)

parser.add_argument('--device', default=0, type=int, help='CPU or GPU.')

parser.add_argument('--seed', default=42, type=int)

parser.add_argument('--dataset', default='Cora')
parser.add_argument('--date', default='1001')

parser.add_argument('--ood', default="", type=str)
parser.add_argument('--version', default='4', type=int)

parser.add_argument('--alpha', default=0.1, type=float)

parser.add_argument('--sp', default=0.1, type=float)
args = parser.parse_args()

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def pipeline_NC(txt_dir):
    # data load
    with open(f'../dataset/{args.dataset}/{args.dataset}_ood.pkl', 'rb') as fin:
        data = pickle.load(fin)

    data.to(device)
    data.x = preprocess_features(data.x)

    node_indice = torch.where(((data.y>=0) & data.test_mask))[0].tolist()

    gnnNets = NodeGCN(num_features=data.x.shape[1], hidden_dim=16, num_classes=args.num_classes, device=device)
    model_state_dict = torch.load(f"./gnn/pretrained/{args.dataset}/best_model")

    process_model_keys = True
    if process_model_keys:
        print(
            f"This model obtained: Train Acc: {model_state_dict['train_acc']:.4f}, "
            f"Val Acc: {model_state_dict['val_acc']:.4f}, "
            f"Test Acc: {model_state_dict['test_acc']:.4f}.")

        gnnNets.load_state_dict(preprocess_model_keys(model_state_dict))

    gnnNets = gnnNets.to(device)
    gnnNets.eval()

    explainer = ORExplainer(gnnNets, epochs=args.epochs, lr=args.lr, num_classes=args.num_classes, args=args,
                            node_indice=node_indice)

    new_node_indice = []
    for idx in node_indice:
        _, edge_index, _, _, _ = \
            explainer.get_subgraph(node_idx=idx, x=data.x, edge_index=data.edge_index[:, ~data.inter_edge], y=data.y)
        if edge_index.shape[1] > 0:
            new_node_indice.append(idx)

    node_indice = new_node_indice
    explainer.node_indice = node_indice

    # ood
    if "rm" in args.ood:
        data.edge_index = data.edge_index[:, ~data.inter_edge]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tic = time.perf_counter()

    explainer.get_explanation_network(data, is_graph_classification=False)


    if torch.cuda.is_available():
        torch.cuda.synchronize()

    toc = time.perf_counter()
    training_duration = toc - tic
    print(f"training time is {training_duration}s ")

    # save model
    # torch.save(explainer.elayers.cpu().state_dict(), explainer.ckpt_path)
    # explainer.elayers.to(device)

    duration = 0.0
    data = data.to(device)
    emb, logits = gnnNets(data)

    explainer.get_explanation_network(None, args.epochs, is_graph_classification=False)

    exc_a = 0 # wrong prediction
    exc_b = 0 # isolated node

    mask_max, mask_min = 0, 0

    Fidelity_p = []
    Fidelity_m = []
    Sparsity = []
    OOD_ratio = []
    for ori_node_idx in tqdm(node_indice):
        tic = time.perf_counter()

        x, edge_index, y, subset, _ = \
            explainer.get_subgraph(node_idx=ori_node_idx, x=data.x, edge_index=data.edge_index, y=data.y)

        if edge_index.shape[1] == 0:
            exc_b += 1
            continue

        node_idx = int(torch.where(subset == ori_node_idx)[0])
        sub_emb = emb[subset]


        edge_mask = explainer.explain_edge_mask(x, edge_index, sub_emb, node_idx)

        sub_data = Data(x=x, edge_index=edge_index).to(device)
        _, logits = gnnNets(sub_data)
        pred_label = logits[node_idx].argmax().item()

        if pred_label != data.y[ori_node_idx].item():
            exc_a += 1
            # continue

        duration += time.perf_counter() - tic

        edge_mask = edge_mask.to(device)

        # print(edge_mask.max().item(), edge_mask.min().item())
        try:
            pass
            args.topk = int(edge_mask.shape[0] * args.sp)

            sub_data = Data(x=x, edge_index=edge_index).to(device)
            Fidelity_p.append(mask_fidelity(sub_data, edge_mask, gnnNets, data.y[ori_node_idx], args.topk , node_idx=node_idx))

            sub_data = Data(x=x, edge_index=edge_index).to(device)
            Fidelity_m.append(mask_fidelity(sub_data, edge_mask, gnnNets, data.y[ori_node_idx], args.topk, node_idx=node_idx, version='minus'))
            Sparsity.append(top_k_sparsity(edge_mask, args.topk))

            ood_ratio = mask_ood_ratio(edge_mask, args.topk, edge_index, y)
            if ood_ratio != -1:
                OOD_ratio.append(ood_ratio)

            mask_max += edge_mask.max().item()
            mask_min += edge_mask.min().item()

        except Exception as e:
            exc_b += 1
            print(e)
            continue

    try:
        print(len(Fidelity_p))
        Fidelity_p = sum(Fidelity_p) / len(Fidelity_p)
        Fidelity_m = sum(Fidelity_m) / len(Fidelity_m)
        Sparsity = sum(Sparsity) / len(Sparsity)
        OOD_ratio = sum(OOD_ratio) / len(OOD_ratio)
    except Exception as e:
        print(e)
        auc_value = 0.0
        Fidelity_p = 0.0
        Fidelity_m = 0.0
        Sparsity = 0.0

    print(f"Fid+: {Fidelity_p:.4f}\n"
          f"Fid-: {Fidelity_m:.4f}\n"
          f"Sparsity: {Sparsity:.4f}\n"
          f"OOD_ratio: {OOD_ratio:.4f}\n")

    with open(txt_dir, 'a+') as txt:
        txt.write(
            f"Version: {explainer.ckpt_path + f'_{args.epochs}.pth'}\n"
            f"Fid+: {Fidelity_p:.4f}\n"
            f"Fid-: {Fidelity_m:.4f}\n"
            f"Sparsity: {Sparsity:.4f}\n"
            f"OOD_ratio: {OOD_ratio:.4f}\n\n")
    return 0


if __name__ == '__main__':
    print(f"python {' '.join(sys.argv)}")
    for key, value in Selector(args.dataset, args.version).args.items():
        setattr(args, key, value)

    args_dict = vars(args)
    args_list = [f"--{k} {v}" for k, v in args_dict.items() if v is not None]  # None 값이 아닌 항목만 문자열로 변환
    print(f"{' '.join(args_list)}")

    date_version = (args.date).replace("_", "/")
    if args.ood == "":
        txt_dir = f'./txt/{date_version}/{args.dataset}/default/alpha_{args.alpha}/ent_{args.coff_ent}/size_{args.coff_size}/seed_{args.seed}'
    else:
        txt_dir = f'./txt/{date_version}/{args.dataset}/{args.ood}/alpha_{args.alpha}/ent_{args.coff_ent}/size_{args.coff_size}/seed_{args.seed}'

    if not os.path.isdir(txt_dir):
        os.makedirs(txt_dir)

    model_name = 'ORExplainer'
    with open(f'{txt_dir}/{model_name}_{args.seed}.txt', 'a+') as txt:
        txt.write(f"python {' '.join(sys.argv)}\n"
                  f"{' '.join(args_list)}\n\n")

    pipeline_NC(f'{txt_dir}/{model_name}_{args.seed}.txt')
