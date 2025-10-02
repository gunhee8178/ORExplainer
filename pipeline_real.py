import os
import time

import torch
import pickle
import numpy as np
import random
import sys
import argparse
from torch_geometric.utils import to_undirected
from tqdm import tqdm

from torch_geometric.data import Data

from Explainer.model_real.orexplainer import ORExplainer

from Explainer.utils.metrics import mask_ood_ratio, mask_fidelity
from GNN.model.gcn_real import GCN


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--lr', default=5e-3, type=float)


parser.add_argument('--hidden', default=16, type=int)
parser.add_argument('--device', default=0, type=int, help='CPU or GPU.')

parser.add_argument('--seed', default=42, type=int)

parser.add_argument('--dataset', default='Citeseer')
parser.add_argument('--date', default='0929', help="For savefile")

parser.add_argument('--ood', default="0", type=str)

parser.add_argument('--gamma', default=0.0, type=float)
parser.add_argument('--temp', default=1.0, type=float)

parser.add_argument('--K', default=2, type=int)
parser.add_argument('--lamda', default=0.5, type=float)

parser.add_argument('--t0', default=1.0, type=float)
parser.add_argument('--t1', default=1.0, type=float)

parser.add_argument('--coff_size', default=1.0, type=float)
parser.add_argument('--coff_ent', default=5e-4, type=float)
parser.add_argument('--sample_bias', default=0.0, type=float)

parser.add_argument('--sparsity', default=0.1, type=float)
args = parser.parse_args()

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def pipeline_NC(txt_dir):

    # Load dataset
    if 'label' in args.ood: # Unseen Label
        with open(f'./Dataset/{args.dataset}/{args.dataset}_label.pkl', 'rb') as fin:
            data = pickle.load(fin).to(device)
            if '0' in args.ood: # without Unseen Label 
                valid_edge = (data.y[data.edge_index] >= 0).all(dim=0)
                data.edge_index = data.edge_index[:, valid_edge]
    else: # Featural OOD
        with open(f'./Dataset/{args.dataset}/{args.dataset}_feature.pkl', 'rb') as fin:
            data = pickle.load(fin).to(device)

        valid_edge = (data.y[data.edge_index] >= int(args.ood) * -1).all(dim=0)
        data.edge_index = data.edge_index[:, valid_edge]

    node_indice = torch.where((data.exp_mask == True))[0].cpu().tolist()[:300]
    nclass = data.y.max().item() + 1

    # Load GNN model
    model = GCN(in_channels=data.x.shape[1], hidden_channels=args.hidden, out_channels=nclass)

    if 'label' in args.ood:
        if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
            model_state_dict = torch.load(f"./GNN/check_point/planetoid/gcn_{args.dataset}_label.pt")
    else:
        if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
            model_state_dict = torch.load(f"./GNN/check_point/planetoid/gcn_{args.dataset}_feature.pt")

    model.load_state_dict(model_state_dict['model_state_dict'])

    data = data.to(device)
    model.to(device)
    model.eval()

    full_logits = model(data)
    temp = args.temp
    energy = temp * -torch.logsumexp(full_logits/temp, dim=-1).squeeze().detach()


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

    mask_sum = []

    Fidelity_p_p = []
    Fidelity_p_m = []
    Fidelity_l_p = []
    Fidelity_l_m = []

    avg_node = 0.0
    explainer.eval()
    for ori_node_idx in tqdm(node_indice):
        # Load data as Subgraph
        data = data.to(device)
        x, edge_index, y, subset, kwargs = \
            explainer.get_subgraph(node_idx=ori_node_idx, x=data.x, edge_index=data.edge_index, y=data.y) # , edge_reals=None, edge_ood=None

        avg_node += subset.shape[0]

        # Skip isolated nodes and wrong prediction nodes
        node_idx = int(torch.where(subset == ori_node_idx)[0])
        sub_data = Data(x=x, edge_index=edge_index).to(device)
        logits = model(sub_data)
        ori_label = logits.argmax(dim=1)

        if edge_index.shape[1] == 0:
            exc_isolated += 1
            continue

        if ori_label[node_idx].item() != data.y[ori_node_idx].item():
            exc_wrong += 1
            continue

        # if sparsity is bigger than 1, it means the number of edges. It is used as top_k in this case.
        if args.sparsity > 1.0:
            args.num_edges = int(args.sparsity)
        else:
            args.num_edges = int((edge_index.shape[1] / 2) * args.sparsity) * 2
        emb1, emb2 = model.embed(x, edge_index)
        sub_emb = torch.cat([x, emb1, emb2], dim=-1)

        # Explain
        edge_mask = explainer.explain_edge_mask(x, edge_index, sub_emb, node_idx)
        edge_mask = edge_mask.to(device)
        edge_index, edge_mask = to_undirected(edge_index, edge_attr=edge_mask, reduce='mean')

        # Evaluate
        top_k = min(max(args.num_edges, 2), edge_mask.shape[0]-1)
        try:
            threshold = float(edge_mask.reshape(-1).sort(descending=True).values[top_k])
        except Exception as e:
            threshold = 1.0
        hard_mask = (edge_mask > threshold)
        # print("threshold:", threshold)

        # Fidelity +
        sub_data = Data(x=x, edge_index=edge_index, y=y).to(device)
        fid_p, fid_l = mask_fidelity(sub_data, hard_mask, model, y, top_k, node_idx=node_idx)
        Fidelity_p_p.append(fid_p)
        Fidelity_l_p.append(fid_l)

        # Fidelity -
        sub_data = Data(x=x, edge_index=edge_index, y=y).to(device)
        fid_p, fid_l = mask_fidelity(sub_data, hard_mask, model, y, top_k, node_idx=node_idx, version='minus')
        Fidelity_p_m.append(fid_p)
        Fidelity_l_m.append(fid_l)

        # OOD ratio
        ood_edge, selected_edge = mask_ood_ratio(hard_mask, top_k, edge_index, y)
        ood_edges  += ood_edge

        selected_edges += selected_edge
        mask_sum.append(edge_mask.mean().item())

    try :
        mask_mean = sum(mask_sum) / len(mask_sum)
        Fidelity_p_p = sum(Fidelity_p_p) / len(Fidelity_p_p)
        Fidelity_p_m = sum(Fidelity_p_m) / len(Fidelity_p_m)
        Fidelity_l_p = sum(Fidelity_l_p) / len(Fidelity_l_p)
        Fidelity_l_m = sum(Fidelity_l_m) / len(Fidelity_l_m)
        ood_edge_precision = ood_edges / selected_edges if selected_edges > 0 else 0.0

    except Exception as e:
        print(e)
        Fidelity_p_p = 0.0
        Fidelity_p_m = 0.0
        Fidelity_l_m = 0.0
        mask_mean = 0.0
        ood_edge_precision = 0.0

    print(f"avg_node : {avg_node/len(node_indice):.2f}")
    print(f"Fid prob+ : {Fidelity_p_p:.4f}\n"
          f"Fid prob- : {Fidelity_p_m:.4f}\n"
          f"Fid label+ : {Fidelity_l_p:.4f}\n"
          f"Fid label- : {Fidelity_l_m:.4f}\n"
          f"OOD: {ood_edge_precision:.4f}\n"
    )

    with open(txt_dir, 'a+') as txt:
        txt.write(
            f"Fid prob+ : {Fidelity_p_p:.4f}\n"
            f"Fid prob- : {Fidelity_p_m:.4f}\n"
            f"Fid label+ : {Fidelity_l_p:.4f}\n"
            f"Fid label- : {Fidelity_l_m:.4f}\n"
            f"OOD: {ood_edge_precision:.4f}\n\n"
        )

    return 0

if __name__ == '__main__':
    print(f"python {' '.join(sys.argv)}")

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
