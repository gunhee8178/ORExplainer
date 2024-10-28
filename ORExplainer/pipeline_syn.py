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

from gnn.GNN_syn import NodeGCN
from gnn.utils import preprocess_features, preprocess_model_keys
from utils.metrics import auc, top_k_sparsity, top_k_ood, top_k_recall

from explainer.ORExplainer import ORExplainer

parser = argparse.ArgumentParser()
parser.add_argument('--hidden', default=60, type=int)
parser.add_argument('--device', default=0, type=int, help='CPU or GPU.')
parser.add_argument('--seed', default=42, type=int)

parser.add_argument('--dataset', default='syn4')

parser.add_argument('--ood', default="", type=str)
parser.add_argument('--version', default='4', type=int)

parser.add_argument('--date', default='1001')

parser.add_argument('--alpha', default=1.0, type=float)

args = parser.parse_args()

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def pipeline_NC(txt_dir):
    # data load
    if args.ood == "":
        with open(f'../dataset/{args.dataset}/{args.dataset}.pkl', 'rb') as fin:
            data = pickle.load(fin).to(device)
    else:
        with open(f'../dataset/{args.dataset}/{args.dataset}_{args.ood}.pkl', 'rb') as fin:
            data = pickle.load(fin).to(device)

    if 'syn2' not in args.dataset:
        data.x = preprocess_features(data.x)

    # 바꿀 부분
    node_indice = range(*args.node_indice)

    gnnNets = NodeGCN(num_features=data.x.squeeze().shape[1], num_classes=args.num_classes, device=device)
    model_state_dict = torch.load(f"./gnn/pretrained/{args.dataset}/best_model")

    process_model_keys = True
    if process_model_keys:
        print(
            f"This model obtained: Train Acc: {model_state_dict['train_acc']:.4f}, "
            f"Val Acc: {model_state_dict['val_acc']:.4f}, "
            f"Test Acc: {model_state_dict['test_acc']:.4f}.")

        gnnNets.load_state_dict(preprocess_model_keys(model_state_dict))

    data = data.to(device)
    gnnNets = gnnNets.to(device)
    gnnNets.train()

    explainer = ORExplainer(gnnNets, epochs=args.epochs, lr=args.lr, num_classes=args.num_classes, args=args,
                            node_indice=node_indice)


    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tic = time.perf_counter()

    explainer.get_explanation_network(data, is_graph_classification=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    toc = time.perf_counter()
    training_duration = toc - tic
    print(f"training time is {training_duration}s ")


    duration = 0.0
    emb, logits = gnnNets(data.to(device))
    ori_label = logits.argmax(dim=1)


    explainer.get_explanation_network(None, args.epochs, is_graph_classification=False)

    # test
    exc_a = 0
    exc_b = 0

    # OOD
    mask_max, mask_min= 0, 0
    ground_truth = []
    concat_mask = []

    motif_edge_recall = []
    ood_edge_recall = []
    mask_sum = []
    for ori_node_idx in tqdm(node_indice):
        tic = time.perf_counter()

        if args.ood != "":
            ood_edge = data.ood_edge.to(device)
        else:
            ood_edge = torch.zeros(data.edge_label.shape).to(device)

        data = data.to(device)
        x, edge_index, y, subset, kwargs = \
            explainer.get_subgraph(node_idx=ori_node_idx, x=data.x, edge_index=data.edge_index, y=data.y, edge_reals=data.edge_label, edge_ood=ood_edge)

        node_idx = int(torch.where(subset == ori_node_idx)[0])
        sub_emb = emb[subset]

        edge_reals = kwargs.get('edge_reals').bool()
        edge_ood = kwargs.get('edge_ood').bool()

        edge_mask = explainer.explain_edge_mask(x, edge_index, sub_emb, node_idx)
        # pred_label = explainer.get_node_prediction(node_idx, x, edge_index)
        edge_mask = edge_mask.cpu()

        if 'syn1' in args.dataset.lower() or 'syn2' in args.dataset.lower():
            edge_reals = edge_reals & \
                     (subset[edge_index[0]] // 5 == ori_node_idx // 5)

        elif 'syn3' in args.dataset.lower() or 'syn4' in args.dataset.lower():
            edge_reals = edge_reals & \
                         ((subset[edge_index[0]] - 511) // args.num_nodes == (ori_node_idx - 511) // args.num_nodes)


        if ori_label[ori_node_idx].item() != data.y[ori_node_idx].item():
            # print("Wrong..!")
            exc_a += 1

        if edge_index.shape[1] == 0 :
            exc_b += 1
            continue

        duration += time.perf_counter() - tic

        edge_mask = edge_mask.to(device)

        try:
            motif_edge_recall.append(top_k_recall(edge_reals, edge_mask, args.num_edges).item())
            ood_edge_recall.append(top_k_recall(edge_ood, edge_mask, args.num_edges).item())

            ground_truth.extend(edge_reals)
            concat_mask.extend(edge_mask)
            mask_sum.append(edge_mask.mean().item())
            mask_max += edge_mask.max().item()
            mask_min += edge_mask.min().item()

        except Exception as e:
            exc_b += 1
            print(ori_node_idx)
            continue

    try :
        auc_value = auc(torch.tensor(ground_truth), torch.tensor(concat_mask)).item()
        motif_edge_recall = sum(motif_edge_recall)/ len(motif_edge_recall)
        ood_edge_recall = sum(ood_edge_recall) / len(ood_edge_recall)
        mask_mean = sum(mask_sum) / len(mask_sum)
    except:
        auc_value = 0.0
        motif_edge_recall = 0.0
        ood_edge_recall = 0.0

    print(f"AUC: {auc_value:.4f}\n"
          # f"Motif: {motif_edge_recall:.4f}\n"
          f"Mask: {mask_mean:.4f}\n"
          f"OOD: {ood_edge_recall:.4f}\n")

    with open(txt_dir, 'a+') as txt:
        txt.write(
            f"Version: {explainer.ckpt_path + f'_{args.epochs}.pth'}\n"
            f"AUC: {auc_value:.4f}\n"
            f"Mask: {mask_mean:.4f}\n"
            # f"Motif: {motif_edge_recall:.4f}\n"
            f"OOD: {ood_edge_recall:.4f}\n\n")

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
        txt.write(  f"python {' '.join(sys.argv)}\n"
                    f"{' '.join(args_list)}\n\n" )

    pipeline_NC(f'{txt_dir}/{model_name}_{args.seed}.txt')
