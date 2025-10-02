import torch
import numpy as np
from torch_geometric.data import Data
from typing import Optional
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, auc
from collections import defaultdict, deque

def roc_auc(edge_reals, edge_mask):
    if edge_mask.device != 'cpu':
        edge_mask = edge_mask.cpu()

    auc_score = roc_auc_score(edge_reals.cpu().numpy(), edge_mask.numpy())

    return auc_score

def pr_auc(edge_reals, edge_mask):
    if edge_mask.device != 'cpu':
        edge_mask = edge_mask.cpu()

    edge_reals = edge_reals.cpu().numpy()
    edge_mask = edge_mask.cpu().numpy()

    positive_ratio = np.sum(edge_reals) / len(edge_mask)
    random_classifier_auc = positive_ratio

    precision, recall, thresholds = precision_recall_curve(edge_reals, edge_mask)
    auc_score = auc(recall, precision)


    return auc_score

def ap(edge_reals, edge_mask):
    if edge_mask.device != 'cpu':
        edge_mask = edge_mask.cpu()

    edge_reals = edge_reals.cpu().numpy()
    edge_mask = edge_mask.cpu().numpy()

    ap_score = average_precision_score(edge_reals, edge_mask)

    return ap_score

def top_k_accuracy(edge_reals, edge_mask, top_k, undirected=True):
    if undirected:
        top_k = 2 * top_k

    threshold = float(edge_mask.reshape(-1).sort(descending=True).values[min(top_k, edge_mask.shape[0] - 1)])
    hard_mask = (edge_mask > threshold).cpu()

    accuracy = (edge_reals.bool().cpu() == hard_mask.bool()).sum() / (hard_mask.shape[0])
    return accuracy


def top_k_recall(edge_reals, hard_mask, top_k, undirected=True):
    recall = (edge_reals.bool() & hard_mask.bool()).sum() / top_k
    return recall


def edge_fidelity(data: Data, edge_mask: np.array, top_k: int,
                   gnnNets: torch.nn.Module, label: int,
                   target_id: int = -1, node_idx: Optional[int] = None,
                   undirected=True, version = 'plus'):
    """ return the fidelity score of the subgraph with top_k score edges  """
    if undirected:
        top_k = 2 * top_k
    all_nodes = np.arange(data.x.shape[0]).tolist()
    score = gnn_score(all_nodes, data, gnnNets, label, target_id, node_idx=node_idx,
                      subgraph_building_method='zero_filling')
    # OOD
    hard_mask = calculate_selected_edges(data, edge_mask, top_k, version=version)

    score_mask_important = gnn_score(all_nodes, data, gnnNets, label, target_id, node_idx=node_idx, edge_mask=hard_mask,
                                     subgraph_building_method='zero_filling')

    return score - score_mask_important

def node_fidelity(data: Data, edge_mask: np.array, top_k: int,
                   gnnNets: torch.nn.Module, label: int,
                   target_id: int = -1, node_idx: Optional[int] = None,
                   undirected=True, version='plus', method='zero_filling'):
    """ return the fidelity score of the subgraph with top_k score edges  """
    if undirected:
        top_k = 2 * top_k
    all_nodes = np.arange(data.x.shape[0]).tolist()
    selected_nodes = calculate_selected_nodes(data, edge_mask, top_k, node_idx)
    score = gnn_score(all_nodes, data, gnnNets, label, target_id, node_idx=node_idx,
                      subgraph_building_method=method)
    # OOD
    if version == 'plus':
        unimportant_nodes = [node for node in all_nodes if node in selected_nodes]
    else:
        unimportant_nodes = [node for node in all_nodes if node not in selected_nodes]

    score_mask_important = gnn_score(unimportant_nodes, data, gnnNets, label, target_id, node_idx=node_idx,
                                     subgraph_building_method=method)

    return score - score_mask_important

def mask_fidelity(data: Data, edge_mask,
                   gnnNets: torch.nn.Module, label, top_k,
                   node_idx: Optional[int] = None,
                   version='plus', undirected=True):
    hard_mask = edge_mask

    logit = gnnNets(data)
    prob = torch.softmax(logit[node_idx], dim=0)
    label_full = torch.argmax(prob).item()
    score = prob[label_full].item()

    if version == 'plus':
        hard_mask = ~hard_mask

    edge_index = data.edge_index.clone().detach()
    data.edge_index = data.edge_index[:, hard_mask]
    logit_mask = gnnNets(data)
    prob_mask = torch.softmax(logit_mask[node_idx], dim=0)
    label_mask = torch.argmax(prob_mask).item()
    score_mask_important = prob_mask[label_full].item()

    '''
    print("OOD\t", (data.y[edge_index[:, hard_mask]] < 0).any(dim=0).sum().item(), '/', (data.y[edge_index] < 0).any(dim=0).sum().item(), version)
    print("TGT\t", (edge_index[:, hard_mask] == node_idx).sum().item(), '/', (edge_index == node_idx).sum().item())
    print(f'{int(hard_mask.float().sum().item())} / {hard_mask.shape[0]} {score:.4f} {score_mask_important:.4f}')
    '''

    return (score - score_mask_important) , (label_full != label_mask)


def top_k_sparsity(edge_mask: np.array, top_k: int, undirected=True):
    """ return the size ratio of the subgraph with top_k score edges"""
    top_k = min(max(top_k, 1), edge_mask.shape[0]-1)

    threshold = float(edge_mask.reshape(-1).sort(descending=True).values[top_k])
    hard_mask = (edge_mask > threshold)

    return 1 - (hard_mask.sum() / hard_mask.shape[0])

def mask_ood_ratio(edge_mask: np.array, top_k: int, edge_index, y):
    hard_mask = edge_mask
    ood_edges = ((y[edge_index[:, hard_mask]] < 0).any(dim=0)).sum()

    if top_k == 0 or hard_mask.sum().item() == 0:
        return 0, 0
    return ood_edges , (hard_mask.sum().item())


def get_graph_build_func(build_method):
    if build_method.lower() == 'zero_filling':
        return graph_build_zero_filling
    elif build_method.lower() == 'split':
        return graph_build_split
    else:
        raise NotImplementedError


def graph_build_zero_filling(X, edge_index, node_mask: np.array):
    """ subgraph building through masking the unselected nodes with zero features """
    node_mask = node_mask.to(X.device)
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index


def graph_build_split(X, edge_index, node_mask: np.array):
    """ subgraph building through spliting the selected nodes from the original graph """
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return X, ret_edge_index


def gnn_score(coalition: list, data: Data, gnnNets, label: int,
              target_id: int = -1, node_idx=None, subgraph_building_method='zero_filling', edge_mask=None) -> torch.Tensor:
    """ the prob of subgraph with selected nodes for required label and target node """
    num_nodes = data.num_nodes
    subgraph_build_func = get_graph_build_func(subgraph_building_method)
    mask = torch.zeros(num_nodes).type(torch.float32).to(data.edge_index.device)
    mask[coalition] = 1.0
    ret_x, ret_edge_index = subgraph_build_func(data.x, data.edge_index, mask)
    mask_data = Data(x=ret_x, edge_index=ret_edge_index, edge_weight=edge_mask)

    # mask_data.edge_weight = edge_mask

    # if edge_mask is not None:
    #     for module in gnnNets.modules():
    #         if isinstance(module, MessagePassing):
    #             module.__explain__ = True
    #             module.__edge_mask__ = edge_mask

    _, logits, probs = gnnNets(mask_data)

    # get the score of predicted class for graph or specific node idx
    node_idx = 0 if node_idx is None else node_idx
    if target_id == -1:
        score = probs[node_idx, label].item()
    else:
        score = probs[node_idx, target_id, label].item()
    return score

