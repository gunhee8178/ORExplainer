from typing import Optional

import os
import random

from tqdm import tqdm
import networkx as nx

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected

from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree

from .utils import k_hop_subgraph_with_default_whole_graph

EPS = 1e-6

class ORExplainer(nn.Module):
    def __init__(self, model, epochs: int = 20, lr: float = 0.003,
                 top_k: int = 6, args=None, num_classes=4, node_indice = None, num_hops: Optional[int] = None):
        # lr=0.005, 0.003
        super(ORExplainer, self).__init__()
        # 내가 추가한 부분

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.top_k = top_k

        self.__num_hops__ = num_hops
        self.device = model.device

        self.coff_size = args.coff_size
        self.coff_ent = args.coff_ent

        self.t0 = args.temps[0]
        self.t1 = args.temps[1]

        self.dataset = args.dataset
        self.num_classes = num_classes
        self.node_indice = node_indice

        self.args = args

        self.elayers = nn.ModuleList()
        # 바꿀 부분
        self.elayers.append(nn.Sequential(nn.Linear(args.hidden * 3, 64), nn.ReLU()))

        self.elayers.append(nn.Linear(64, 1))



        date_version = (args.date).replace('_', '/')
        if args.ood == "":
            path = f'./checkpoint/{date_version}/{args.dataset}/default/alpha_{args.alpha}/ent_{args.coff_ent}/size_{args.coff_size}/seed_{args.seed}'
        else:
            path = f'./checkpoint/{date_version}/{args.dataset}/{args.ood}/alpha_{args.alpha}/ent_{args.coff_ent}/size_{args.coff_size}/seed_{args.seed}'

        self.ckpt_path = os.path.join(path, f'ORE_generator')
        os.makedirs(os.path.join(path), exist_ok=True)

        self.sub_path = os.path.join(f'./checkpoint/subgraph/{args.dataset}')


    @property
    def num_hops(self):
        """ return the number of layers of GNN model """
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def propagation(self, e, edge_index, edge_mask, prop_layers=1, alpha=0.5, undirected=True):
        '''energy belief propagation, return the energy after propagation'''
        e = e.unsqueeze(1)
        N = e.shape[0]

        if undirected:
            edge_index, edge_mask = to_undirected(edge_index, edge_mask, num_nodes=N, reduce='mean')
        row, col = edge_index
        # d = scatter(src=edge_mask, index=col, dim_size=N, reduce='add')
        d = degree(col, N).float()

        d_norm = 1. / d[col]
        value = edge_mask * d_norm

        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        return e.squeeze(1)

    def __loss__(self, prob, ori_pred, edge_mask):
        """
        the pred loss encourages the masked graph with higher probability,
        the size loss encourage small size edge mask,
        the entropy loss encourage the mask to be continuous.
        """
        # print(prob.device, ori_pred.device, edge_mask.device)
        prob = prob.squeeze()
        ori_pred = ori_pred.squeeze()
        ori_pred = ori_pred.to(self.device)
        logit = F.cross_entropy(prob, ori_pred).to(self.device)

        # print(prob, ori_pred)
        pred_loss = logit

        size_loss = self.coff_size * torch.sum(edge_mask)

        # entropy
        # Nan to zero

        # Nan in edge_mask print
        edge_mask = edge_mask * 0.99 + 5e-7
        mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)

        mask_ent_loss = self.coff_ent * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss

        return loss, pred_loss, size_loss, mask_ent_loss

    def ood_loss(self, logits, edge_index, edge_mask, node_idx, temp=1.0):
        logits = logits.to(self.device)
        edge_index = edge_index.to(self.device)

        # prob = torch.softmax(logits, dim=-1)
        energy = temp * -torch.logsumexp(logits/temp, dim=-1).squeeze()

        if torch.isnan(energy).any():
            print(energy, node_idx)

        energy = self.propagation(energy, edge_index, edge_mask, prop_layers=self.num_hops, alpha=0.5)

        return energy[node_idx]

    def concrete_sample(self, sampling_weights, beta=1.0, training=True):
        """ Sample from the instantiation of concrete distribution when training
        \epsilon \sim  U(0,1), \hat{e}_{ij} = \sigma (\frac{\log \epsilon-\log (1-\epsilon)+\omega_{i j}}{\tau})
        """
        if training:
            bias = self.args.sample_bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size(), device=self.device) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / beta
            gate_inputs =  torch.sigmoid(gate_inputs)

        else:
            gate_inputs = sampling_weights.sigmoid()

        return gate_inputs

    def forward(self, inputs, training=None, undirected=False):
        x, embed, edge_index, node_id, tmp = inputs
        nodesize = embed.shape[0]

        embed = embed.to(self.device)
        edge_index = edge_index.to(self.device)

        col, row = edge_index
        f1 = embed[col]
        f2 = embed[row]
        node_embed = embed[node_id].repeat(col.shape[0], 1)

        # using the node embedding to calculate the edge weight
        # print(f1.shape, f2.shape, node_embed.shape)
        h = torch.cat([f1, f2, node_embed], dim=-1)
        for elayer in self.elayers:
            h = elayer(h)

        values = h.reshape(-1)
        values = self.concrete_sample(values, beta=tmp, training=training)
        if undirected:
            _, values = to_undirected(edge_index, values, num_nodes=nodesize, reduce='mean')

        # the model prediction with edge mask
        data = Data(x=x, edge_index=edge_index, edge_weight=values)
        data = data.to(self.device)

        emb, logits = self.model(data)
        return logits.squeeze(), emb.squeeze(), values

    def get_model_output(self, x, edge_index, edge_mask=None, **kwargs):
        """ return the model outputs with or without (w/wo) edge mask  """
        self.model.eval()

        with torch.no_grad():
            # data = Data(x=x, edge_index=edge_index)
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_mask)
            data = data.to(self.device)
            outputs = self.model(data)
        return outputs


    def get_explanation_network(self, dataset, version=None, is_graph_classification=True):
        if version is None:
            version = self.epochs
        # print(self.ckpt_path+f'_{version}.pth')
        if os.path.isfile(self.ckpt_path+f'_{version}.pth'):
            print("fetch network parameters from the saved files")
            state_dict = torch.load(self.ckpt_path+f'_{version}.pth')
            self.elayers.load_state_dict(state_dict)
            self.to(self.device)
        elif not is_graph_classification:
            self.to(self.device)
            self.train_NC_explanation_network(dataset)

    def eval_probs(self, x: torch.Tensor, edge_index: torch.Tensor,
                   edge_mask: torch.Tensor=None, **kwargs) -> None:
        _, prob = self.get_model_output(x, edge_index, edge_mask=edge_mask)
        return prob.squeeze()

    def explain_edge_mask(self, x, edge_index, emb, node_idx, **kwargs):

        with torch.no_grad():
            _, prob = self.get_model_output(x, edge_index)
            _, _, edge_mask = self.forward((x, emb, edge_index, node_idx, 1.0), training=False)
        return edge_mask

    def get_subgraph(self, node_idx, x, edge_index, y, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        y = y[subset]

        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                kwargs[key] = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                kwargs[key] = item[edge_mask]

        return x, edge_index, y, subset, kwargs

    def train_NC_explanation_network(self, data):
        if self.node_indice is not None:
            dataset_indices = self.node_indice
        else:
            dataset_indices = range(300, 700, 5)

        optimizer = Adam(self.elayers.parameters(), lr=self.lr, weight_decay=5e-4)

        # collect the embedding of nodes
        x_dict = {}
        y_dict = {}
        edge_index_dict = {}
        node_idx_dict = {}
        subset_dict = {}
        pred_dict = {}

        with torch.no_grad():
            self.model.eval()
            emb, full_logits = self.get_model_output(data.x, data.edge_index)

            for gid in tqdm(dataset_indices):
                x, edge_index, y, subset, _ = \
                    self.get_subgraph(node_idx=gid, x=data.x,
                                      edge_index=data.edge_index, y=data.y)
                _, logits = self.get_model_output(x, edge_index)

                x_dict[gid] = x.cpu()
                y_dict[gid] = y.cpu()
                edge_index_dict[gid] = edge_index.cpu()
                node_idx_dict[gid] = int(torch.where(subset == gid)[0])
                subset_dict[gid] = subset.cpu()
                pred_dict[gid] = logits.argmax(dim=-1)[node_idx_dict[gid]].cpu()

            data.detach().cpu()
        # train the explanation network

        for epoch in range(self.epochs):
            loss_pge_total = 0.0
            loss_ood_total = 0.0
            loss_total = 0.0

            optimizer.zero_grad()
            tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))

            self.elayers.to(self.device)
            self.elayers.train()
            for gid in tqdm(dataset_indices):
                if edge_index_dict[gid].shape[1] == 0:
                    continue
                subset = subset_dict[gid].to(self.device)

                pred, x3, edge_mask = self.forward(
                    (x_dict[gid], emb[subset], edge_index_dict[gid], node_idx_dict[gid], tmp)
                    , training=True)

                pge_loss, pred_loss, size_loss, ent_loss = self.__loss__(pred[node_idx_dict[gid]], pred_dict[gid], edge_mask)
                if self.args.alpha == 0:
                    loss_ood = 0.0

                loss_ood = self.ood_loss(full_logits[subset], edge_index_dict[gid], edge_mask, node_idx_dict[gid])

                loss_pge_total += pred_loss
                loss_ood_total += (loss_ood)

                loss = pge_loss + (loss_ood  * self.args.alpha)
                loss_total += loss
                loss.backward()

            optimizer.step()
            print(f'Epoch: {epoch} | Loss: {loss_total:.4f} | PGE Loss: {loss_pge_total:.4f} | OOD Loss: {loss_ood_total:.4f} * {self.args.alpha:.2f}')
            # if epoch % 10 == 9:
            #     torch.save(self.elayers.cpu().state_dict(), self.ckpt_path+f'_{epoch+1}.pth')
            #     self.elayers.to(self.device)
        torch.save(self.elayers.cpu().state_dict(), self.ckpt_path + f'_{self.epochs}.pth')
        self.elayers.to(self.device)

    def eval_node_probs(self, node_idx: int, x: torch.Tensor,
                        edge_index: torch.Tensor, edge_mask: torch.Tensor, **kwargs):
        probs = self.eval_probs(x=x, edge_index=edge_index,
                                edge_mask=edge_mask, **kwargs)
        return probs[node_idx].squeeze()

    def get_node_prediction(self, node_idx: int, x: torch.Tensor, edge_index: torch.Tensor, **kwargs):
        _, output = self.get_model_output(x, edge_index, edge_mask=None, **kwargs)
        return output[node_idx].argmax(dim=-1)

    def __repr__(self):
        return f'{self.__class__.__name__}()'
