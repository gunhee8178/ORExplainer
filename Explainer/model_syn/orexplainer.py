from typing import Optional

import os
import random

from tqdm import tqdm
import networkx as nx
from sklearn.decomposition import PCA

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected

from torch_sparse import SparseTensor
from torch_geometric.utils import degree

from .utils import k_hop_subgraph_with_default_whole_graph

EPS = 1e-6

class ORExplainer(nn.Module):
    def __init__(self, model, epochs: int = 20, lr: float = 0.003,
                 top_k: int = 6, args=None, num_classes=4, node_indice = None, num_hops: Optional[int] = None):

        super(ORExplainer, self).__init__()

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.top_k = top_k

        self.__num_hops__ = num_hops
        self.device = next(model.parameters()).device

        self.coff_size = args.coff_size
        self.coff_ent = args.coff_ent

        self.t0 = args.temps[0]
        self.t1 = args.temps[1]

        self.dataset = args.dataset
        self.num_classes = num_classes
        self.node_indice = node_indice

        self.args = args
        in_channels = (args.hidden) * self.num_hops * 3

        self.elayers = nn.ModuleList()
        self.elayers.append(nn.Sequential(nn.Linear(in_channels, 64), nn.ReLU()))
        self.elayers.append(nn.Linear(64, 1))
        self.elayers.to(self.device)

        model_name = 'OR'

        date_version = (args.date).replace('_', '/')
        if args.gamma == 0:
            args.temp = 1.0
        path = f'./Explainer/check_point/{date_version}/{model_name}/{args.dataset}/{args.ood}/gamma_{args.gamma}/temp_{args.temp}/lamda_{args.lamda}/ent_{args.coff_ent}/size_{args.coff_size}/seed_{args.seed}'
        self.ckpt_path = os.path.join(path, f'OR_generator')
        os.makedirs(os.path.join(path), exist_ok=True)

        self.sub_path = os.path.join(f'./Explainer/checkpoint/subgraph/{args.dataset}')

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

    # Weighted Energy Propagation
    def propagation(self, energy, edge_index, edge_mask, gamma=0.5, K=1, undirected=False):
        num_nodes = energy.size(0)
        row, col = edge_index

        # degree
        deg = degree(col, num_nodes, dtype=energy.dtype)
        deg_inv_sqrt = deg.pow(-1.0)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = edge_mask * deg_inv_sqrt[col]


        adj = SparseTensor(row=col, col=row, value=norm, sparse_sizes=(num_nodes, num_nodes))

        e = energy.view(-1, 1)
        for _ in range(K):
            neigh = adj @ e  # [N,1]
            e = gamma * e + (1 - gamma) * neigh
        return e.squeeze(1)

    def ene_loss(self, energy, edge_index, edge_mask, node_idx):
        edge_index = edge_index.to(self.device)

        if torch.isnan(energy).any():
            print(energy, node_idx)

        propagated_energy = self.propagation(energy, edge_index, edge_mask, K=self.num_hops, gamma=self.args.lamda)

        return propagated_energy[node_idx]

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
        ce_loss = F.cross_entropy(prob, ori_pred).to(self.device)

        size_loss = self.coff_size * torch.sum(edge_mask)

        edge_mask = edge_mask * 0.99 + 5e-7
        mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)

        mask_ent_loss = self.coff_ent * torch.mean(mask_ent)
        loss = ce_loss + size_loss + mask_ent_loss

        return loss, ce_loss, size_loss, mask_ent_loss

    def concrete_sample(self, sampling_weights, beta=1.0, training=True):
        """ Sample from the instantiation of concrete distribution when training
        \epsilon \sim  U(0,1), \hat{e}_{ij} = \sigma (\frac{\log \epsilon-\log (1-\epsilon)+\omega_{i j}}{\tau})
        """
        if training:
            bias = self.args.sample_bias + 0.0001
            eps = torch.rand(sampling_weights.size(), device=self.device) * (1 - 2 * bias) + bias
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / beta
            gate_inputs =  torch.sigmoid(gate_inputs)


        else:
            gate_inputs = sampling_weights.sigmoid()

        return gate_inputs

    def forward(self, inputs, training=None, undirected=True):
        x, embed, edge_index, node_id, tmp = inputs
        nodesize = embed.shape[0]

        x = x.to(self.device)
        embed = embed.to(self.device)
        edge_index = edge_index.to(self.device)

        col, row = edge_index
        f1 = embed[col]
        f2 = embed[row]
        node_embed = embed[node_id].repeat(col.shape[0], 1)

        h = torch.cat([f1, f2, node_embed], dim=-1)
        for elayer in self.elayers:
            h = elayer(h)

        values = h.reshape(-1)
        values = self.concrete_sample(values, beta=tmp, training=training)

        return values

    def get_model_output(self, x, edge_index, edge_mask=None, **kwargs):
        """ return the model outputs with or without (w/wo) edge mask  """
        self.model.eval()

        with torch.no_grad():
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_mask)
            data = data.to(self.device)
            embedding = self.model.embed(data.x, data.edge_index, data.edge_weight)
            logits = self.model.lin(embedding)
        return embedding, logits


    def get_explanation_network(self, dataset, energy=None):
        if os.path.isfile(self.ckpt_path+f'_{self.args.epochs}.pth'):
            print("fetch network parameters from the saved files")
            state_dict = torch.load(self.ckpt_path+f'_{self.args.epochs}.pth')
            self.elayers.load_state_dict(state_dict)
            self.to(self.device)
        else:
            self.to(self.device)
            self.train_NC_explanation_network(dataset, energy)

    def eval_probs(self, x: torch.Tensor, edge_index: torch.Tensor,
                   edge_mask: torch.Tensor=None, **kwargs) -> None:
        _, prob = self.get_model_output(x, edge_index, edge_mask=edge_mask)
        return prob.squeeze()

    def explain_edge_mask(self, x, edge_index, emb, node_idx, **kwargs):
        with torch.no_grad():
            edge_mask = self.forward((x, emb, edge_index, node_idx, 1.0), training=False)
        return edge_mask

    def get_subgraph(self, node_idx, x, edge_index, y, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        '''
        inv = torch.empty(num_nodes, dtype=torch.long).to(self.device)
        inv[subset] = torch.arange(subset.size(0), dtype=torch.long).to(self.device)
        edge_index = inv[edge_index.to(self.device)]
        '''

        x = x[subset]
        y = y[subset]

        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                kwargs[key] = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                kwargs[key] = item[edge_mask]

        return x, edge_index, y, subset, kwargs

    def train_NC_explanation_network(self, data, energy):
        if self.node_indice is not None:
            dataset_indices = self.node_indice
        else:
            dataset_indices = range(300, 700, 5)

        optimizer = Adam(self.elayers.parameters(), lr=self.lr)
        x_dict = {}
        y_dict = {}
        edge_index_dict = {}
        node_idx_dict = {}
        subset_dict = {}
        pred_dict = {}
        embedding_dic = {}

        with torch.no_grad():
            self.model.eval()
            for gid in tqdm(dataset_indices):
                x, edge_index, y, subset, _ = \
                    self.get_subgraph(node_idx=gid, x=data.x,
                                      edge_index=data.edge_index, y=data.y)

                embeddings, logits = self.get_model_output(x, edge_index)
                logits.argmax(dim=-1)

                x_dict[gid] = x.cpu()
                y_dict[gid] = y.cpu()
                edge_index_dict[gid] = edge_index.cpu()
                node_idx_dict[gid] = int(torch.where(subset == gid)[0])
                subset_dict[gid] = subset.cpu()
                pred_dict[gid] = logits.argmax(dim=-1)[node_idx_dict[gid]].cpu()
                embedding_dic[gid] = embeddings.detach().cpu()
            data.detach()
        # train the explanation network

        for epoch in range(self.epochs):
            loss_CE_total = 0.0
            loss_ene_total = 0.0
            loss_total = 0.0

            optimizer.zero_grad()
            tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))

            self.elayers.to(self.device)
            self.elayers.train()
            for gid in tqdm(dataset_indices):
                # Skip Isolated nodes
                if edge_index_dict[gid].shape[1] == 0:
                    continue
                # if pred_dict[gid] !=data.y[gid].cpu():
                #     continue

                subset = subset_dict[gid].to(self.device)

                edge_mask = self.forward(
                    (x_dict[gid], embedding_dic[gid], edge_index_dict[gid], node_idx_dict[gid], tmp)
                    , training=True)

                node_size = x_dict[gid].shape[0]
                edge_index = edge_index_dict[gid].to(self.device)
                edge_index, edge_mask = to_undirected(edge_index, edge_attr=edge_mask, num_nodes=node_size, reduce='mean')

                # the model prediction with edge mask
                sub_data = Data(x=x_dict[gid], edge_index=edge_index, edge_weight=edge_mask)
                sub_data = sub_data.to(self.device)
                pred = self.model(sub_data)

                exp_loss, CE_loss, size_loss, ent_loss = self.__loss__(pred[node_idx_dict[gid]], pred_dict[gid], edge_mask)

                if self.args.gamma > 0:
                    loss_ene = self.ene_loss(energy[subset], edge_index, edge_mask, node_idx_dict[gid])
                else:
                    loss_ene = 0.0

                loss_CE_total += CE_loss
                loss_ene_total += loss_ene
                loss = exp_loss + (loss_ene  * self.args.gamma) # +_+
                loss_total += loss
                loss.backward()

            optimizer.step()
            print(f'Epoch: {epoch} | Loss: {loss_total/len(dataset_indices):.4f} | CE Loss: {loss_CE_total/len(dataset_indices):.4f} | ENE Loss: {loss_ene_total/len(dataset_indices):.4f} * {self.args.gamma:.2f}')

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
