import networkx as nx
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

""" 
The function in this file is largely copied from the orginal PGExplainer codebase. The decision was made to largely copy this file to ensure
that the graph visualization between the original and replicate results would be as similar as possible. Additional comments were added
to clarify the code. 
"""

def plot(graph, edge_weigths, labels, idx, thres_min, thres_snip, dataset, model=None, ood=None, show_idx=0, show=False):
    """
    Function that can plot an explanation (sub)graph and store the image.

    :param graph: graph provided by explainer
    :param edge_weigths: Mask of edge weights provided by explainer
    :param labels: Label of each node required for coloring of nodes
    :param idx: Node index of interesting node
    :param thresh_min: total number of edges
    :param thres_snip: number of top edges
    :param args: Object containing arguments from configuration
    :param gt: Ground Truth
    :param show: flag to show plot made
    """
    # Sort the edge weights in descending order
    sorted_edge_weigths, _ = torch.sort(edge_weigths, descending=True)

    # Set the threshold based on thres_snip
    if thres_snip != -1:
        thres_index = min(thres_snip, edge_weigths.shape[0])  # Limit to the available number of edges
        thres = sorted_edge_weigths[thres_index - 1]  # thres_snip gives the number of edges to highlight
    else:
        thres = torch.min(edge_weigths) - 1  # No edges will be highlighted if thres_snip is -1

    filter_thres_index = min(thres_min, edge_weigths.shape[0])  # Use thres_min to filter all edges
    filter_thres = sorted_edge_weigths[filter_thres_index - 1]  # Minimum threshold for displaying edges

    # Init sets and lists for edges and nodes
    filter_nodes = set()
    filter_edges = []
    thick_edges = []  # To store edges that are above thres_snip (important edges)

    # Select edges to plot
    for i in range(edge_weigths.shape[0]):
        node1, node2 = graph[0][i].item(), graph[1][i].item()

        # Add edges above the filter threshold to the plot
        if edge_weigths[i] >= filter_thres and node1 != node2:
            filter_edges.append((node1, node2))
            filter_nodes.add(node1)
            filter_nodes.add(node2)

            # If the edge is above the thres_snip threshold, mark it as a "thick" edge
            if edge_weigths[i] >= thres:
                thick_edges.append((node1, node2))

    # Create the graph object
    G = nx.Graph()
    G.add_edges_from(filter_edges)

    # Get layout for the nodes
    pos = nx.kamada_kawai_layout(G)

    # Add random noise to node positions to avoid edge overlap
    def add_noise(position, noise_level=0.05):
        return {key: np.array([x + noise_level * (random.random() - 0.5) for x in value]) for key, value in position.items()}

    pos = add_noise(pos, noise_level=0.0)

    # Create label list for node coloring
    label = [int(labels[node]) for node in filter_nodes]

    if dataset == 'syn1':
        colors = ['orange', 'red', 'green', 'blue', 'black', 'brown', 'darkslategray', 'paleturquoise', 'darksalmon',
                  'slategray', 'mediumseagreen', 'mediumblue', 'orchid', ]
    elif dataset == 'syn2':
        colors = ['orange', 'red', 'green', 'blue', 'maroon', 'brown', 'darkslategray', 'paleturquoise',
                  'darksalmon', 'black', 'mediumseagreen', 'mediumblue', 'orchid', ]
    elif dataset == 'syn3':
        colors = ['orange', 'blue', 'black', 'black']
    elif dataset == 'syn4':
        colors = ['orange', 'blue', 'black', 'black', 'blue']

    # Create label-to-node mappings
    label2nodes = [[] for _ in range(np.max(label) + 1)]
    for i, node in enumerate(filter_nodes):
        label2nodes[label[i]].append(node)

    # Draw nodes
    for i, nodes in enumerate(label2nodes):
        if nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors[i % len(colors)], node_size=200)

    # Highlight the target node (idx) in a larger size
    if idx in pos.keys():
        nx.draw_networkx_nodes(G, pos, nodelist=[idx], node_color=colors[labels[idx]], node_size=300)

    # Draw edges (regular edges first, then thicker important edges)
    nx.draw_networkx_edges(G, pos, edgelist=filter_edges, width=3, alpha=0.3, edge_color='grey')  # Regular edges
    nx.draw_networkx_edges(G, pos, edgelist=thick_edges, width=7, alpha=0.8, edge_color='black') # Important edges

    # Remove axis
    plt.axis('off')

    if ood == "":
        ood = 'default'
    # Show or save the plot
    if show:
        plt.show()
    else:
        save_path = f'./qualitative/{dataset}/{ood}/{model}/'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_path}{show_idx}.png')
        plt.clf()
