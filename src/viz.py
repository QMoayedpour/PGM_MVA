import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def simple_graph_from_adj(adj_matrix, figsize=(6, 6), labels=True):
    """Simple graph from adjency matrix

    adj_matrix (np.array)
    figsize (tuple)
    labels (bool)

    plot graph
    """
    G = nx.from_numpy_array(adj_matrix)
    pos = nx.circular_layout(G)
    plt.figure(figsize=figsize)
    nx.draw(G, pos, with_labels=labels, node_size=500)
    plt.show()


def plot_sbm(G, n_per_class):
    """
    plot the sbm graph

    G (networkx.Graph): graph (sbm)

    n_per_class : list of int
        Taille des blocs (groupes de nœuds).
    """

    color_map = []
    start = 0
    for i, size in enumerate(n_per_class):
        color_map.extend([f"C{i}"] * size)
        start += size

    pos = nx.spring_layout(G)

    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=False, node_size=100, node_color=color_map,
            edge_color="gray", font_size=1)
    plt.show()
