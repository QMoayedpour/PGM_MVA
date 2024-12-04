import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def get_adjancy(edges, total_nodes):

    adj_matrix = np.zeros((total_nodes, total_nodes), dtype=int)

    for u, v in edges:
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1

    return adj_matrix


def random_permute_adj(adj_matrix, plot=True):
    n = adj_matrix.shape[0]
    idx = np.arange(n)

    np.random.shuffle(idx)

    perm_adj = adj_matrix[np.ix_(idx, idx)]

    if plot:
        plt.imshow(-perm_adj, cmap="gray", extent=(0, perm_adj.shape[1], perm_adj.shape[0], 0))
        plt.axis("off")
        plt.show()

    return (perm_adj, idx)
