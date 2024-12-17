import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def get_adjancy(edges, total_nodes):
    """Renvoi la matrice d'agence a partir des edges et nodes
    """

    adj_matrix = np.zeros((total_nodes, total_nodes), dtype=int)

    for u, v in edges:
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1

    return adj_matrix


def random_permute_adj(adj_matrix, plot=True):
    """Opère une permutation aléatoire de la matrice d'agence car pour nos graphs
    générées, la matrice est déjà ordonnée
    """
    n = adj_matrix.shape[0]
    idx = np.arange(n)

    np.random.shuffle(idx)

    perm_adj = adj_matrix[np.ix_(idx, idx)]

    if plot:
        plt.imshow(-perm_adj, cmap="gray", extent=(0, perm_adj.shape[1], perm_adj.shape[0], 0))
        plt.axis("off")
        plt.show()

    return (perm_adj, idx)


def convert_to_ranks(array):

    unique_sorted = sorted(set(array))

    value_to_rank = {value: rank for rank, value in enumerate(unique_sorted)}

    return [value_to_rank[value] for value in array]


def remove_isolated(adjacency_matrix):
    connected = np.any(adjacency_matrix, axis=0) | np.any(adjacency_matrix, axis=1)

    filtered_matrix = adjacency_matrix[np.ix_(connected, connected)]

    isolated_idx = np.where(~connected)[0]

    return filtered_matrix, isolated_idx
