import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


def simple_graph_from_adj(adj_matrix, figsize=(6,6), labels=True):
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

def generate_sbm(n_per_class, probs):
    """
    Generate a graph from SBM

    n_per_class (np.array): array containing the number of edges by classes 

    probs (np.array): symmetric matrix containing the probabilities (for instance, index i,j represent
    the proba of a edge in class i to be linked with an edge in class j) 

    G : networkx.Graph
    """

    if len(n_per_class) != len(probs):
        raise ValueError("number of classes doesnt maths probability matrix")

    if not all(len(row) == len(probs) for row in probs):
        raise ValueError("probs must be squared (and symmetric)")

    G = nx.stochastic_block_model(n_per_class, probs)

    return G


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
    nx.draw(G, pos, with_labels=False, node_size=100, node_color=color_map, edge_color="gray", font_size=1)
    plt.show()


def get_adjancy(edges, total_nodes):

    adj_matrix = np.zeros((total_nodes, total_nodes), dtype=int)

    for u, v in edges:
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1

    return adj_matrix  


class MySbmFromScratch(object):
    def __init__(self):
        self.G = None  
        self.edges = None
        self.nodes = None   

    def generate_graph(self, N=200, K=3, n_classes=None, probs=None, _lambda=1, multiplier=1, alpha=1):

        if n_classes == None and probs == None:
            self.edges, self.nodes = self._generate_from_nk(N, K, _lambda=_lambda, alpha=alpha, multiplier=multiplier)
        else:
            self.edges, self.nodes = self._generate_from_probs(n_classes, probs)
            self.n_classes = n_classes
        self.adj = get_adjancy(self.edges, len(self.nodes))

        self.G = nx.from_numpy_array(self.adj)

    def plot_graph(self):
        plot_sbm(self.G, self.n_classes)    

    def _generate_from_nk(self, N, K, alpha=1., _lambda=5, multiplier=1):
        """If we want to generate from the number of edges (N) and the number of classes (K)
        N (int): number of edges
        K (int): number of classes
        alpha (float): parameter of dirichlet
        _lambda (float): parameter of exponential

        return edges, nodes
        """

        class_probs = np.random.dirichlet([alpha] * K, size=1).flatten()
        self.n_classes = [int(N*class_probs[i]) for i in range(K)]

        probs = np.random.rand(K, K)**2

        probs = (probs @ probs.T)
        probs = probs/np.sum(probs) * multiplier + np.eye(K)*_lambda
        self.probs = (np.clip(probs, 0, 1))

        return self._generate_from_probs(self.n_classes, self.probs)

    def plot_adj(self):
        cumulative_indices = np.cumsum(self.n_classes)

        plt.imshow(-self.adj, cmap="gray", extent=(0, self.adj.shape[1], self.adj.shape[0], 0))
        plt.axis("off")
        for index in cumulative_indices:
            plt.axhline(y=index, color='red', linestyle='-', linewidth=1)
            plt.axvline(x=index, color='red', linestyle='-', linewidth=1)

        plt.show()

    def _generate_from_probs(self, n_classes, probs):
        """
        generate edges and nodes from :
        n_classes (np.array): array (or list) of the number of edges per "class"
        probs (np.array): matrix (symmetric) of probability of a edge from class i
        to be connected to an edge from class j

        return edges, nodes
        """
        total_nodes = sum(n_classes)

        edges = []

        nodes = []
        start = 0
        for i, size in enumerate(n_classes):
            nodes.extend([i] * size)
            start += size

        for i in range(total_nodes):
            for j in range(i + 1, total_nodes):
                block_i = nodes[i]

                block_j = nodes[j]

                prob = probs[block_i][block_j]

                if random.random() < prob:
                    edges.append((i, j))

        return edges, nodes
