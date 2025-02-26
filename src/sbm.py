import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from .utils import get_adjancy, remove_isolated
from .viz import plot_sbm


def generate_sbm(n_per_class, probs):
    """
    Generate a graph from SBM

    n_per_class (np.array): array containing the number of edges by classes

    probs (np.array): symmetric matrix containing the probabilities (for instance,
    index i,j represent the proba of a edge in class i to be linked with an edge in class j)

    G : networkx.Graph
    """

    if len(n_per_class) != len(probs):
        raise ValueError("number of classes doesnt maths probability matrix")

    if not all(len(row) == len(probs) for row in probs):
        raise ValueError("probs must be squared (and symmetric)")

    G = nx.stochastic_block_model(n_per_class, probs)

    return G


class MySbmFromScratch(object):
    def __init__(self):
        self.G = None
        self.edges = None
        self.nodes = None

    def generate_graph(self, N=200, K=3, n_classes=None, probs=None,
                       _lambda=1, multiplier=1, alpha=1):
        """La fonction prends plusieurs entrées possibles:
        Soit, on donne directement les parametres du modèles (ie le nombre de points par classes et la 
        matrice pi)
        Soit on génère aléatoirement un graph: lambda controle les connexions intra clusters,
        multiplier l'intensité des probabilités de la matrice pi, et alpha la répartition des points par 
        classes, (Parametre d'une loi dirichlet)
        """

        if n_classes is None and probs is None:
            self.edges, self.nodes = self._generate_from_nk(N, K, _lambda=_lambda, alpha=alpha,
                                                            multiplier=multiplier)
        else:
            self.edges, self.nodes = self._generate_from_probs(n_classes, probs)
            self.n_classes = n_classes

        adj = get_adjancy(self.edges, len(self.nodes))
        self.adj, isolated_idx = remove_isolated(adj)

        self.G = nx.from_numpy_array(self.adj)

        labels = [[i]*j for i, j in enumerate(self.n_classes)]
        labels = [item for sublist in labels for item in sublist]

        labels = [label for idx, label in enumerate(labels) if idx not in isolated_idx]
        self.n_classes = [labels.count(i) for i in range(len(self.n_classes))]
        node_values = dict(zip(self.G.nodes, labels))
        nx.set_node_attributes(self.G, node_values, name="gt")

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

    def _init_graph_blocks(self, G):
        """
        Initialise l'attribut 'block' de chaque nœud à 0.

        Paramètres:
        G (networkx.Graph) : Le graphe sur lequel appliquer l'attribut.

        Retourne:
        dict : Un dictionnaire contenant les nœuds et leurs attributs.
        """
        nx.set_node_attributes(G, 0, 'gt')

        self.G = {node: {'gt': data['gt']} for node, data in G.nodes(data=True)}
