import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def assign_colors_to_classes(graph, class_name):
    classes = {graph.nodes[node][class_name] for node in graph.nodes}

    colormap = cm.get_cmap('tab20', len(classes))
    class_to_color = {cls: colormap(i) for i, cls in enumerate(classes)}

    node_colors = [class_to_color[graph.nodes[node][class_name]] for node in graph.nodes]

    return class_to_color, node_colors


def plot_class_graph(graph, class_name):
    class_to_color, node_colors = assign_colors_to_classes(graph, 'gt')

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_color=node_colors,
        edge_color="gray",
        node_size=50,
        font_size=10
    )


def plot_adjacency_matrix_by_class(graph, class_name="gt"):
    """
    Affiche la matrice d'adjacence avec les nœuds regroupés par classe.
    Contrairement a l'autre fonction on utilise les attributs du graph
    """

    node_classes = {node: graph.nodes[node][class_name] for node in graph.nodes}
    sorted_nodes = sorted(graph.nodes, key=lambda n: node_classes[n])
    classes = [node_classes[node] for node in sorted_nodes]

    adjacency_matrix = nx.to_numpy_array(graph, nodelist=sorted_nodes)
    
    class_boundaries = []
    current_class = classes[0]
    for i, cls in enumerate(classes):
        if cls != current_class:
            class_boundaries.append(i)
            current_class = cls
    class_boundaries.append(len(classes))

    plt.figure(figsize=(8, 8))
    plt.imshow(adjacency_matrix, cmap='Blues', interpolation='none')

    for boundary in class_boundaries:
        plt.axhline(boundary - 0.5, color='red', linewidth=1.5)
        plt.axvline(boundary - 0.5, color='red', linewidth=1.5)
    
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.tight_layout()
    plt.show()