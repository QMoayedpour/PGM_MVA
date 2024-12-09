import numpy as np
import torch
from networkx.algorithms.community import modularity
import networkx as nx
from collections import defaultdict


def predictions_to_partition(predictions, graph):
    """
    Convertit les prédictions de labels en une partition sous forme de liste d'ensembles.
    on prends en entrée les nodes du graph car leur label sont pas toujours ceux décris dans la liste..
    """
    nodes = list(graph.nodes)
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.tolist()

    community_dict = defaultdict(set)
    for node, label in enumerate(predictions):
        community_dict[label].add(nodes[node])

    partition = list(community_dict.values())
    return partition


def comp_relative_modularity(graph, tau):
    """
    Calcul la modularité relative a la modularité du graph avec les vraies classes
    """
    preds = predictions_to_partition(tau.argmax(dim=1), graph)

    modularity_preds = modularity(graph, preds)

    modularity_gt = compute_modularity_from_gt(graph)

    if modularity_gt != 0:
        return modularity_preds/modularity_gt

    else:
        raise ValueError("Modularity of the graph is =0")


def compute_modularity_from_gt(graph):
    communities = defaultdict(set)
    for node, data in graph.nodes(data=True):
        if 'gt' in data:
            communities[data['gt']].add(node)
        else:
            raise ValueError(f"Le nœud {node} n'a pas d'attribut 'gt'.")

    community_list = list(communities.values())

    return modularity(graph, community_list)
