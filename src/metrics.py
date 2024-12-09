import numpy as np
import torch
from networkx.algorithms.community import modularity
import networkx as nx
from collections import defaultdict
import math
from collections import Counter


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


def comp_nmi(graph, tau):
    """
    Calcul le NMI entre les vrais labels et labels du modèle (en partant de tau)
    """
    list_gt = [data["gt"] for _, data in graph.nodes(data=True)]
    return nmi(list_gt, tau.argmax(dim=1).numpy().tolist())


def compute_modularity_from_gt(graph):
    communities = defaultdict(set)
    for node, data in graph.nodes(data=True):
        if 'gt' in data:
            communities[data['gt']].add(node)
        else:
            raise ValueError(f"Le nœud {node} n'a pas d'attribut 'gt'.")

    community_list = list(communities.values())

    return modularity(graph, community_list)


def comp_entropy(labels):
    total = len(labels)
    label_counts = Counter(labels)
    entropy = 0.0
    for count in label_counts.values():
        prob = count / total
        entropy -= prob * math.log(prob, 2)  # Log base 2
    return entropy

def comp_mi(labels_true, labels_pred):
    """Calcule l'information mutuelle entre deux distributions."""
    total = len(labels_true)
    count_true = Counter(labels_true)
    count_pred = Counter(labels_pred)
    joint_counts = Counter(zip(labels_true, labels_pred))
    
    mutual_info = 0.0
    for (label_t, label_p), joint_count in joint_counts.items():
        prob_joint = joint_count / total
        prob_true = count_true[label_t] / total
        prob_pred = count_pred[label_p] / total
        mutual_info += prob_joint * math.log(prob_joint / (prob_true * prob_pred), 2)
    return mutual_info

def nmi(labels_true, labels_pred):
    """Calcule le NMI entre deux partitions."""
    entropy_true = comp_entropy(labels_true)
    entropy_pred = comp_entropy(labels_pred)
    mutual_info = comp_mi(labels_true, labels_pred)

    if entropy_true == 0 or entropy_pred == 0:
        return 0.0
    
    return mutual_info / math.sqrt(entropy_true * entropy_pred)