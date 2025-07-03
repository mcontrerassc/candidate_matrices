from votekit.utils import mentions, ballots_by_first_cand
from votekit.cleaning import remove_and_condense
from votekit.pref_profile import PreferenceProfile
from votekit.utils import mentions
from votekit.matrices import matrix_heatmap, boost_matrix
import matplotlib.pyplot as plt
import numpy as np
from cdlib import algorithms
import networkx as nx


def create_graph_louvain_bm(boost: np.ndarray):
    n = boost.shape[0]
    # build a graph
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            weight = (boost[i, j] + boost[j, i])/2
            if weight > 0:  # only positive weights -- eeek algos.louvain doesn't work with negs?  , TODO think ab this later
                G.add_edge(i, j, weight=weight)
    return G



def louvain_partition_from_graph(G: nx.Graph, profile: PreferenceProfile, resolution: float, randomize: bool):
    mentions_dict = mentions(profile)
    all_cands_sorted_by_mentions = sorted(profile.candidates, reverse=True, key = lambda x: mentions_dict[x])
    partition = algorithms.louvain(G, weight='weight', resolution = resolution, randomize = randomize)
    clusters = {}
    for node, clust_ids in partition.to_node_community_map().items():
        clust_id = clust_ids[0] 
        clusters.setdefault(clust_id, []).append(node)

    # from chris' code: sort by mentions within cluster
    clusters = {cluster_id: sorted(cluster_list, reverse = True, 
                key = lambda x: mentions_dict[all_cands_sorted_by_mentions[x]]) for cluster_id, cluster_list in clusters.items()}
    partition = []
    for cluster_id, cluster_list in clusters.items():
        list_cands = []
        for node in cluster_list:
            list_cands.append(all_cands_sorted_by_mentions[node])
        partition.append(list_cands)
    return partition