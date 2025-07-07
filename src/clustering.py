from votekit.utils import mentions, ballots_by_first_cand
from votekit.cleaning import remove_and_condense
from votekit.pref_profile import PreferenceProfile
from votekit.utils import mentions
from votekit.matrices import matrix_heatmap, boost_matrix
import matplotlib.pyplot as plt
import numpy as np
from cdlib import algorithms, evaluation
import networkx as nx
from src.tools import enumerate_bipartitions
from src.scores import distance_to_slate_across_profile
from tqdm import tqdm
import jsonlines as jl


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
    modularity_score = evaluation.newman_girvan_modularity(G, partition, weight='weight').score
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
    return partition, modularity_score


def find_distance_to_slate_optimal_bipartition(profile: PreferenceProfile):
    ''''
        TODO: add some parameters to indicate that this method is
        writing to current directory

        computes the distance to slate scores for all possible
            partitions of the given profile, wrt the ballots in the given
            profile. 
        returns: list of list containing each partition and its
        distance to slate scores.
    '''

    all_bipartitions = enumerate_bipartitions(profile.candidates)
    #distance_to_slate_scores =
    #[distance_to_slate_across_profile(profile, bipart) for bipart in
    #all_bipartitions]
    

    distance_to_slate_scores = []

    with jl.open("bipart_results_as_gen_2.jsonl", "w") as writer:
        for bipart in tqdm(all_bipartitions):

            dts_score = distance_to_slate_across_profile(profile, bipart) 
            distance_to_slate_scores.append(dts_score)
            writer.write({
                "bipart": [list(part) for part in bipart],
                "distance_to_slate": dts_score
            })

    parts_and_score = list(zip(all_bipartitions, distance_to_slate_scores)) 
    return sorted(parts_and_score, key=lambda entry: entry[1])

