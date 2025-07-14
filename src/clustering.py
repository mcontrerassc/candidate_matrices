from votekit.utils import mentions, ballots_by_first_cand
from votekit.cleaning import remove_and_condense
from votekit.pref_profile import PreferenceProfile
from votekit.utils import mentions
from votekit.matrices import matrix_heatmap, boost_matrix
import matplotlib.pyplot as plt
import numpy as np
from cdlib import algorithms, evaluation
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
    # DO NOT SORT OR THE VIZ WILL GET *ANGRY* (you'll get a nonsensical matrix) D:
    #all_cands_sorted_by_mentions = sorted(profile.candidates, reverse=True, key = lambda x: mentions_dict[x])
    partition = algorithms.louvain(G, weight='weight', resolution = resolution, randomize = randomize)
    modularity_score = evaluation.newman_girvan_modularity(G, partition, weight='weight').score
    clusters = {}
    for node, clust_ids in partition.to_node_community_map().items():
        clust_id = clust_ids[0] 
        clusters.setdefault(clust_id, []).append(node)

    # from chris' code: sort by mentions within cluster
    clusters = {cluster_id: sorted(cluster_list, reverse = True, 
                key = lambda x: mentions_dict[profile.candidates[x]]) for cluster_id, cluster_list in clusters.items()}
    partition = []
    for cluster_id, cluster_list in clusters.items():
        list_cands = []
        for node in cluster_list:
            list_cands.append(profile.candidates[node])
        partition.append(list_cands)
    return partition, modularity_score

def split_matrix(A):
    A_pos = np.where(A > 0, A, 0)
    A_neg = np.where(A < 0, A, 0)
    return A_pos, A_neg

def make_modularity_matrix(A):
    kout = A.sum(axis=1)
    kin = A.sum(axis=0)
    m = A.sum()
    expected = np.outer(kout, kin) / m if m > 0 else np.zeros_like(A)
    return A - expected

def modularity_from_B(B, assignment, pm='pos', m=None):
    n = len(assignment)
    Q = 0.0
    for i in range(n):
        for j in range(n):
            same_group = assignment[i] == assignment[j]
            if (pm == 'pos' and same_group) or (pm == 'neg' and not same_group):
                Q += B[i, j]
    if m is None:
        m = B.sum()
    return Q / m if m != 0 else 0.0

import numpy as np
from collections import defaultdict

def louvain_directed(A, pm='pos', max_iter=100):
    n = A.shape[0]
    assignment = np.arange(n)  # Start with each node in its own group

    # Step 1: separate positive and negative components
    A_pos, A_neg = split_matrix(A)
    A_used = A_pos if pm == 'pos' else -A_neg  # use positive weights throughout
    m = A_used.sum()

    if m == 0:
        return assignment  # trivial partition

    # Step 2: precompute modularity matrix
    B = make_modularity_matrix(A_used)

    # Step 3: run local improvement loop
    for _ in range(max_iter):
        changed = False

        for node in range(n):
            current_group = assignment[node]

            # compute modularity gain for moving node to each existing group
            group_scores = defaultdict(float)
            for other in range(n):
                if node == other:
                    continue
                proposed_group = assignment[other]
                assignment[node] = proposed_group  # temporarily change
                delta_Q = modularity_from_B(B, assignment, pm=pm, m=m)
                group_scores[proposed_group] = delta_Q
            # restore
            assignment[node] = current_group

            # find best group
            best_group = max(group_scores, key=group_scores.get, default=current_group)
            if group_scores[best_group] > modularity_from_B(B, assignment, pm=pm, m=m):
                assignment[node] = best_group
                changed = True

        if not changed:
            break

    return assignment

import numpy as np
from collections import defaultdict

def louvain_directed_hybrid(A, max_iter=100):
    n = A.shape[0]
    assignment = np.arange(n)  # each node starts in its own group

    # Step 1: Split into positive and negative parts
    A_pos, A_neg = split_matrix(A)
    A_neg = -A_neg  # flip sign so entries are positive

    m_pos = A_pos.sum()
    m_neg = A_neg.sum()

    if m_pos + m_neg == 0:
        return assignment

    # Step 2: Build modularity matrices
    B_pos = make_modularity_matrix(A_pos)
    B_neg = make_modularity_matrix(A_neg)

    # Step 3: Local modularity maximization loop
    for _ in range(max_iter):
        changed = False

        for node in range(n):
            current_group = assignment[node]

            # Compute current hybrid modularity
            Q_current = (
                modularity_from_B(B_pos, assignment, pm='pos', m=m_pos) -
                modularity_from_B(B_neg, assignment, pm='neg', m=m_neg)
            )

            # Try placing node in each other group
            group_scores = {}
            unique_groups = set(assignment) - {current_group}

            for g in unique_groups:
                assignment[node] = g
                Q_new = (
                    modularity_from_B(B_pos, assignment, pm='pos', m=m_pos) -
                    modularity_from_B(B_neg, assignment, pm='neg', m=m_neg)
                )
                group_scores[g] = Q_new

            # Restore original group
            assignment[node] = current_group

            # Choose best move
            if group_scores:
                best_group = max(group_scores, key=group_scores.get)
                if group_scores[best_group] > Q_current:
                    assignment[node] = best_group
                    changed = True

        if not changed:
            break

    return assignment
