from votekit.utils import mentions, ballots_by_first_cand
from votekit.cleaning import remove_and_condense
from votekit.pref_profile import PreferenceProfile
from votekit.utils import mentions
from votekit.matrices import matrix_heatmap, boost_matrix
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union, Tuple
from scipy import sparse
import networkx as nx
from sknetwork.utils.check import check_format
from sknetwork.utils.format import get_adjacency
from sknetwork.utils.membership import get_membership
from cdlib import algorithms, evaluation
from collections import defaultdict
from networkx.algorithms import community
import sknetwork as skn
from src.scores import make_good, make_not_bad, fast_adj



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


def second_modularity(G: nx.Graph, resolution):
        model = skn.clustering.Louvain(resolution=resolution)   
        labels = model.fit_predict(G)
            # convert label_dict to our standard format for a clustering: a list of elections.
        k = max(label_dict.values()) + 1 # number of clusters
        C = [{} for _ in range(k)]
        for ballot, weight in election.items():
            truncated_ballot = ballot[:trunc]
            label = label_dict[truncated_ballot]
            C[label][ballot] = weight
        # remove empty clusters (which come from labels only assigned to uncast ballots)
        C = [c for c in C if sum(c.values()) > 0]  

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


def get_directed_modularity(adjacency, labels, resolution=1.0, nature = 'pos'):
    adjacency = check_format(adjacency)
    m = adjacency.data.sum()

    if nature == 'pos':
        membership = get_membership(labels).astype(float).toarray()
    elif nature == 'neg':
        membership = 1 - get_membership(labels).astype(float).toarray()
    
    k_out = adjacency.sum(axis=1).A1  # row sums
    k_in = adjacency.sum(axis=0).A1  # col sums

    # actual edge weight 
    fit = membership.T.dot(adjacency.dot(membership)).diagonal().sum() 

    # expected edge weight
    expected = (membership.T.dot(k_out).dot(membership.T.dot(k_in))) / m

    return (fit - resolution * expected)/m


## mother louvain -> for now can take 3 scores, Q+, make_good, make_not_bad
## TODO @michelle : hybrid, make honest not bad?, Q^- (use adj neg) ??!
def homemade_louvain(adjacency_pos, adjacency_neg, resolution, max_iter, score, boost, adj):
    # get each candidate's neighs, considering pos and neg for adjecencies @michelle check this logic later!
    graph_structure = (adjacency_pos + adjacency_neg.T).astype(bool).astype(int).tocsr()
    neighbors_list = [ graph_structure.indices[graph_structure.indptr[i]:graph_structure.indptr[i+1]]
    for i in range(graph_structure.shape[0])]
    n = len(neighbors_list)

    #  each node in its own cluster
    labels = np.arange(n)

    for i in range(max_iter):
        moved = False

        # the nondeterministic component of louvain -> nodes are chekced in != orders
        for node in np.random.permutation(n):
            current_label = labels[node]
            neighbors = neighbors_list[node]
            neigh_labels = np.unique(labels[neighbors])

            # TODO make this float +- inf depending on score, 0 works well for now but think ab edge cases
            best_gain = 0
            if score == 'make_not_bad' or score == 'make_good':
                best_gain = float("inf")
            best_label = current_label

            for label in neigh_labels:
                if label == current_label:
                    continue

                # the new tentative cluster for curr node
                old_label = labels[node]
                labels[node] = label

                if (score == 'make_good'):
                    new_mod = make_good(boost, labels, "matrix")(labels)
                    labels[node] = old_label
                    old_mod = make_good(boost, labels, "matrix")(labels)
                    gain = new_mod - old_mod

                    if gain < best_gain:
                        best_gain = gain
                        best_label = label
                
                elif (score == 'make_not_bad'):
                    new_mod = make_not_bad(adj, labels, "matrix")(labels)
                    labels[node] = old_label
                    old_mod = make_not_bad(adj, labels, "matrix")(labels)
                    gain = new_mod - old_mod

                    if gain < best_gain:
                        best_gain = gain
                        best_label = label

                elif (score == 'green_diagonal'):
                    new_mod = get_directed_modularity(adjacency_pos, labels, resolution)
                    labels[node] = old_label
                    old_mod = get_directed_modularity(adjacency_pos, labels, resolution)
                    gain = new_mod - old_mod

                    if gain > best_gain:
                        best_gain = gain
                        best_label = label

            if best_label != current_label:
                labels[node] = best_label
                moved = True

        if not moved:
            break  # no improvement

    return labels
    
def create_graph_louvain_signed(boost: np.ndarray, pos: bool):
    n = boost.shape[0]
    if pos == True:
        boost = np.where(boost > 0, boost, 0)
    else:
        boost = np.where(boost < 0, -boost, 0)
    # build a graph
    G = nx.Graph()
    for i in range(n):
        #for j in range(i+1, n):
        for j in range(n):
            G.add_edge(i, j, weight = boost[i, j])
    return G

# convert labels to a partition
def labels_to_partitions(labels, candidates):
    clusters = defaultdict(list)
    for label, candidate in zip(labels, candidates):
        clusters[label].append(candidate)
    # return as list of lists sorted by cluster label (optional)
    return [clusters[k] for k in sorted(clusters.keys())]

# resolution kept as 1 
def louvain_partition_signed_graph(boost, profile, resolution=1, iter=100, score='green_diagonal'):
    adj = fast_adj(profile)
    G_neg = create_graph_louvain_signed(boost, False)
    G_pos = create_graph_louvain_signed(boost, True)
    adj_pos = sparse.csr_matrix(nx.to_scipy_sparse_array(G_pos, weight='weight', format='csr'))
    adj_neg = sparse.csr_matrix(nx.to_scipy_sparse_array(G_neg, weight='weight', format='csr'))
    labels = homemade_louvain(adj_pos, adj_neg, resolution,iter,score, boost,adj)
    partition = labels_to_partitions(labels, list(profile.candidates))
    if score == 'green_diagonal':
        metric = get_directed_modularity(adj_pos, labels)
    elif score == 'make_not_bad':
        metric = make_not_bad(boost, labels, "matrix")(labels)
    elif score == 'make_good':
        metric = make_good(boost, labels, "matrix")(labels)
    return partition, metric

