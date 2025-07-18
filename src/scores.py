import numpy as np
from votekit.pref_profile import PreferenceProfile
from votekit.utils import mentions
from itertools import accumulate
from src.markov import forward_convert, backward_convert, fast_proposal_generator
from votekit.matrices import comention, boost_matrix
from votekit.ballot import Ballot
from typing import Tuple

def bal_to_tuple(ballot, cand_ref): #blame Chris
    sane_tuple = tuple(set(fset).pop() for fset in ballot.ranking)
    return tuple(cand_ref[c] for c in sane_tuple), ballot.weight

def tuple_to_bal(tup, weight, candidates):
    ranking = [{candidates[i]} for i in tup]
    return Ballot(ranking=ranking, weight=weight)

# returns an n x n matrix where (i, j) is the row-normalized weight of ballots 
# where a candidate from block j was ranked 1st and a candidate from block i was ranked 2nd
# each block is a list of candidate names, and blocks is a list of these blocks.
def blockiness_mtx(profile: PreferenceProfile, partition):
    num_blocks = len(partition)
    block_lists = [set(block) for block in partition]
    blockiness = np.zeros((num_blocks, num_blocks))

    for ballot in profile.ballots:
        # get first ranked candidate and their block    
        first_rank = ballot.ranking[0]
        first_cand = list(first_rank)[0]
        first_block = None
        for i, block in enumerate(block_lists):
            if first_cand in block:
                first_block = i
                break
        # if bullet ballot then count it as if second vote was also in that block
        # TODO is this the "right" approach?
        if len(ballot.ranking) == 1:
            blockiness[first_block, first_block] += float(ballot.weight)
            continue

        # get secon ranked candidate and their block
        second_rank = ballot.ranking[1]
        second_cand = list(second_rank)[0]
        second_block = None
        for i, block in enumerate(block_lists):
            if second_cand in block:
                second_block = i
                break
                   
        blockiness[second_block, first_block] += float(ballot.weight)

    # normalize along the columns 
    column_sums = blockiness.sum(axis=0)
    normed_blockiness = np.zeros_like(blockiness)

    for i in range(len(blockiness)):
        if column_sums[i] != 0:
            normed_blockiness[i] = blockiness[i] / column_sums[i]

    return normed_blockiness

def fpv_boost_prob(i: str, j: str, pref_profile: PreferenceProfile) -> Tuple[float, float]:
    candidate_to_index = {candidate: i for i, candidate in enumerate(pref_profile.candidates)}
    candidates = list(pref_profile.candidates)
    i_fpv_mentions = 0.0
    j_mentions = 0.0
    fpv_i_j_mentions = 0.0
    for ballot in pref_profile.ballots:
        bal_tuple, weight = bal_to_tuple(ballot, candidate_to_index)
        bal_str = tuple(candidates[idx] for idx in bal_tuple)
        if (bal_str[0] == i):
            i_fpv_mentions += weight
        if comention(j, ballot):
            j_mentions += weight
            if (bal_str[0] == i):
                fpv_i_j_mentions += weight
    return (
        float(fpv_i_j_mentions) / j_mentions if (j_mentions != 0) else np.nan,
        (
            float(i_fpv_mentions) / pref_profile.total_ballot_wt
            if (pref_profile.total_ballot_wt != 0)
            else np.nan
        ),
    )
def fpv_boost_matrix(pref_profile: PreferenceProfile) -> np.ndarray:
    candidates = list(pref_profile.candidates)
    fpv_boost_matrix = {c: {c: 0.0 for c in candidates} for c in candidates}
    for i in candidates:
        for j in candidates:
            if i != j:
                cond, uncond = fpv_boost_prob(i, j, pref_profile)
                fpv_boost_matrix[i][j] = cond - uncond
            else:
                fpv_boost_matrix[i][j] = np.nan
    size = len(candidates)
    fpv_boost_array = np.empty((size, size))
    for i, cand_i in enumerate(candidates):
        for j, cand_j in enumerate(candidates):
            fpv_boost_array[i, j] = fpv_boost_matrix[cand_i][cand_j]
    return np.nan_to_num(fpv_boost_array)

def make_boost_matrix(profile: PreferenceProfile, candidates = None, example_partition8=None):
    boost  = boost_matrix(profile, candidates = list(profile.candidates))
    boost_clean = np.nan_to_num(boost)
    return boost_clean

def sum_mass(M, matrix_name, example_partition8=None): #this score expects a matrix with negative and positive entries (like a boost matrix)
    def score(partition8):
        summ = 0
        for i, s in enumerate(partition8[:-1]):
            for j, t in enumerate(partition8[i:]):
                if s == t:
                    summ += M[i, j+i] + M[j+i, i]
        return -summ
    name = f"{matrix_name}_sum_mass"
    score.score_name = name
    return score



def balanced_sum_mass(M, matrix_name, factor = 1,example_partition8=None): #this score expects a matrix with positive entries only, and punishes slates that are too small
    #only keep positive entries
    if type(example_partition8) == np.ndarray:
        k = max(example_partition8) + 1
    elif type(example_partition8) == int:
        k = example_partition8
    M_pos = np.where(M > 0, M, 0)
    def score(partition8):
        summ = 0
        for i, s in enumerate(partition8[:-1]):
            for j, t in enumerate(partition8[i+1:]):
                if s == t:
                    summ += M[i, j+i+1] + M[j+i+1, i]
        return summ
    eye = np.eye(M.shape[0])
    my_head_is_empty = make_good(eye, example_partition8=example_partition8, matrix_name='eye')
    #run a short (1000 steps) chain to callibrate the score to have median 100 and standard deviation 25
    proposal = fast_proposal_generator(example_partition8)
    cur = example_partition8.copy()
    mass_scores = []
    eye_scores = []
    for _ in range(1000):
        cur = proposal(cur)
        mass_scores.append(score(cur))
        eye_scores.append(my_head_is_empty(cur))
    mass_scores = np.array(mass_scores)
    eye_scores = np.array(eye_scores)
    mass_median = np.median(mass_scores)
    #mass_iqr = np.percentile(mass_scores, 75) - np.percentile(mass_scores, 25)
    eye_median = np.median(eye_scores)
    #eye_std = np.std(eye_scores)
    #mass_std = np.std(mass_scores)
    #eye_IQR = np.percentile(eye_scores, 75) - np.percentile(eye_scores, 25)
    #scale the scores to have median 100
    mass_scale = 100/mass_median
    #mass_scale = 1
    #mass_shift = 100 - mass_median*mass_scale
    eye_scale = 100/eye_median 
    #eye_shift = 100 - eye_median*eye_scale
    #print(eye_median, eye_IQR, eye_std, mass_median, mass_iqr, mass_std)
    def scaled_score(partition8):
        return -(mass_scale * score(partition8) ) + (factor*eye_scale * my_head_is_empty(partition8)) 
    name = f"{matrix_name}_balanced_sum_mass"
    scaled_score.score_name = name
    return scaled_score

def gen_share_mentions(profile: PreferenceProfile, key_type = "string"):
    num_mentions = mentions(profile)
    total_mentions = sum(num_mentions.values())
    candidates = profile.candidates #canonical
    if key_type == "string":
        share_mentions = {key : value/total_mentions for key,value in num_mentions.items()}
    elif key_type == "index":
        share_mentions = {candidates.index(key) : value/total_mentions for key, value in num_mentions.items()}
    return share_mentions

def relative_size_score(profile: PreferenceProfile, partitions):
    if type(partitions) == np.ndarray:
        partition = backward_convert(partitions, profile.candidates)
    else:
        partition = partitions.copy()
    share_mentions = gen_share_mentions(profile)
    sizes = []
    for part in partition:
        sizes.append(
            sum([share_mentions[candidate] for candidate in part])
        )
    score = 1
    for size in sizes:
        score *= (size+.01) # add a small constant to avoid division by zero
    return 1/score

def relative_size_score_generator(profile: PreferenceProfile, k):
    menshons = gen_share_mentions(profile, key_type = "index")
    def fast_relative_size_score(partition: np.ndarray):
        sizes = np.zeros(k) + 0.01
        for i, t in enumerate(partition):
            sizes[t] += menshons[i]
        return 1/np.prod(sizes)
    fast_relative_size_score.score_name = "rel_size"
    return fast_relative_size_score

def make_adjacency_matrix(profile: PreferenceProfile,candidate_to_index):
    adjacencies = np.zeros((len(candidate_to_index), len(candidate_to_index)))
    for ballot in profile.ballots:
        ranking = []
        for fset in ballot.ranking: #please never call a variable "set" in Python
            if len(fset) == 1: #this should always be true, otherwise the ranking would be listed as "overvote"
                name, = fset
                if name in ranking:
                    continue #Repeated name in this one. Throw it out.
                if name in candidate_to_index.keys():
                    ranking.append(name) #Ignore names that aren't in the candidate list
        
        if len(ranking) <= 1:
            continue
        
        for i in range(len(ranking) - 1):
            adjacencies[candidate_to_index[ranking[i]],
                        candidate_to_index[ranking[i+1]] #this seems like an error, should be i+1
                        ] += ballot.weight
            
    #normalize the adjacency matrix so the whole matrix sums to 1
    total_weight = np.sum(adjacencies)
    if total_weight > 0:
        adjacencies /= total_weight
    return adjacencies

def fast_adj(profile: PreferenceProfile):
    candidate_to_index = {candidate: i for i, candidate in enumerate(profile.candidates)} #this is the canonical ordering of candidates
    adjacencies = np.zeros((len(profile.candidates), len(profile.candidates)))
    for bal in profile.ballots:
        good_bal, w = bal_to_tuple(bal, candidate_to_index)
        if len(good_bal)>1:
            for i in range(len(good_bal) - 1):
                adjacencies[good_bal[i], good_bal[i+1]] += w
    #normalize the adjacency matrix so the whole matrix sums to 1
    total_weight = np.sum(adjacencies)
    if total_weight > 0:
        adjacencies /= total_weight
    return adjacencies

def proportional_successive_matrix(profile: PreferenceProfile):
    candidate_to_index = {candidate: i for i, candidate in enumerate(profile.candidates)} #this is the canonical ordering of candidates
    adjacencies = np.zeros((len(profile.candidates), len(profile.candidates)))
    for bal in profile.ballots:
        good_bal, w = bal_to_tuple(bal, candidate_to_index)
        if len(good_bal)>1:
            for i in range(len(good_bal) - 1):
                adjacencies[good_bal[i], good_bal[i+1]] += w
    menshons = mentions(profile)
    for cand, i in candidate_to_index.items():
        if menshons[cand] > 0:
            adjacencies[i, :] /= menshons[cand]
    return adjacencies

def cut_score_generator(profile: PreferenceProfile, k=None):
    adjacencies = fast_adj(profile)
    def fast_cut_score(partition8):
        sum = 0
        for i, s in enumerate(partition8[:-1]):
            for j, t in enumerate(partition8[i+1:]):
                if s != t:
                    sum += adjacencies[i, j+i+1] + adjacencies[j+i+1, i]
        return sum
    fast_cut_score.score_name = "cut"
    return fast_cut_score

def cut_score(profile: PreferenceProfile, partitions): #data structure is a list of lists of candidates
    if type(partitions) == np.ndarray:
        partition = backward_convert(partitions, profile.candidates)
    else:
        partition = partitions
    sum = 0
    cands = profile.candidates
    candidate_to_index = {candidate : i for i, candidate in enumerate(cands)}
    adjacencies = make_adjacency_matrix(profile, candidate_to_index)
    for part1 in partition:
        for part2 in partition:
            if part1 == part2:
                continue
            for c1 in part1:
                for c2 in part2:
                    sum += adjacencies[
                        candidate_to_index[c1],
                        candidate_to_index[c2]
                        ]
    return sum

def split_matrix(A):
    A_pos = np.where(A > 0, A, 0)
    A_neg = np.where(A < 0, A, 0)
    return A_pos, A_neg

def make_modularity_matrix(A):
    kout = A.sum(axis=1)  # shape (n,)
    kin = A.sum(axis=0)  # shape (n,)
    m = A.sum()

    if m == 0:
        return np.zeros_like(A)

    expected = np.outer(kout, kin) / m
    B = A - expected
    #set all diagonal entries to 0
    np.fill_diagonal(B, 0)
    return B

def modularity_from_B(B, m, assignment, mod_type = 'standard', pm='pos'):
    assert pm in {'pos', 'neg'}
    n = len(assignment)
    Q = 0.0
    if m == 0:
        return 0.0

    for i in range(n):
        for j in range(n):
            same_group = (assignment[i] == assignment[j])
            if mod_type == 'standard':
                if (pm == 'pos' and same_group) or (pm == 'neg' and not same_group):
                    Q += B[i, j]
            elif mod_type == 'hybrid':
                if (pm == 'pos' and same_group) or (pm == 'neg' and not same_group):
                    Q += B[i, j]
                else:
                    Q -= B[i, j]
            elif mod_type == 'reverse':
                if (pm == 'pos' and not same_group) or (pm == 'neg' and same_group):
                    Q -= B[i, j]
            else:
                raise ValueError(f"Unknown mod_type: {mod_type}")

    return Q / m

def hybrid_modularity(M, matrix_name, example_partition8=None):
    A,B = split_matrix(M)
    A_mod = make_modularity_matrix(A)
    m_A = A.sum()
    B_mod = make_modularity_matrix(-B)
    m_B = -B.sum()
    def score(partition8):
        Q_A = modularity_from_B(A_mod, m_A, partition8, mod_type='standard', pm='pos')
        Q_B = modularity_from_B(B_mod, m_B, partition8, mod_type='standard', pm='neg')
        return -Q_A - Q_B
    score.score_name = f"{matrix_name}_hybrid"
    return score

def standard_modularity(M, matrix_name, example_partition8=None):
    A,B = split_matrix(M)
    A1 = make_modularity_matrix(A)
    m_A = A.sum()
    def score(partition8):
        Q = modularity_from_B(A1, m_A, partition8, mod_type='standard', pm='pos')
        return -Q
    score.score_name = f"{matrix_name}_standard"
    return score

def make_not_bad(matrix, example_partition8= None, matrix_name = "matrix"): #this matrix better be in canonical order or so help me god
    def fast_score(partition8):
        summ = 0
        for i, s in enumerate(partition8[:-1]):
            for j, t in enumerate(partition8[i+1:]):
                if s != t:
                    summ += matrix[i, j+i+1] + matrix[j+i+1, i]
        return summ
    #name should look like <matrix>_not_bad
    name = f"{matrix_name}_not_bad"
    fast_score.score_name = name
    return fast_score

def make_good(matrix, example_partition8, matrix_name = "matrix"): #canonical order or riot
    #if we just summed the diagonal entries of the matrix, this would usually give us a score we want to maximize
    #to make it a score we want to minimize, keep the sum over each slate separate, and take their reciprocals -- so now we want to maximize the score of each slate separately
    #this also makes it so the resulting score dislikes empty slates
    if type(example_partition8) == np.ndarray:
        k = max(example_partition8) + 1
    elif type(example_partition8) == int:
        k = int(example_partition8) + 1
    def fast_score(partition8):
        slate_sums = np.zeros(k) + .01
        for i, s in enumerate(partition8[:-1]):
            for j, t in enumerate(partition8[i:]):
                if s == t:
                    slate_sums[s] += matrix[i, j+i] + matrix[j+i, i]
        return sum(1/s for s in slate_sums)
    #name should look like <matrix>_good
    name = f"{matrix_name}_good"
    fast_score.score_name = name
    return fast_score

def first_second_score(profile: PreferenceProfile, partitions):
    score = np.trace(blockiness_mtx(profile, partitions))
    return score

def combined_score(profile: PreferenceProfile, partition,alpha=1,beta=10000):
    #alpha, beta = 1, 10000 #Change these to balance the objectives!
    return alpha * cut_score(profile, partition) + beta * relative_size_score(profile, partition)# - np.trace(blockiness_mtx(profile, partition))


## distance metric from ``Learning Blocs and Slates from Ranked-Choice Ballots''

## question: do we want to enforce a particular encoding on the slates and ballots?
def distance_to_slate(ballot, bipartition):
    '''
    takes in a ballot and a bi-partition and returns the distance to
    slate score cited above.
    args:
        ballot: Tuple/list; a ranking of the candidates, must include each
            element of [n] exactly once
        bipartition: A pair list of size two, containing
            (sets/tuples?). The union of the substructures must equal
            [n] (as sets). 
            The first substructure is considered slate A and the
            second substructure is considered slate B
    returns:
        float: the distance to slate score
    '''
    # perhaps do some error hadling on bipartition here
    slate_A, slate_B = bipartition
    slate_A_ind = [ballot.index(A_cand) for A_cand in slate_A]
    slate_B_ind = [ballot.index(B_cand) for B_cand in slate_B]

    # for each B index, count the number of A_indices which lay above
    numerator = sum([len(list(filter(lambda a_ind: a_ind > B_ind, slate_A_ind))) for B_ind in slate_B_ind])

    return numerator / (len(slate_A) * len(slate_B))


def distance_to_slate_across_profile(profile: PreferenceProfile, partition):
    '''
        Takes in a bipartition and then computes the distance to slate
        score across the profile.

        partition: a list of lists which union to the candidates 
    ''' 
    
    # TODO: is this the way we want to handle the distance to slate
    # score?
    if len(partition) != 2:
        raise Exception("More than two blocks given for distance_to_slate_across_profile. Cannot" \
            "compute distance to slate for more than two blocks")

    # TODO: do we want to compute the distance to slate in both
    # orientations and return the min/max?

    ballots = profile.ballots()
    ballots_to_distance_first_slate = [distance_to_slate(ballot, partition) for ballot in ballots]
    ballots_to_distance_second_slate = [distance_to_slate(ballot, partition[::-1]) for ballot in ballots]
    return max(sum(ballots_to_distance_first_slate), sum(ballots_to_distance_second_slate))

def truncate_profile(profile: PreferenceProfile, length = 3):
    new_ballots = []
    candidates = list(profile.candidates)
    candidate_to_index = {c: i for i, c in enumerate(candidates)}
    for ballot in profile.ballots:
        tup, w = bal_to_tuple(ballot, candidate_to_index)
        new_tup = tup[:length]
        new_ballot = tuple_to_bal(new_tup, w, candidates)
        new_ballots.append(new_ballot)
    return PreferenceProfile(ballots=tuple(new_ballots), candidates=candidates, max_ranking_length=length)

def truncated_boost(profile: PreferenceProfile, L = 3):
    truncated_profile = truncate_profile(profile, length = L)
    boost = make_boost_matrix(truncated_profile)
    return np.nan_to_num(boost)