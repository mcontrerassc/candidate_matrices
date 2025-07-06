import numpy as np
from votekit.pref_profile import PreferenceProfile
from votekit.utils import mentions
from itertools import accumulate
from src.markov import forward_convert, backward_convert

def bal_to_tuple(ballot, cand_ref): #blame Chris
    sane_tuple = tuple(set(fset).pop() for fset in ballot.ranking)
    return tuple(cand_ref[c] for c in sane_tuple), ballot.weight

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

def cut_score_generator(profile: PreferenceProfile):
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
            for j, t in enumerate(partition8[i+1:]):
                if s == t:
                    slate_sums[s] += matrix[i, j+i+1] + matrix[j+i+1, i]
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
