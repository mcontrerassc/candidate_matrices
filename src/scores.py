import numpy as np
from votekit.pref_profile import PreferenceProfile
from votekit.utils import mentions


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

    row_sums = blockiness.sum(axis=1)
    normed_blockiness = np.zeros_like(blockiness)

    for i in range(len(blockiness)):
        if row_sums[i] != 0:
            normed_blockiness[i] = blockiness[i] / row_sums[i]

    return normed_blockiness

def gen_share_mentions(profile: PreferenceProfile):
    num_mentions = mentions(profile)
    total_mentions = sum(num_mentions.values())
    share_mentions = {key : value/total_mentions for key,value in num_mentions.items()}
    return share_mentions

def relative_size_score(profile: PreferenceProfile, partitions):
    share_mentions = gen_share_mentions(profile)
    sizes = []
    for part in partitions:
        sizes.append(
            sum([share_mentions[candidate] for candidate in part])
        )
    score = 1
    for size in sizes:
        score *= (size+.1) # add a small constant to avoid division by zero
    return 1/score

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
            
    adjacencies = adjacencies + adjacencies.T
    return adjacencies

def cut_score(profile: PreferenceProfile, partitions): #data structure is a list of lists of candidates
    sum = 0
    cands = profile.candidates
    candidate_to_index = {candidate : i for i, candidate in enumerate(cands)}
    adjacencies = make_adjacency_matrix(profile, candidate_to_index)
    for part1 in partitions:
        for part2 in partitions:
            if part1 == part2:
                continue
            for c1 in part1:
                for c2 in part2:
                    sum += adjacencies[
                        candidate_to_index[c1],
                        candidate_to_index[c2]
                        ]
    return sum

def brantley_score(profile: PreferenceProfile, partition):
    alpha, beta = 1, 10000 #Change these to balance the objectives!
    return alpha * cut_score(profile, partition) + beta * relative_size_score(profile, partition)# - np.trace(blockiness_mtx(profile, partition))