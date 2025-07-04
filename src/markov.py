from votekit.utils import mentions
from votekit.pref_profile import PreferenceProfile
import random
import numpy as np
import tqdm

def gen_mentions_partition(profile:PreferenceProfile, cands, k):
    my_mentions= mentions(profile) # this is a dict with keys the string names of candidates and values the number of mentions
    ncand = len(cands)
    sorted_candidates = sorted(cands, key=my_mentions.get, reverse=True) #sorted from most mentioned to least mentioned
    q, r = divmod(ncand, k)
    # r parts of size q+1, k-r parts of size q
    bloc_sizes = [q+1]*r + [q]*(k - r)
    partition = []
    for L in bloc_sizes:
        partition.append(sorted_candidates[:L])
        sorted_candidates = sorted_candidates[L:]
    return partition

def random_partition(cands, k):
    shuffled = cands[:]
    random.shuffle(shuffled)
    partition = [[] for _ in range(k)]
    for idx, item in enumerate(shuffled):
        partition[idx % k].append(item)
    return partition


def random_partition_variable_lengths(cands, part_lengths):
    '''
    Randomly partitions the candidates into parts of the given lengths
    args:
        cands: an iterable of candidates
        part_lengths: a list of integers st sum(part_lengths) =
            len(cands)
    returns:
        random partition where each part is of the specified length

    TODO: write some unittests
    '''
    if sum(part_lengths) != len(cands):
        raise Exception("part_lengths not valid")

    result = []
    for l in part_lengths:
        part_i = []
        for i in range(l):
            part_i.append(cands.pop(random.randint(0,len(cands))-1)) # TODO: check that cands is being passed by value and not by ref
        result.append(part_i)
    return result

def random_partition_random_lengths(cands):
    '''
    Randomly partitions the given candidates into parts of random size
    NOTE: this may output the trivial partition (is that what we want?)
    '''
    part_lengths = []
    remaining_lenght = len(cands)
    while sum(part_lengths) < len(cands):
        next_part_size = random.randint(1, remaining_lenght)
        part_lengths.append(next_part_size)
        remaining_lenght -= next_part_size

    return random_partition_variable_lengths(cands, part_lengths)


def naive_proposal(partition):
    """Randomly propose a new partition as follows:
    1. Choose a candidate uniformly at random.
    2. Choose a slate uniformly at random, and move the candidate to that slate."""
    candidates = [cand for bloc in partition for cand in bloc]
    new_partition = [part.copy() for part in partition]
    random_candidate = candidates[random.randint(0,len(candidates)-1)]
    random_partition = random.randint(0,len(new_partition)-1)
    #Remove the random candidate from their partition
    for part in new_partition:
        if random_candidate in part:
            part.remove(random_candidate)
    new_partition[random_partition].append(random_candidate)
    return new_partition

def tilted_run(profile: PreferenceProfile, partition, score_fn, proposal = naive_proposal,iterations=1000, beta=np.log(2)/10000, maximize = False):
    cur_score = score_fn(profile, partition)
    best_score = float(cur_score)
    cur_partition = [part.copy() for part in partition]
    best_partition = [part.copy() for part in partition]
    my_beta = beta  # This is the "tilt" parameter; lower values make it more likely to accept worse proposals.
    if maximize:
        beta*= -1
    for _ in range(iterations): #hill-climb interpretation.
        new_partition = proposal(cur_partition)
        new_score = score_fn(profile, new_partition)
        cutoff = np.exp(beta*(cur_score - new_score))
        if np.random.random() < cutoff:
            cur_partition = new_partition
            cur_score = new_score
            if cur_score < best_score:
                best_score = float(cur_score)
                best_partition = [part.copy() for part in cur_partition]
                #print("New best!")
    if best_score == float(score_fn(profile, partition)):
        print("Chain did not find a better partition. Consider increasing iterations!")
        return cur_partition
    
    return best_partition

def short_burst(profile: PreferenceProfile, partition, score_fn, burst_size, num_bursts):
    status_quo = score_fn(profile, partition)
    burst_best = partition.copy()
    for _ in tqdm.tqdm(range(num_bursts)):
        trial_step = burst_best
        for _ in range(burst_size):
            trial_step = naive_proposal(trial_step)
            quo = score_fn(profile, trial_step)
            if quo <= status_quo:
                burst_best = trial_step.copy()
                status_quo = int(quo)
    return burst_best
