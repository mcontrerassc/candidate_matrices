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

def fast_proposal_generator(partition8):
    k = max(partition8)
    ncand = len(partition8)
    def fast_proposal(partition):
        new_partition = partition.copy()
        new_partition[np.random.randint(0, ncand-1)] = np.random.randint(0, k+1) 
        return new_partition
    return fast_proposal

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
        cutoff = np.exp(my_beta*(cur_score - new_score))
        if np.random.random() < cutoff:
            cur_partition = new_partition
            cur_score = new_score
            if cur_score < best_score:
                best_score = float(cur_score)
                best_partition = [part.copy() for part in cur_partition]
    if best_score == float(score_fn(profile, partition)):
        print("Chain did not find a better partition. Consider increasing iterations!")
    return best_partition

def fast_tilted_run(starting_partition, score_fn, proposal_gen = fast_proposal_generator, iterations=1000, beta=np.log(20), maximize = False): #starting_partition is a numpy int8 array
    proposal = proposal_gen(starting_partition)
    cur_score = score_fn(starting_partition)
    best_score = float(cur_score)
    cur_partition = starting_partition.copy()
    best_partition = starting_partition.copy()
    my_beta = beta  # This is the "tilt" parameter; lower values make it more likely to accept worse proposals.
    if maximize:
        beta*= -1
    for _ in range(iterations): #hill-climb interpretation.
        new_partition = proposal(cur_partition)
        new_score = score_fn(new_partition)
        cutoff = np.exp(my_beta*(cur_score - new_score))
        if np.random.random() < cutoff:
            cur_partition = new_partition.copy()
            cur_score = float(new_score)
            if cur_score < best_score:
                best_score = float(cur_score)
                best_partition = cur_partition.copy()
    if best_score == float(score_fn(starting_partition)):
        print("Chain did not find a better partition. Consider increasing iterations!")
    return best_partition

def fast_short_burst(starting_partition, score_fn, proposal_gen = fast_proposal_generator, burst_size=40, num_bursts=50):
    status_quo = score_fn(starting_partition)
    burst_best = starting_partition.copy()
    proposal = proposal_gen(burst_best)
    for _ in tqdm.tqdm(range(num_bursts)):
        trial_step = burst_best
        for _ in range(burst_size):
            trial_step = proposal(trial_step)
            quo = score_fn(trial_step)
            if quo <= status_quo:
                burst_best = trial_step.copy()
                status_quo = float(quo)
    return burst_best

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

def calibrate_scores(score_list, starting_partition: np.ndarray, iterations=1000, print_weights = False):
    scores = [[] for score in score_list] 
    new_partition = starting_partition.copy()
    proposal = fast_proposal_generator(new_partition)
    ideal = 100 / len(score_list)  # ideal median score for each function if they are to be equally weighted
    for _ in range(iterations):
        new_partition = proposal(new_partition)
        for i, score_fn in enumerate(score_list):
            scores[i].append(score_fn(new_partition))
    medians = [np.median(s) for s in scores]
    if 0 in medians:
        raise ValueError("One of the score functions returned a median of zero. This may indicate that the score function is not appropriate for the given partition.")
    weights = [ideal / median for median in medians]
    def ideal_score_fn(partition):
        return sum(w * score_fn(partition) for w, score_fn in zip(weights, score_list))
    if print_weights:
        print("Weights for score functions (to achieve equal median scores):")
        for i, weight in enumerate(weights):
            print(f"Score function {i}: {weight}")
    return ideal_score_fn

def forward_convert(partition, canonical_candidates):
    """Convert a partition (list of lists) into a numpy int8 array."""
    cand_dict = {candidate: i for i, candidate in enumerate(canonical_candidates)}
    array = np.zeros(sum(len(block) for block in partition), dtype=np.int8)
    for i, bloc in enumerate(partition):
        for candidate in bloc:
            array[cand_dict[candidate]] = i
    return array

def backward_convert(array, canonical_candidates):
    """Convert a numpy int8 array back into a partition (list of lists)."""
    cand_dict = {i: candidate for i, candidate in enumerate(canonical_candidates)}
    partition = []
    for i in range(max(array) + 1):
        bloc = [cand_dict[j] for j in range(len(array)) if array[j] == i]
        partition.append(bloc)
    return partition
