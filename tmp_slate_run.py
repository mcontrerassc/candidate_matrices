import numpy as np
from pickle import load
from votekit.utils import mentions
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
#from src.viz import viz_order
from src.markov import gen_mentions_partition, random_partition, naive_proposal, tilted_run, short_burst
from src.scores import combined_score, cut_score, relative_size_score
from src.cleaning import Portland_clean
from src.clustering import create_graph_louvain_bm, louvain_partition_from_graph, find_distance_to_slate_optimal_bipartition
from functools import partial

if __name__ == "__main__":
    with open('./data/Portland_D1.pkl', 'rb') as file:
        loaded = load(file)
    boost = loaded['boost']
    profile = loaded['profile']
    #profile, boost = clean_and_make_boost(loaded['profile'])
    candidates = list(profile.candidates) #note: we call this the canonical ordering of candidates. It is important for viz + conversion tools.
    print(candidates)

    all_bipart_distance_scores = find_distance_to_slate_optimal_bipartition(profile)
