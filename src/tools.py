from itertools import combinations
import numpy as np

# misc functions

def enumerate_bipartitions(candidates, as_generator = False):
    '''
        candidates: a list of candidate names
        returns:
            a 2d generator which enumeratees all bipartitions
    '''

    '''
        let's just do the naive thing and generate the powerset of the
        candidates. Then we can think about how write it as a
        generator instead. 
        Do we want to generate the trivial partitions?
    '''
    if as_generator:
        pass
    
    # generate the powerset of candidates
    powerset = [combinations(candidates, r) for r in range(len(candidates) + 1)] 
    set_cands = set(candidates)
    return np.array([np.array([set(subset), set_cands - set(subset)] for subset in powerset)])
