from itertools import combinations, chain
import numpy as np

# misc functions

def enumerate_bipartitions(candidates, as_generator = False):
    '''
        candidates: a list of candidate names
        returns:
            a 2d generator which enumeratees all bipartitions
    '''
    # generate the powerset of candidates
    powerset = chain.from_iterable(combinations(candidates, r) for r in range(len(candidates) + 1))
    set_cands = set(candidates)
    all_bi_parts = np.array([ [set(subset), set_cands - set(subset)] for subset in powerset ])
    #return chain.from_iterable(all_bi_parts) if as_generator else
    #all_bi_parts
    
    # TODO: need to implement manual yield to return tuples rather
    # than using chain constructor. Uncomment the above return line
    # when ready
    return all_bi_parts
