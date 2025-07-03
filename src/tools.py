from itertools import combinations, chain
import numpy as np

# misc functions

def enumerate_bipartitions(candidates, as_generator = False, disallow_trivial = True):
    '''
        candidates: a list of candidate name
        as_generator: Flag to indicate that the result should be
            returned as a generator
        disallow_trivial: Flag to indicate that we should exclude
            partitions where one part is empty
        returns:
            all bipartitions of candidates as a list of lists
    '''
    # generate the powerset of candidates
    comb_range = range(1, len(candidates)) if disallow_trivial else range(len(candidates) + 1)
    powerset = chain.from_iterable(combinations(candidates, r) for r in comb_range)
    set_cands = set(candidates)
    all_bi_parts = np.array([ [set(subset), set_cands - set(subset)] for subset in powerset ])
    #return chain.from_iterable(all_bi_parts) if as_generator else
    #all_bi_parts
    
    # TODO: need to implement manual yield to return tuples rather
    # than using chain constructor. Uncomment the above return line
    # when ready
    return all_bi_parts
