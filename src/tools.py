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


# TODO: is this the right name for this method?
def profile_ballots_to_list(ballots):
    '''
        Takes in a list of Votekit ballots and converts each ballot to
        a tuple of strings
        args:
            ballots: list of Ballot objects
        returns:
            list of tuples
    '''
    # TODO: add some type checking to the argument (or at least see
    # what guarentees votekit gives)
    # TODO: add some tests

    # each ballot.ranking is a tuple of frozen sets
    # cannot return this as np array because the shape is inhomogenous
    # we could return it as a generator?
    return [tuple([list(cand)[0] for cand in bal.ranking]) for bal in ballots]
