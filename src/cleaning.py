from votekit.cvr_loaders import load_csv, load_scottish
from votekit.cleaning import remove_and_condense
from votekit.utils import mentions
from votekit.matrices import boost_matrix 
import numpy as np

def profile_to_boost_and_cands(profile):
    num_ballots_cast = profile.total_ballot_wt

    new_profile = remove_and_condense("overvote", profile)

    num_ballots_spoiled_by_ov_skips = num_ballots_cast - new_profile.total_ballot_wt 
    print(f"{num_ballots_spoiled_by_ov_skips} ballots, or {num_ballots_spoiled_by_ov_skips/num_ballots_cast:.1%} of all ballots, were spoiled by overvotes or skips.")

    # remove UWI's here
    write_in_candidates = [key for key in profile.candidates if str(key).startswith("Write-in")]
    write_in_candidates += [key for key in profile.candidates if str(key).startswith("Uncertified")]

    # Remove and condense write-in candidates
    new_profile = remove_and_condense(write_in_candidates, new_profile)

    num_ballots_scrubbed_by_wi = num_ballots_cast - num_ballots_spoiled_by_ov_skips-new_profile.total_ballot_wt 
    print(f"{num_ballots_scrubbed_by_wi} ballots, or {num_ballots_scrubbed_by_wi/num_ballots_cast:.1%} of all ballots, were scrubbed by write ins.")

    # get list of candidates
    candidates = list(new_profile.candidates)
    # get boost matrix from profile and cadidates, clean it (replace nans with 0s)
    bm  = boost_matrix(profile, candidates)
    bm_clean = np.nan_to_num(bm)
    return new_profile, bm_clean, candidates


def Portland_clean(profile):
    num_ballots_cast = profile.total_ballot_wt

    num_overvotes_first_place = sum(b.weight for b in profile.ballots if b.ranking[0] == {"overvote"})
    num_ballots_with_overvotes = sum(b.weight for b in profile.ballots if any(cand_set == {"overvote"} for cand_set in b.ranking))

    new_profile = remove_and_condense("overvote", profile)

    num_ballots_spoiled_by_ov_skips = num_ballots_cast - new_profile.total_ballot_wt 
    print(f"{num_ballots_spoiled_by_ov_skips} ballots, or {num_ballots_spoiled_by_ov_skips/num_ballots_cast:.1%} of all ballots, were spoiled by overvotes or skips in D1.")

    # remove UWI's here
    write_in_candidates = [key for key in profile.candidates if str(key).startswith("Write-in")]
    write_in_candidates += [key for key in profile.candidates if str(key).startswith("Uncertified")]

    # Remove and condense write-in candidates
    new_profile = remove_and_condense(write_in_candidates, new_profile)

    num_ballots_scrubbed_by_wi = num_ballots_cast - num_ballots_spoiled_by_ov_skips-new_profile.total_ballot_wt 
    print(f"{num_ballots_scrubbed_by_wi} ballots, or {num_ballots_scrubbed_by_wi/num_ballots_cast:.1%} of all ballots, were scrubbed by write ins in D1.")
    return new_profile
