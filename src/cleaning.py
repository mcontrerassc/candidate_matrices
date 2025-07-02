from votekit.cvr_loaders import load_csv
from votekit.cleaning import remove_and_condense

def Portland_clean(profile):
    num_ballots_cast = profile.total_ballot_wt

    num_overvotes_first_place = sum(b.weight for b in profile.ballots if b.ranking[0] == {"overvote"})
    num_ballots_with_overvotes = sum(b.weight for b in profile.ballots if any(cand_set == {"overvote"} for cand_set in b.ranking))

    new_profile = remove_and_condense("overvote", profile)

    num_ballots_spoiled_by_ov_skips = num_ballots_cast - new_profile.total_ballot_wt 
    print(f"{num_ballots_spoiled_by_ov_skips} ballots, or {num_ballots_spoiled_by_ov_skips/num_ballots_cast:.1%} of all ballots, were spoiled by overvotes or skips in D1.")

    # remove UWI's here
    write_in_candidates = [key for key in profile.candidates if str(key).startswith("Write-in")]

    # Remove and condense write-in candidates
    new_profile = remove_and_condense(write_in_candidates, new_profile)

    num_ballots_scrubbed_by_wi = num_ballots_cast - num_ballots_spoiled_by_ov_skips-new_profile.total_ballot_wt 
    print(f"{num_ballots_scrubbed_by_wi} ballots, or {num_ballots_scrubbed_by_wi/num_ballots_cast:.1%} of all ballots, were scrubbed by write ins in D1.")
    return new_profile