import click
import json
import jsonlines as jl
import numpy as np
import pickle
from votekit import PreferenceProfile
import warnings
from src.markov import forward_convert, backward_convert, random_partition, calibrate_scores, fast_tilted_run3, fast_short_burst2
from src.scores import fast_adj, proportional_successive_matrix, cut_score_generator, relative_size_score_generator, make_good, make_not_bad, fpv_boost_matrix, sum_mass, make_boost_matrix, balanced_sum_mass, hybrid_modularity, standard_modularity
#from src.cleaning

#it turns out schmillion = 30

#warnings.filterwarnings("ignore")

matrix_types = {
    "ADJ": fast_adj,
    "PSM": proportional_successive_matrix,
    "boost": make_boost_matrix,
    "FPV": fpv_boost_matrix,
    #"top2boost"
}

score_types = {
    "cut": cut_score_generator,
    "rel": relative_size_score_generator,
    "good": make_good,
    "notbad": make_not_bad,
    "sum_mass": sum_mass,
    "mod_standard": standard_modularity,
    "mod_hybrid": hybrid_modularity,
    "balanced_mass": balanced_sum_mass,
}

runner_types = {
    "tilted": fast_tilted_run3,
    "short_burst": fast_short_burst2, 
}

matrix_scores = ["good", "notbad"] #for the sous-chef, all scores are assumed to be matrix scores.

def thechef( #let him cook
        path_to_data:str,
        run_type:str,
        score1:str,
        score2:str,
        matrix1:str = None,
        matrix2:str = None,
        k: int = 3,
        output_file: click.Path = "./outputs/best/chef_results.jsonl"
):
    with open(path_to_data, 'rb') as file:
        loaded = pickle.load(file)
    profile = loaded['profile']
    boost = loaded['boost'] #probs don't need this
    starting_partition = forward_convert(random_partition(list(profile.candidates), k), profile.candidates)
    if score1 in matrix_scores:
        if matrix1 is None: 
            warnings.warn(f"Matrix type not specified for score {score1}. Defaulting to 'ADJ'.")
            matrix1 = "ADJ"
        M1 = matrix_types[matrix1](profile)
        score_fn1 = score_types[score1](M1, example_partition8=starting_partition, matrix_name=matrix1)
    else:
        score_fn1 = score_types[score1](profile, k=k)
    if score2 in matrix_scores:
        if matrix2 is None: 
            warnings.warn(f"Matrix type not specified for score {score2}. Defaulting to 'ADJ'.")
            matrix2 = "ADJ"
        M2 = matrix_types[matrix2](profile)
        score_fn2 = score_types[score2](M2, example_partition8=starting_partition, matrix_name=matrix2)
    else:
        score_fn2 = score_types[score2](profile, k=k)

    combined_score = calibrate_scores([score_fn1, score_fn2], starting_partition)
    if run_type == "tilted":
        best = fast_tilted_run3(
            starting_partition,
            combined_score,
            iterations=1000000,  #how much is a schmillion?
        )
    elif run_type == "short_burst":
        best = fast_short_burst2(
            starting_partition,
            combined_score,
            burst_size=50,
            num_bursts=20000  # 50*20,000 = 1 (sch)million
        )
    else:
        raise ValueError(f"Unknown run type: {run_type}. Must be one of {list(runner_types.keys())}.")
    #save best as the last line of a jsonl, and also record combined_score.score_name
    with jl.open(output_file, "w") as writer:
        writer.write({
            "partition": backward_convert(best, profile.candidates),
            "score1": score_fn1.score_name,
            "score2": score_fn2.score_name,
            "combined_score": combined_score.score_name,
            "matrix1": matrix1,
            "matrix2": matrix2,
        })

def the_sous_chef( #when we don't take linear combinations of two scores
        path_to_data:str,
        run_type:str,
        score:str,
        matrix:str,
        k: int = 3,
        #output_file: click.Path = "./outputs/best/sous_chef_results.jsonl" #currently the chains themselves are in charge of exporting their results
):
    with open(path_to_data, 'rb') as file:
        loaded = pickle.load(file)
    profile = loaded['profile']
    starting_partition = np.array([k-1 for _ in profile.candidates], dtype=np.int8)
    M1 = matrix_types[matrix](profile)
    score_fn = score_types[score](M1, example_partition8=starting_partition, matrix_name=matrix)
    print(f"Running sous-chef with score function: {score_fn.score_name}")
    if run_type == "tilted":
        best = fast_tilted_run3(
            starting_partition,
            score_fn,
            iterations=1000000,  #how much is a schmillion?
        )
    elif run_type == "short_burst":
        best = fast_short_burst2(
            starting_partition,
            score_fn,
            burst_size=10,
            num_bursts=1000  # 50*20,000 = 1 (sch)million
        )
    else:
        raise ValueError(f"Unknown run type: {run_type}. Must be one of {list(runner_types.keys())}.")
    #save best as the last line of a jsonl, and also record combined_score.score_name
    

# ==============================================#
# Now make a little CLI to wrap this function in
# ==============================================


@click.command()
@click.option(
    "--path-to-data",
    type=click.Path(exists=True),
    help="Path to the input data file (pickle format).",
)
@click.option(
    "--run-type",
    type=click.Choice(list(runner_types.keys()), case_sensitive=False),
    default="tilted",
    help="The type of run to perform (e.g., 'tilted' or 'short_burst').",
)
@click.option("--score",
    type=click.Choice(list(score_types.keys()), case_sensitive=False),
    default="cut",
    help="The first score type to use for the run.",
)
@click.option("--matrix",
    type=click.Choice(list(matrix_types.keys()), case_sensitive=False),
    default=None,
    help="The matrix type to use for the first score (if applicable).",
)
@click.option("--k",
    type=int,
    default=3,
    help="The number of clusters to use for the partitioning (default is 3).",
)

def the_sous_chef_cli(
    path_to_data: str,
    run_type: str,
    score: str,
    matrix: str,
    k: int,
):
    the_sous_chef(
        path_to_data=path_to_data,
        run_type=run_type,
        score=score,
        matrix=matrix,
        k=k,
    )

def thechef_cli(
    path_to_data: str,
    run_type: str,
    score1: str,
    score2: str,
    matrix1: str,
    matrix2: str,
    k: int,
    output_file: click.Path,
):
    thechef(
        path_to_data=path_to_data,
        run_type=run_type,
        score1=score1,
        score2=score2,
        matrix1=matrix1,
        matrix2=matrix2,
        k=k,
        output_file=output_file,
    )
#def run_election_cli(
#    ballot_generator: str,
#    num_voters: int,
#    num_seats: int,
#    election_type: str,
#    ballot_generator_kwargs_settings_file: str,
#    num_iterations: int,
#    output_file: click.Path,
#):
#    with open(ballot_generator_kwargs_settings_file, "r") as f:
#        ballot_generator_kwargs = json.load(f)

#    if ballot_generator_kwargs is None:
#        ballot_generator_kwargs = {}

#    run_election(
#        ballot_generator=ballot_generator,
#        num_voters=num_voters,
#        num_seats=num_seats,
#        ballot_generator_kwargs=ballot_generator_kwargs,
#        election_type=election_type,
#        num_iterations=num_iterations,
#        output_file=output_file,
#    )


if __name__ == "__main__":
    the_sous_chef_cli()
