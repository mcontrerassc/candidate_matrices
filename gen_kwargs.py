import jsonlines

runner_types = ["short_burst"]

score_types = [
    "good",
    "sum_mass",
    "mod_standard",
    "mod_hybrid",
    "balanced_mass",
]

matrix_types = ["ADJ", "PSM", "boost", "FPV"]

# Mapping of which scores are allowed with which matrices
allowed_matrices_by_score = {
    "good": {"ADJ", "PSM"},
    "balanced_mass": {"ADJ", "PSM"},
    "sum_mass": {"boost", "FPV"},
    "mod_hybrid": {"boost", "FPV"},
    "mod_standard": {"ADJ", "PSM", "boost", "FPV"}, 
}

datafile = "./data/Portland_D4.pkl"
k = 3

with jsonlines.open("schmillion_params.jsonl", mode="w") as writer:
    for run in runner_types:
        for score in score_types:
            legal_matrices = allowed_matrices_by_score.get(score, set(matrix_types))
            for matrix in legal_matrices:
                writer.write({
                    "run_type": run,
                    "score": score,
                    "matrix": matrix,
                    "k": k,
                    "path_to_data": datafile
                })

print("schmillion_params.jsonl created.")
