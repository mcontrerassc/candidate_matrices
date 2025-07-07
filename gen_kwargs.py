import jsonlines

runner_types = ["tilted", "short_burst"]
score_types = ["cut", "rel", "good", "notbad"]
matrix_types = ["ADJ", "PSM"]
matrix_scores = ["good", "notbad"]

datafile = "./data/Portland_D4.pkl"
k = 3

with jsonlines.open("schmillion_params.jsonl", mode="w") as writer:
    for run in runner_types:
        for s1 in score_types:
            for s2 in score_types:
                # cases
                if s1 in matrix_scores and s2 in matrix_scores:
                    for m1 in matrix_types:
                        for m2 in matrix_types:
                            writer.write({
                                "run_type": run,
                                "score1": s1,
                                "score2": s2,
                                "matrix1": m1,
                                "matrix2": m2,
                                "k": k,
                                "path_to_data": datafile
                            })
                elif s1 in matrix_scores:
                    for m1 in matrix_types:
                        writer.write({
                            "run_type": run,
                            "score1": s1,
                            "score2": s2,
                            "matrix1": m1,
                            "matrix2": None,
                            "k": k,
                            "path_to_data": datafile
                        })
                elif s2 in matrix_scores:
                    for m2 in matrix_types:
                        writer.write({
                            "run_type": run,
                            "score1": s1,
                            "score2": s2,
                            "matrix1": None,
                            "matrix2": m2,
                            "k": k,
                            "path_to_data": datafile
                        })
                else:
                    writer.write({
                        "run_type": run,
                        "score1": s1,
                        "score2": s2,
                        "matrix1": None,
                        "matrix2": None,
                        "k": k,
                        "path_to_data": datafile
                    })
print("schmillion_params.jsonl created.")
