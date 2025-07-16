import json

with open("schmillion_params.jsonl") as f:
    for line in f:
        job = json.loads(line)
        args = [
            job["run_type"],
            job["score"],
            job["matrix"],
            str(job["k"]),
            job["path_to_data"],
        ]
        print(" ".join(args))
