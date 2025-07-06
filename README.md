
# README

## Installing Required Packages

To install the required packages listed in `requirements.txt`, run:

    pip install -r requirements.txt

## Running a Schmillion Experiments

This repository provides a parallel experiment runner to execute many variants of `thechef` routine with different scoring functions, runner types, and matrix configurations. The workflow is broken into two steps:

---

### 1. Generate the argument combinations

The script `generate_kwargs.py` systematically enumerates all valid argument combinations for:
- runner type (`tilted` or `short_burst`)
- score functions (`cut`, `rel`, `good`, `notbad`)
- matrix types (`ADJ`, `PSM`)

It writes these combinations into a JSONL file (`schmillion_params.jsonl`), with each line representing one run configuration.

To regenerate the combinations, simply run:

    python generate_kwargs.py

If you would like to change:
- the input data file (currently `./data/Portland_D4.pkl`)
- the number of clusters `k` (currently `k=3`)

you can edit the `generate_kwargs.py` script directly:

    datafile = "./data/Portland_D4.pkl"
    k = 3

and rerun it.

The length of each chain is currently fixed to get a million steps in total; to change this, edit the `schmillion_cli.py` script. 

---

### 2. Execute all runs in parallel

The script `a_schmillion_run.sh` reads each JSON line from `schmillion_params.jsonl` and launches a separate `schmillion_cli.py` process for it. It respects a concurrency limit (by default, 10 parallel jobs).

To launch the whole sweep, run:

    bash a_schmillion_run.sh

Each individual run will log its output to a `logs/` directory, and its result JSONL will be stored in `outputs/best/`. The chef also stores the best partition from each run in a dictionary, which it stores as a line of `outputs/best/chef_results.jsonl.`

There are viz functions to visualize all of these output formats in `src/viz.py`


---

## Notes

- If you add new scoring functions or matrix types (e.g. modularity, optimized distance to slate), be sure to update the lists inside `generate_kwargs.py`, as well as import them into `schmillion_cli.py`. 
- You can adjust the concurrency limit in `a_schmillion_run.sh` by changing this line:

      while [ "$(jobs -rp | wc -l)" -ge 10 ]; do

  to a different maximum.

- I'm not sure how exactly this does slurm things (maybe that happens on the cluster side?) but this is based on Peter's code.
---
