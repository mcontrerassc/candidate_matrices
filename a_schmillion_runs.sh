#!/usr/bin/env bash

mkdir -p logs

while IFS= read -r line; do
    # extract each value using python -c
    run_type=$(python -c "import json; print(json.loads('$line')['run_type'])")
    score=$(python -c "import json; print(json.loads('$line')['score'])")
    matrix=$(python -c "import json; print(json.loads('$line')['matrix'])")
    k=$(python -c "import json; print(json.loads('$line')['k'])")
    path_to_data=$(python -c "import json; print(json.loads('$line')['path_to_data'])")

    log_file="logs/${run_type}_${score}_${matrix}.log"

    python schmillion_cli.py \
      --path-to-data "$path_to_data" \
      --run-type "$run_type" \
      --score "$score" \
      --matrix "$matrix" \
      --k "$k" \
      > "$log_file" 2>&1 &

    # limit concurrency
    while [ "$(jobs -rp | wc -l)" -ge 10 ]; do
        sleep 2
    done

done < schmillion_params.jsonl

wait
echo "All schmillion runs completed."
