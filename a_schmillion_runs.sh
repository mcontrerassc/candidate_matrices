#!/usr/bin/env bash

mkdir -p logs

while IFS= read -r line; do
    # parse the JSON line
    run_type=$(echo "$line" | jq -r .run_type)
    score1=$(echo "$line" | jq -r .score1)
    score2=$(echo "$line" | jq -r .score2)
    matrix1=$(echo "$line" | jq -r .matrix1)
    matrix2=$(echo "$line" | jq -r .matrix2)
    k=$(echo "$line" | jq -r .k)
    path_to_data=$(echo "$line" | jq -r .path_to_data)

    outfile="./outputs/best/${run_type}_${score1}_${score2}"
    if [ "$matrix1" != "null" ]; then
      outfile="${outfile}_${matrix1}"
    fi
    if [ "$matrix2" != "null" ]; then
      outfile="${outfile}_${matrix2}"
    fi
    outfile="${outfile}.jsonl"

    echo "Launching $outfile"
    python schmillion_cli.py \
      --path-to-data "$path_to_data" \
      --run-type "$run_type" \
      --score1 "$score1" \
      --score2 "$score2" \
      --matrix1 "$matrix1" \
      --matrix2 "$matrix2" \
      --k "$k" \
      -o "$outfile" \
      > "logs/${run_type}_${score1}_${score2}.log" 2>&1 &

    # limit concurrency
    while [ "$(jobs -rp | wc -l)" -ge 10 ]; do
      sleep 2
    done

done < schmillion_params.jsonl

wait
echo "All schmillion runs completed."
