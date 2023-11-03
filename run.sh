#!/bin/bash
# activate virtualenv for torch examples
. ~/torch_env/bin/activate

export PYTHONPATH=~/slurm_log_collector

# run train code with batch_size parameter

if (($# > 2)); then
    python3 "$1" "--model" "$2" "--batch-size" "$3"
else
    python3 "$1" "--batch-size" "$2"

fi
