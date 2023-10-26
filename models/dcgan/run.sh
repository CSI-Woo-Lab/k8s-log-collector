#!/bin/bash
# /bin/hostname
. ~/torch_env/bin/activate

# nvidia-smi -L

export PYTHONPATH=~/deepops/workloads/examples/slurm/slurm_log_collector/models

pythonscript="./models/dcgan/main.py"



python3 "$pythonscript" "--batch-size" "$1"
sleep 3
echo "done"
sleep 10
