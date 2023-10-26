#!/bin/bash
# /bin/hostname
. ~/torch_env/bin/activate

# nvidia-smi -L
# DIR=/home/shjeong/deepops/workloads/examples/slurm/examples
# echo $DIR

pythonscript="./models/vision_transformer/main.py"
python3 "$pythonscript" "--batch-size" "$1"
sleep 3
echo "done"
sleep 10