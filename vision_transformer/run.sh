#!/bin/bash
/bin/hostname
. ~/torch_env/bin/activate

nvidia-smi -L
DIR=/home/shjeong/deepops/workloads/examples/slurm/examples
echo $DIR
pythonscript="$DIR/vision_transformer/main.py"
python3 "$pythonscript"

# DIR=$PWD
# echo $DIR
# pythonscript="$DIR/main.py"
# python3 "$pythonscript"

