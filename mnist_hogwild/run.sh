#!/bin/bash
/bin/hostname
. ~/torch_env/bin/activate

nvidia-smi -L
DIR=$PWD
echo $DIR
pythonscript="$DIR/main.py"
python3 "$pythonscript"

