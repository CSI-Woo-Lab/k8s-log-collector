#!/bin/bash
/bin/hostname
. ~/torch_env/bin/activate

# nvidia-smi -L

pythonscript="./mnist_rnn/main.py"

python3 "$pythonscript" "--batch-size" "$1"
sleep 3
echo "done"
sleep 10
