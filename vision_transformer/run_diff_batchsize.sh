#!/bin/bash
# /bin/hostname
. ~/torch_env/bin/activate

# nvidia-smi -L
# DIR=/home/shjeong/deepops/workloads/examples/slurm/examples
# echo $DIR
pythonscript="./vision_transformer/main.py"
# python3 "$pythonscript"
for i in {4..30}
do 
    python "$pythonscript" "--batch-size" "$i"
    sleep 12
done

echo "done"