#!/bin/bash
# /bin/hostname
. ~/torch_env/bin/activate

# nvidia-smi -L
# DIR=/home/shjeong/deepops/workloads/examples/slurm/examples
# echo $DIR
pythonscript="./mnist/main.py"
# python3 "$pythonscript"
array=(64 128 256 512 1024 2048)

for i in "${array[@]}"
do 
    python "$pythonscript" "--batch-size" "$i"
    sleep 12
done

echo "done"