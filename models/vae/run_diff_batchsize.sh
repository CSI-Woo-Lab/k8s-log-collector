#!/bin/bash
# /bin/hostname
. ~/torch_env/bin/activate

# nvidia-smi -L
# DIR=/home/shjeong/deepops/workloads/examples/slurm/examples
# echo $DIR
pythonscript="./vae/main.py"
# python3 "$pythonscript"

array=(128 256 512 1024 2048 4096)
for i in "${array[@]}"
do 
    python "$pythonscript" "--batch-size" "$i"
    sleep 12
done

echo "done"

# DIR=$PWD
# echo $DIR
# pythonscript="$DIR/main.py"
# python3 "$pythonscript"