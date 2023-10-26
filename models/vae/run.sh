#!/bin/bash
. ~/torch_env/bin/activate

pythonscript="./models/vae/main.py"

python3 "$pythonscript" "--batch-size" "$1"