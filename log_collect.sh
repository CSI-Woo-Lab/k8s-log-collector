#!/bin/bash
/bin/hostname
DIR=$PWD
echo $DIR
pythonscript="$DIR/log_collect.py"
python3 "$pythonscript"

