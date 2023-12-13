#!/bin/bash
# export PYTHONPATH=/workspace/k8s-log-collector
# run train code with batch_size parameter

# yolov8 contains model -> 3 inputs.
if (($# > 2)); then
    python3 "$1" "--model" "$2" "--batch-size" "$3"
# others do not contain models. -> 2 inputs.
else
    python3 "$1" "--batch-size" "$2"
fi

exit