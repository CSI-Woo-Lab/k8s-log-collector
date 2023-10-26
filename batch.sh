#!/bin/bash

srun --nodelist=gpu01 --gres=gpu:gpu0:1 nvidia-smi