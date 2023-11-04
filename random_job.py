import os
import zmq
import csv
import yaml
import time
import random
import subprocess
import argparse
from threading import Thread


#################### CONFIGURATION ####################
with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser(description='job select')
parser.add_argument('--gpu', required=True,
                    help='gpu server name')
args = parser.parse_args()
#################### CONFIGURATION ####################

if args.gpu == "gpu01-gpu0":
    index = 8
else:
    index = 24
while True:

    _job = random.choice(cfg['jobs'])
    _batch_size = random.choice(cfg['batch_size_for_{}'.format(index)][_job])

    _cmd = "./run.sh {} {}".format(cfg['train_file'][_job], _batch_size)
    # _cmd = "srun --nodelist={} --gres={} -o /dev/null ./run.sh {} {}".format(log[j][0], log[j][1], cfg['train_file'][log[j][2]], log[j][3])
    _proc = subprocess.Popen(_cmd, shell=True, text=True)
    
    _proc.wait()
