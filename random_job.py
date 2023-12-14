import os
import zmq
import csv
import yaml
import time
import random
import subprocess
import argparse
from threading import Thread


'''
this is python file that decide random jobs and execute in a pod of Kubernetes.
Jobs, batch_size are in config.yaml.
'''
#################### CONFIGURATION #################### 
with open("scripts/config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader) 

parser = argparse.ArgumentParser(description='job select')
parser.add_argument('--gpu', required=True,
                    help='gpu server name')
args = parser.parse_args()
os.environ['PYTHONPATH'] = '/workspace/k8s-log-collector'
#################### CONFIGURATION ####################

# Decide batch size according to gpu specification.
if args.gpu == "8G":
    index = 8
elif args.gpu == "12G":
    index = 12
elif args.gpu == "16G":
    index = 16
else:
    index = 24
while True:

    _job = random.choice(cfg['jobs'])
    _batch_size = random.choice(cfg['batch_size_for_{}'.format(index)][_job])
    # execute run.sh file so that execute python file with batch_size and model.

    train_file = cfg['train_file'][_job].split()
    
    if len(train_file) == 1:
        _cmd = "python3 {} --batch-size {}".format(train_file[0], _batch_size)
    else:
        _cmd = "python3 {} --model {} --batch-size {}".format(train_file[0], train_file[1], _batch_size)

    # _cmd = "./scripts/run.sh {} {}".format(cfg['train_file'][_job], _batch_size)
    _proc = subprocess.Popen(_cmd, shell=True, text=True)
    _proc.wait()

    time.sleep(5)
    _proc.kill()
