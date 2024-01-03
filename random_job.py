import os
import zmq
import csv
import yaml
import time
import random
import subprocess
import psutil
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

def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

def kill_zombie():
    for proc in psutil.process_iter(['pid', 'status', 'memory_percent']):
        if proc.info['memory_percent'] is not None and \
                proc.info['memory_percent'] >= 0.0 and \
                proc.info['status'] == psutil.STATUS_SLEEPING:
            pid = proc.info['pid']
            os.kill(pid, 9)

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
    
    if _job == "offline_RL":
        _offlineRL_job = random.choice(cfg['offlineRL_model'])
        task = random.choice(cfg['mujoco_env'])
        offlineRL_train_file = cfg[train_file[0]][_offlineRL_job]
        _cmd = "python3 {} --task {} --batch-size {}".format(offlineRL_train_file, task, _batch_size)
    elif len(train_file) == 1:
        _cmd = "python3 {} --batch-size {}".format(train_file[0], _batch_size)
    else:
        _cmd = "python3 {} --model {} --batch-size {}".format(train_file[0], train_file[1], _batch_size)

    # _cmd = "./scripts/run.sh {} {}".format(cfg['train_file'][_job], _batch_size)
    _proc = subprocess.Popen(_cmd, shell=True, text=True)
    try:
        _proc.wait(timeout=1000)
    except subprocess.TimeoutExpired:
        kill(_proc.pid)
    try:
        kill(_proc.pid)
        time.sleep(5)
    except:
        time.sleep(5)
    kill_zombie()
