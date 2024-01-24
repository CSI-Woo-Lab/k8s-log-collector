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
os.environ['PYTHONPATH'] += ':/workspace/k8s-log-collector/models/OfflineRL-Kit'
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin'
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
    addr = "115.145.175.74"
    _job = random.choice(cfg['jobs'])
    _batch_size = random.choice(cfg['batch_size_for_{}'.format(index)][_job])
    dataset = random.choice(cfg['datasets'][_job])
    # num_iteration in epoch
    # epoch = random.choice([0,1,2])
    image_size = random.choice(cfg['batch_size_for_{}'.format(index)]['image_size'])
    workers = random.choice(cfg['batch_size_for_{}'.format(index)]['workers'])
    # execute run.sh file so that execute python file with batch_size and model.

    train_file = cfg['train_file'][_job].split()
    
    if _job == "mnist":
        if args.gpu == "8G":
            rank = '0'
        else:
            rank = '1'
        _cmd = "python3 -m torch.distributed.launch --nnodes {} --nproc_per_node {} --node_rank {} --master_addr {} --master_port {} {}".format(
            '2', '1', rank, addr, '2024', train_file[0])
    elif len(train_file) == 1:
        if dataset == 'mujoco_env':
            dataset = random.choice(cfg["mujoco_env"])
        _cmd = "python3 {} --batch-size {} --dataset {} --image-size {} --workers {}".format(train_file[0], _batch_size, dataset, image_size, workers)
    else:
        _cmd = "python3 {} --model {} --batch-size {} --dataset {} --image-size {} --workers {}".format(train_file[0], train_file[1], _batch_size)

    # _cmd = "./scripts/run.sh {} {}".format(cfg['train_file'][_job], _batch_size)
    _proc = subprocess.Popen(_cmd, shell=True, text=True)
    try:
        _proc.wait()
    except subprocess.TimeoutExpired:
        kill(_proc.pid)
    try:
        kill(_proc.pid)
        time.sleep(5)
    except:
        time.sleep(5)
    kill_zombie()
