import os
import zmq
import csv
import yaml
import time
import random
import subprocess
from threading import Thread


#################### CONFIGURATION ####################
with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

num_of_jobs = len(cfg["node_and_gpu"])
times = 2
communication_interval = 10.0
logging_interval = 0.1
filename = "out.csv"
#################### CONFIGURATION ####################


def collecting(num_of_jobs):
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind("tcp://115.145.175.74:9000")

    logs = []
    while True:     # Entire logging loop
        connected_procs = {}
        _tmp = []
        received_times = 0
        while True:     # 1 single log sample
            identi, data = socket.recv_multipart()
            
            if identi not in connected_procs.keys() and data.decode() == "ready":
                connected_procs[identi] = 0
                if len(connected_procs) == num_of_jobs:
                    cmd = 'start {} {} {}'.format(times, communication_interval, logging_interval).encode()
                    for proc in connected_procs.keys():
                        socket.send_multipart([proc, cmd])
            else:
                if identi in connected_procs.keys():
                    _tmp.append(data)
                    connected_procs[identi] += 1
                
                if all(i==(received_times+1) for i in connected_procs.values()):
                    if received_times != 0:
                        logs.append(_tmp)
                    _tmp = []
                    received_times += 1
                    if received_times == times:
                        for proc in connected_procs.keys():
                            socket.send_multipart([proc, b'kill'])
                        for i in range(len(logs[-1])):
                            print(logs[-1][i])
                        
                        with open(filename,'a') as f:
                            wr = csv.writer(f)
                            for line in logs[-1]:
                                wr.writerow(line.decode("utf-8").split(','))
                            wr.writerow([])
                        break


Thread(target=collecting, args=[num_of_jobs]).start()
while True:
    # initialization
    # data format : node, gpu, job, batch_size, gpu_name, gpu_utilization, gpu_memory_utilization, iteration, 
    log = []
    for node, gpu, mem in cfg['node_and_gpu']:
        log.append([node, gpu, mem])

    # select job and hyper parameters
    # TODO : Organize executable jobs separately by gpu
    for j in range(num_of_jobs):
        _job = random.choice(cfg['jobs'])
        # if j == 0 : 
        #     _job = "vision_transformer"
        # if j == 1 :
        #     _job = "vision_transformer"
        # _batch_size = max(cfg['batch_size_for_{}'.format(log[j][2])][_job])
        _batch_size = random.choice(cfg['batch_size_for_{}'.format(log[j][2])][_job])
        log[j] += [_job, _batch_size]

    process_list = []
    for j in range(num_of_jobs):
        # _cmd = "srun --nodelist={} --gres={} ./run.sh {} {}".format(log[j][0], log[j][1], cfg['train_file'][log[j][3]], log[j][4])
        _cmd = "srun --nodelist={} --gres={} --mem=16000 -o /dev/null ./run.sh {} {}".format(log[j][0], log[j][1], cfg['train_file'][log[j][3]], log[j][4])
        _proc = subprocess.Popen(_cmd, shell=True, text=True)
        process_list.append(_proc)
        
    for proc in process_list:
        proc.wait()
    
    time.sleep(3)
