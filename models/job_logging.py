import os
import signal
import schedule
import time
import threading
import sys
import nvidia_smi
import socket
import numpy as np
import torch


class JobLogging:
    def __init__(self, batch_size, job_name):
        # self.total_epoch = total_epoch
        self.current_epoch = 1
        # self.total_iteration = total_iteration
        self.current_iteration = 1
        self.gpu_memory = []
        self.gpu_memory_total = 0
        self.gpu_util = []
        self.batch_size = batch_size
        self.job_name = job_name
        self.hostname = socket.gethostname()
        self.gpu = torch.cuda.get_device_name(torch.cuda.current_device())
        # self.file = './out.txt'
        # f = open(file, 'a').close()

    def logging(self):
        gpu_memory_mean = np.mean(self.gpu_memory)
        gpu_util_mean = np.mean(self.gpu_util)
        self.gpu_memory_percent = "{:.2f}".format(100 * (gpu_memory_mean / self.gpu_memory_total))
        print('length:', len(self.gpu_memory), flush=True)
        # self.gpu_util = res.gpu
        # f = open(self.file, 'a')
        hostname = self.hostname
        job_name = self.job_name
        gpu = self.gpu
        gpu_memory = str(round(gpu_memory_mean, 2)) + "MiB"
        gpu_memory_percent = str(self.gpu_memory_percent) + "%"
        gpu_util = str(round(gpu_util_mean, 2)) + "%"

        batch_size = str(self.batch_size)
        current_iteration = self.current_iteration

        data = '%s,%s,%s,%s,%s,%s,%s,%d' \
            %(hostname, job_name, gpu, gpu_memory, gpu_memory_percent, gpu_util, batch_size, current_iteration)
    
        print(data, flush=True)
        # f.write(data)
        # f.close()

    def log_gpu(self):
        nvidia_smi.nvmlInit()
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        self.gpu_util.append(res.gpu)
        self.gpu_memory.append(info.used)
        self.gpu_memory_total = info.total
        nvidia_smi.nvmlShutdown()

    def change_iteration(self, iteration):
        self.current_iteration = iteration

def input_start_signal():
    print('ready', flush=True)
    s = sys.stdin.readline()

def init_schedule(log_collect):
    schedule.every(0.5).seconds.do(log_collect.log_gpu)
    schedule.every(10).seconds.do(log_collect.logging)
    schedule_thread = threading.Thread(target= start_schedule, daemon=True)
    schedule_thread.start()

def start_schedule():
    n = 0
    while n <= 20:
        time.sleep(0.5)
        schedule.run_pending()
        n += 1
    os.kill(os.getpid(), signal.SIGUSR1)
