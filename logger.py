import os
import zmq
import time
import torch
import socket
import threading
import nvidia_smi
import numpy as np
from zmq.asyncio import Context

class Logger():
    """
        Logger for job information after training.
        It records node name, gpu name, job name, batch size, gpu utilization, gpu memory, iteration.
        When it comes time to communicate with control node, it sends logs and terminates jobs. 
    """

    def __init__(self, job_name, batch_size):
        self.gpu_util_log = []
        self.gpu_mem_log = []
        self.gpu_mem_total = 0
        self.iteration = 0
        self.job_name = job_name
        self.batch_size = batch_size
        self.logging_interval = 0
        self.communication_interval = 0
        self.times = 0
        self.hostname = socket.gethostname()
        self.gpuname = torch.cuda.get_device_name(torch.cuda.current_device())

        # make TCP communication socket between control node and gpu node using zmq 
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.connect("tcp://115.145.175.74:9000")

    def ready_for_training(self):

        # gpu node sends 'ready' to control node.
        self.socket.send_string(f"ready")
        # when control node receive 'ready' from all gpu node, 
        # it sends logging times, communication_interval, logging_interval.
        data = self.socket.recv_multipart()  # from msg, we got self.times, communication_interval, logging_interval.
        self.times = int(data[0].decode("utf-8").split()[1])
        self.communication_interval = float(data[0].decode("utf-8").split()[2])
        self.logging_interval = float(data[0].decode("utf-8").split()[3])

        def log_gpu():
            nvidia_smi.nvmlInit()
            deviceCount = nvidia_smi.nvmlDeviceGetCount()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            self.gpu_util_log.append(res.gpu)
            self.gpu_mem_log.append(info.used)
            self.gpu_mem_total = info.total
            nvidia_smi.nvmlShutdown()

            # log gpu memory, gpu utilization according to logging_interval.
            self.gpu_logging_timer = threading.Timer(self.logging_interval, log_gpu)             
            self.gpu_logging_timer.start()

        def send_data():
            gpu_util = np.mean(self.gpu_util_log)
            gpu_mem_util = np.mean(self.gpu_mem_log)
            data = f'%s,%s,%s,%d,%s,%s,%s,%d' \
                % (self.hostname, self.gpuname, self.job_name, self.batch_size, gpu_util, gpu_mem_util, self.gpu_mem_total, self.iteration)
            # send log data to control node
            self.socket.send_string(data)
            self.times -= 1
            if self.times == 0:
                # kill the job
                self.socket.recv_multipart()
                self.gpu_logging_timer.cancel()
                self.server_send_timer.cancel()
                torch.cuda.empty_cache()
                os.kill(os.getpid(), 9)

            self.reset_log_buffer()
            self.server_send_timer = threading.Timer(self.communication_interval, send_data)
            self.server_send_timer.start()
        
        self.gpu_logging_timer = threading.Timer(self.logging_interval, log_gpu)
        self.server_send_timer = threading.Timer(self.communication_interval, send_data)
        self.gpu_logging_timer.start()
        self.server_send_timer.start()

    def every_iteration(self):
        # add 1 to total iteration at the end of each iteration.
        self.iteration += 1

    def reset_log_buffer(self):
        # reaset the log_buffer.
        self.gpu_util_log = []
        self.gpu_mem_log = []
        self.gpu_memory_total = 0
        self.iteration = 0