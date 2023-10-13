from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


class JobLogging:
    def __init__(self, batch_size):
        import socket
        # self.total_epoch = total_epoch
        # self.current_epoch = 1
        # self.total_iteration = total_iteration
        self.current_iteration = 1
        self.gpu_memory = 0
        self.gpu_usage = 0
        self.batch_size = batch_size
        self.job_name = 'time_sequence_prediction'
        self.hostname = socket.gethostname()
        self.gpu = torch.cuda.get_device_name(torch.cuda.current_device())
        self.file = './out.txt'
        # f = open(file, 'a').close()

    def logging(self):
        import nvidia_smi

        nvidia_smi.nvmlInit()
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        self.gpu_memory = "{}".format(info.used)
        # self.gpu_memory = "{:.3f}".format(100 * (info.used / info.total))
        self.gpu_usage = res.gpu
        nvidia_smi.nvmlShutdown()

        f = open(self.file, 'a')
        hostname = 'server : ' + self.hostname
        job_name = 'job : ' + self.job_name
        gpu = 'gpu : ' + self.gpu
        gpu_memory = "gpu_memory : " + str(self.gpu_memory) + "MiB"
        gpu_usage = "gpu_util : " + str(self.gpu_usage) + "%"

        batch_size = "batch_size : " + str(self.batch_size)
        current_iteration = 'current_iteration: ' + str(self.current_iteration) + '\n'

        data = '%s, %s, %s, %s, %s, %s, %s' \
            %(hostname, job_name, gpu, gpu_memory, gpu_usage, batch_size, current_iteration)

        f.write(data)
        f.close()

    def change_iteration(self, iteration):
        self.current_iteration = iteration

def input_start_signal():
    import sys
    print('ready', flush = True)
    s = sys.stdin.readline()

def init_schedule(log_collect):
    import schedule
    import threading
    schedule.every(10).seconds.do(log_collect.logging)
    schedule_thread = threading.Thread(target= start_schedule, daemon=True)
    schedule_thread.start()

def start_schedule():
    import os
    import signal
    import schedule
    time.sleep(10)
    schedule.run_pending()
    os.kill(os.getpid(), signal.SIGUSR1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=15, help='steps to run')
    opt = parser.parse_args()
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])
    # build the model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    log_collect = JobLogging(args.batch_size) ####################
    input_start_signal()
    init_schedule(log_collect)

    #begin to train
    for i in range(opt.steps):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf'%i)
        plt.close()
