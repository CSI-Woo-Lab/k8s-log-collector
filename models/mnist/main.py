from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, log_collect):
    model.train()
    length = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):

        log_collect.change_iteration(batch_idx + length * (epoch - 1)) #################

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))


class JobLogging:
    def __init__(self, batch_size):
        import socket
        # self.total_epoch = total_epoch
        self.current_epoch = 1
        # self.total_iteration = total_iteration
        self.current_iteration = 1
        self.gpu_memory = 0
        self.gpu_usage = 0
        self.batch_size = batch_size
        self.job_name = 'mnist_cnn'
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
        hostname = self.hostname
        job_name = self.job_name
        gpu = self.gpu
        gpu_memory = str(self.gpu_memory) + "MiB"
        gpu_usage = str(self.gpu_usage) + "%"

        batch_size = str(self.batch_size)
        current_iteration = self.current_iteration

        data = '%s,%s,%s,%s,%s,%s,%d' \
            %(hostname, job_name, gpu, gpu_memory, gpu_usage, batch_size, current_iteration)
    
        print(data, flush=True)
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
    import time
    time.sleep(10)
    schedule.run_pending()
    os.kill(os.getpid(), signal.SIGUSR1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    log_collect = JobLogging(args.batch_size) ####################
    input_start_signal()
    # print(s, flush = True)
    init_schedule(log_collect)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, log_collect)
        # test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
