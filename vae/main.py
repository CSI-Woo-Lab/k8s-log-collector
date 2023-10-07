from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import sys
import schedule
import time
path = '/home/shjeong/deepops/workloads/examples/slurm/examples/vae/'
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         path + 'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


class JobLogging:
    def __init__(self, batch_size, total_epoch, total_iteration):
        import socket
        self.total_epoch = total_epoch
        self.current_epoch = 1
        self.total_iteration = total_iteration
        self.current_iteration = 1
        self.gpu_memory = 0
        self.gpu_memory2 = 0
        self.gpu_usage = 0
        self.batch_size = batch_size
        self.job_name = 'vision_transformer'
        self.hostname = socket.gethostname()
        self.gpu = torch.cuda.get_device_name(torch.cuda.current_device())
        file = path+'/out.txt'
        f = open(file, 'w').close()

    def logging(self):
        import nvidia_smi

        nvidia_smi.nvmlInit()
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        self.gpu_memory = "{:.3f}".format(100 * (info.used / info.total))
        self.gpu_memory2 = res.memory
        self.gpu_usage = res.gpu
        nvidia_smi.nvmlShutdown()

        file = path+'/out.txt'
        f = open(file, 'a')
        hostname = 'server : ' + self.hostname
        job_name = 'job : ' + self.job_name
        gpu = 'gpu : ' + self.gpu
        gpu_memory = "gpu_memory : " + str(self.gpu_memory) + "%"
        gpu_memory2 = "gpu_memory2 :" + str(self.gpu_memory2) + "%"
        gpu_usage = "gpu_usage : " + str(self.gpu_usage) + "%"

        batch_size = "batch_size : " + str(self.batch_size)

        total_epoch = 'total_epoch : ' + str(self.total_epoch)
        current_epoch = 'current_epoch : ' + str(self.current_epoch)

        total_iteration = 'total_iteration : ' + str(self.total_iteration)
        current_iteration = 'current_iteration: ' + str(self.current_iteration) + '\n'

        data = '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s' \
            %(hostname, job_name, gpu, gpu_memory, gpu_memory2, gpu_usage, batch_size, total_epoch, current_epoch, total_iteration, current_iteration)

        f.write(data)
        f.close()
    
    def change_epoch(self, epoch):
        self.current_epoch = epoch
    def change_iteration(self, iteration):
        self.current_iteration = iteration

if __name__ == "__main__":
    s = sys.stdin.readline()
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        # test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       path + 'results/sample_' + str(epoch) + '.png')
