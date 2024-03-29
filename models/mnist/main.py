from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os

def init_for_distributed(args):
    # 1. setting for distributed training
    args.global_rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(args.local_rank)
    if args.global_rank is not None and args.local_rank is not None:
        print("Use GPU: [{}/{}] for training".format(args.global_rank, args.local_rank))

    # 2. init_process_group
    dist.init_process_group(backend="nccl")
    # if put this function, the all processes block at all.
    torch.distributed.barrier()
    # convert print fn iif rank is zero
    return

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


def train(args, model, device, train_loader, optimizer, epoch, x):
    model.train()
    length = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # total iteration increased by one after each iterations ended.
        ######### MINGEUN ###########
        x.every_iteration()
        ######### MINGEUN ###########
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        #     if args.dry_run:
        #         break


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

    ######### MINGEUN ###########
    parser.add_argument('--dataset', default='mnist', help='used dataset')
    parser.add_argument('--image-size', default='64', help='size of image for training if used')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--gpu_ids', nargs="+", default=['0'])
    parser.add_argument('--port', type=int, default=2024)
    # parser.add_argument('--world_size', type=int, default=24)
    parser.add_argument('--global_rank', type=int, default=0)
    ######### MINGEUN ###########

    args = parser.parse_args()
    init_for_distributed(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    args.seed = np.random.randint(1,10000)
    torch.manual_seed(args.seed)

    # logger model load
    ######### MINGEUN ###########
    from logger import Logger
    args.image_size = 28
    x = Logger("mnist_cnn", args.batch_size, args.dataset, args.image_size, args.workers) 
    ######### MINGEUN ###########

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': args.workers,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../datasets', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../datasets', train=False,
                       transform=transform)

    ########## MINGEUN ##########
    train_sampler = DistributedSampler(dataset=dataset1, shuffle=True)
    test_sampler = DistributedSampler(dataset=dataset2, shuffle=True)

    DDP_kwargs1 = {
        'sampler' : train_sampler,
        'shuffle' : False
    }
    DDP_kwargs2 = {
        'sampler' : test_sampler,
        'shuffle' : False
    }
    train_kwargs.update(DDP_kwargs1)
    test_kwargs.update(DDP_kwargs2)
    ########## MINGEUN ##########
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    model = DDP(module=model, device_id=[args.local_rank])
    ########## MINGEUN ##########


    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # logger wait until messeage received from control node. 
    ######### MINGEUN ###########
    torch.cuda.empty_cache()
    x.ready_for_training()
    ######### MINGEUN ###########

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, x)
        # test(model, device, test_loader)
        scheduler.step()

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()

