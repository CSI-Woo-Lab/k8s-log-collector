from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from data import get_training_set, get_test_set

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=3, help="super resolution upscale factor")
parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', default=True, help='use cuda?')
parser.add_argument('--mps', action='store_true', default=True, help='enables macOS GPU training')
parser.add_argument('--workers', type=int, default=4, help='number of workers for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
############ MINGEUN ############
parser.add_argument('--dataset', default='bsd300', help='used dataset')
parser.add_argument('--image-size', default='64', help='size of image for training if used')
############ MINGEUN ############

opt = parser.parse_args()

# logger model load
######### MINGEUN ###########
from logger import Logger
opt.upscale_factor = int(256 // opt.image_size)
x = Logger("super_resolution", opt.batch_size, opt.dataset, opt.image_size, opt.workers) 
######### MINGEUN ###########



if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
if not opt.mps and torch.backends.mps.is_available():
    raise Exception("Found mps device, please run with --mps to enable macOS GPU")

torch.manual_seed(opt.seed)
use_mps = opt.mps and torch.backends.mps.is_available()

if opt.cuda:
    device = torch.device("cuda")
    # print("cuda_true!!!!")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.workers, batch_size=opt.batch_size, shuffle=True)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.workers, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
print('device:', device)
model = Net(upscale_factor=opt.upscale_factor).to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def train(epoch, x):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        
        

        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        # total iteration increased by one after each iterations ended.
        ######### MINGEUN ###########
        x.every_iteration()
        ######### MINGEUN ###########

        # print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    # print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

# logger wait until messeage received from control node. 
######### MINGEUN ###########
torch.cuda.empty_cache()
x.ready_for_training()
######### MINGEUN ###########

for epoch in range(1, opt.nEpochs + 1):
    train(epoch, x)
    # test()
    # checkpoint(epoch)
