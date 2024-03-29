from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='imagenet', help='cifar10 | lsun | mnist |imagenet | folder | lfw | coco | fake ')
parser.add_argument('--dataroot', required=False, default='../datasets', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch-size', type=int, default=64, help='input batch size')
parser.add_argument('--image-size', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--mps', action='store_true', default=False, help='enables macOS GPU training')
parser.add_argument('--epoch', type=int, default=0, help='num_iteration in epoch')
opt = parser.parse_args()

opt.workers=16

# logger model load
######### MINGEUN ###########
from logger import Logger
opt.image_size = 64
x = Logger("dcgan", opt.batch_size, opt.dataset, opt.image_size, opt.workers) 
######### MINGEUN ###########


try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if torch.backends.mps.is_available() and not opt.mps:
    print("WARNING: You have mps device, to enable macOS GPU run with --mps")
  
if opt.dataroot is None and str(opt.dataset).lower() != 'fake':
    raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % opt.dataset)

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    from ImageNetDataset import ImageNetDataset
    opt.dataroot = "../datasets/ImageNet/train"
    dataset = ImageNetDataset(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.image_size),
                                   transforms.CenterCrop(opt.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3

elif opt.dataset == 'imagenet_preprocess':
    from ImageNetDataset import ImageNetDataset
    opt.dataroot = "../datasets/ImageNet/train_crop"
    dataset = ImageNetDataset(root=opt.dataroot,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                              ]))
    nc=3

elif opt.dataset == 'lsun':
    opt.classes = 'bedroom,bridge,church_outdoor,conference_room,tower,restaurant,dining_room,classroom,kitchen,living_room'
    # opt.classes = 'tower'
    opt.dataroot = "../datasets/ImageNet/lsun"
    classes = [ c + '_train' for c in opt.classes.split(',')]
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(opt.image_size),
                            transforms.CenterCrop(opt.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    from ImageNetDataset import ImageNetDataset
    from CIFAR10Dataset import CIFAR10Dataset
    # dataset = dset.CIFAR10(root=opt.dataroot, download=True,)
    dataset = CIFAR10Dataset(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif opt.dataset == 'cifar10_preprocess':
    from ImageNetDataset import ImageNetDataset
    opt.dataroot = '../datasets/CIFAR10/train'
    dataset = ImageNetDataset(root=opt.dataroot,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                              ]))
    nc=3

elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                        transform=transforms.Compose([
                            transforms.Resize(opt.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ]))
    nc=1

elif opt.dataset == 'coco':
    from CocoDataset import CocoDataset
    opt.dataroot = "../datasets/coco/train2017"
    # opt.dataroot = "../datasets/ImageNet/coco/train2017"
    opt.image_size = (64, 64)
    dataset = CocoDataset(root=opt.dataroot, annFile = "../datasets/coco/annotations/instances_train2017.json",
                           transform=transforms.Compose([
                               transforms.Resize(opt.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
                        )
    nc=3

elif opt.dataset == 'coco_preprocess':
    from ImageNetDataset import ImageNetDataset
    opt.dataroot='../datasets/coco/train2017_crop'
    dataset = ImageNetDataset(root=opt.dataroot,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                              ]))
    nc=3

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.image_size, opt.image_size),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=int(opt.workers))
use_mps = opt.mps and torch.backends.mps.is_available()
if opt.cuda:
    device = torch.device("cuda:0")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
# print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batch_size, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# if opt.dry_run:
#     opt.niter = 1

# logger wait until messeage received from control node. 
######### MINGEUN ###########
torch.cuda.empty_cache()
x.ready_for_training()
num_iteration = 0

should_break = False
def exit_program():
    import time
    global should_break
    should_break = True
    print("exit the program.")
    print("num_iteration :", num_iteration)
    x.terminate()
    
######### MINGEUN ###########
import threading
from tqdm import tqdm
for epoch in range(opt.niter):
    # print("epoch:", epoch)
    # print("terminate_epoch:", opt.epoch)
    if epoch == opt.epoch:
        # print("timer start!")
        timer = threading.Timer(10, exit_program)
        timer.start()

    if should_break:
        break
    for i, data in enumerate(tqdm(dataloader)):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label,
                           dtype=real_cpu.dtype, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # total iteration increased by one after each iterations ended.
        ######### MINGEUN ###########
        if epoch >= opt.epoch:
            x.every_iteration()
            num_iteration += 1
        ######### MINGEUN ###########

        # print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
            #   % (epoch, opt.niter, i, len(dataloader),
            #      errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        # if i % 100 == 0:
        #     vutils.save_image(real_cpu,
        #             '%s/real_samples.png' % opt.outf,
        #             normalize=True)
        #     fake = netG(fixed_noise)
        #     vutils.save_image(fake.detach(),
        #             '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
        #             normalize=True)

        # if opt.dry_run:
        #     break
    # do checkpointing
    # torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    # torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
