import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.strategies import DDPStrategy

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='models/PyTorch-VAE/configs/vae.yaml')
# custom batch_size is received
############ MINGEUN ############
parser.add_argument('--batch-size', type=int, default = 64, metavar='N',
                    help = 'batch_size of training and eval')
############ MINGEUN ############
args = parser.parse_args()

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# logger model and batch_size load
# random seed is selected 
########### MINGEUN ############
from logger import Logger
x = Logger("Pytorch_VAE", args.batch_size)
flag = 0
config['data_params']['train_batch_size'] = args.batch_size
config['data_params']['val_batch_size'] = args.batch_size
config['exp_params']['manual_seed'] = np.random.randint(10000)
########### MINGEUN ############

# callbackfunction
############# MINGEUN ##################
from pytorch_lightning.callbacks import Callback
class MyIterationCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global x
        # total iteration increased by one after each iterations ended.
        x.every_iteration()
    def on_train_epoch_start(self, trainer, pl_module):
        global x
        global flag
        # wait training start at first epoch only
        if flag == 0:
            flag += 1
            # logger wait until messeage received from control node. 
            x.ready_for_training()
            
############# MINGEUN ##################

tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                    #  ModelCheckpoint(save_top_k=2, 
                    #                  dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                    #                  monitor= "val_loss",
                    #                  save_last= True),
                     MyIterationCallback(),
                 ],
                 strategy=DDPStrategy(find_unused_parameters=False),
                 **config['trainer_params'])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


# print(f"======= Training {config['model_params']['name']} =======")

runner.fit(experiment, datamodule=data)
