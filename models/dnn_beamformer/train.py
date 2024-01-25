import pathlib
from argparse import ArgumentParser

import lightning.pytorch as pl
from datamodule import L3DAS22DataModule
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import TensorBoardLogger
from model import DNNBeamformer, DNNBeamformerLightningModule


parser = ArgumentParser()
parser.add_argument(
    "--checkpoint-path",
    default=None,
    type=pathlib.Path,
    help="Path to checkpoint to use for evaluation.",
)
parser.add_argument(
    "--exp-dir",
    default=pathlib.Path("./exp/"),
    type=pathlib.Path,
    help="Directory to save checkpoints and logs to. (Default: './exp/')",
)
parser.add_argument(
    "--dataset-path",
    type=pathlib.Path,
    help="Path to L3DAS22 datasets.",
    default = "../datasets/L3DAS22/",
)
parser.add_argument(
    "--batch_size",
    default=4,
    type=int,
    help="Batch size for training. (Default: 4)",
)
parser.add_argument(
    "--gpus",
    default=1,
    type=int,
    help="Number of GPUs per node to use for training. (Default: 1)",
)
parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    help="Number of epochs to train for. (Default: 100)",
)
######### MINGEUN ###########
parser.add_argument('--dataset', default='L3DAS22', help='used dataset')
parser.add_argument('--image-size', default=None, help='size of image for training if used')
parser.add_argument('--workers', type=int, default=16)
######### MINGEUN ###########

args = parser.parse_args()
args.image_size = 'audio'
########### MINGEUN ############
from logger import Logger
x = Logger("dnn_beamformer", args.batch_size, args.dataset, args.image_size, args.workers)
########### MINGEUN ############


############# MINGEUN ##################
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
            torch.cuda.empty_cache()
            x.ready_for_training()
            
############# MINGEUN ##################

def run_train(args):
    
    pl.seed_everything(1)
    # logger = TensorBoardLogger(args.exp_dir)
    callbacks = [
        ModelCheckpoint(
            args.checkpoint_path,
            monitor="val/loss",
            save_top_k=5,
            mode="min",
            save_last=True,
        ),
        MyIterationCallback(),

    ]

    trainer = pl.trainer.trainer.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        accelerator="gpu",
        devices=args.gpus,
        accumulate_grad_batches=1,
        logger=None,
        gradient_clip_val=5,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    model = DNNBeamformer()
    model_module = DNNBeamformerLightningModule(model)
    data_module = L3DAS22DataModule(dataset_path=args.dataset_path, batch_size=args.batch_size, workers=args.workers)

    trainer.fit(model_module, datamodule=data_module)


def cli_main():
    run_train(args)


if __name__ == "__main__":
    cli_main()
