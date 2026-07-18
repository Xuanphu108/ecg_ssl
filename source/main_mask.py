###############
# generic
from functools import partial
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.nn.functional as F
import torchvision
import os
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
import copy

#################
# specific
from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import *
import random
from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np
from models.masked import *
from torch.optim.lr_scheduler import _LRScheduler
import math


def setup_seed(seed):
    # Fix the random seed for reproducibility
    pl.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Ensuring deterministic DataLoader worker seeds
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


setup_seed(42)


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, hparams, num_batches, last_epoch = -1):
        self.hparams = hparams
        self.num_batches = num_batches
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # Here last_epoch plays a role as the last step
        data_iter_step = self.last_epoch % self.num_batches
        # current epoch
        current_epoch = self.last_epoch // self.num_batches

        progress = (data_iter_step / self.num_batches) + current_epoch
        min_lr = 0
        new_lr = self.hparams.lr * self.hparams.batch_size / 256

        if progress < self.hparams.warmup_epochs:
            lr = new_lr * progress / self.hparams.warmup_epochs
        else:
            total_epochs = self.hparams.epochs - self.hparams.warmup_epochs
            progress_after_warmup = progress - self.hparams.warmup_epochs
            lr = min_lr + (new_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * progress_after_warmup / total_epochs))

        return [lr * param_group.get("lr_scale", 1.0) for param_group in self.optimizer.param_groups]

    def step(self, epoch = None, batch_idx = None):
        if epoch is not None and batch_idx is not None:
            self.last_epoch = epoch * self.num_batches + batch_idx
        else:
            self.last_epoch += 1
        
        self._last_lr = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group['lr'] = lr


def _freeze_bn_stats(model, freeze = True):
    for m in model.modules():
        if (isinstance(m, nn.BatchNorm1d)):
            if (freeze):
                m.eval()
            else:
                m.train()
                

def sanity_check(model, state_dict_pre):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading state dict for sanity check")
    state_dict = model.state_dict()

    for k in list(state_dict.keys()):
        print(k)
        # only ignore fc layer
        if 'head.1.weight' in k or 'head.1.bias' in k:
            continue

        assert ((state_dict[k].cpu() == state_dict_pre[k].cpu()).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class LightningMasked(pl.LightningModule):

    def __init__(self, hparams):
        super(LightningMasked, self).__init__()
        setup_seed(42)
        self.hparams_f = hparams
        self.lr = self.hparams_f.lr

        if self.hparams_f.model_selection == 0: # masked ViT
            self.model = ST_MEM(
                seq_len = self.hparams_f.input_size, patch_size = self.hparams_f.patch_size, 
                num_leads = 12, embed_dim = 768, depth = 12, num_heads = 12,
                decoder_embed_dim = 256, decoder_depth = 4, decoder_num_heads = 4,
                mlp_ratio = 4, qkv_bias = True, norm_layer = partial(nn.LayerNorm, eps = 1e-6),
                norm_pix_loss = True, loss_select = self.hparams_f.loss_select,
            )
        elif self.hparams_f.model_selection == 1: # masked layer-wise ViT
            self.model = ST_MEM_lw(
                seq_len = self.hparams_f.input_size, patch_size = self.hparams_f.patch_size,
                num_leads = 12, embed_dim = 768, depth = 12, num_heads = 12,
                decoder_embed_dim = 256, decoder_depth = 4, decoder_num_heads = 4,
                mlp_ratio = 4, qkv_bias = True, norm_layer = partial(nn.LayerNorm, eps = 1e-6),
                norm_pix_loss = True, loss_select = self.hparams_f.loss_select,
            ) # and v2

    def forward(self, x):
        return self.model(x, mask_ratio = 0.75)
        
    def _step(self, data_batch, batch_idx, train):       
        loss = self.model.forward_loss(data_batch[0], mask_ratio = 0.75)
        self.log("loss" if train else "val_loss", loss)
        return loss
      
    def training_step(self, train_batch, batch_idx):
        return self._step(train_batch, batch_idx, True)
        
    def validation_step(self, val_batch, batch_idx, dataloader_idx = 0):
        return self._step(val_batch, batch_idx, False)
    
    def setup(self, stage):
        # configure dataset params
        chunkify_train = False
        chunk_length_train = self.hparams_f.input_size if chunkify_train else 0
        stride_train = self.hparams_f.input_size
        
        chunkify_valtest = True
        chunk_length_valtest = self.hparams_f.input_size if chunkify_valtest else 0
        stride_valtest = self.hparams_f.input_size // 2

        tfms_mask = transforms.Compose([Normalize(np.zeros(12), np.ones(12)), ToTensor()]) # a given mean/std calculated from certain datasets
            
        ### - Calculate mean/std - ###
        if self.hparams_f.normalize == 2:
            total_train = pd.DataFrame()
            for i, target_folder in enumerate(self.hparams_f.data):
                target_folder = Path(target_folder)           
                df_mapped, lbl_itos, mean, std = load_dataset(target_folder)
                self.lbl_itos = lbl_itos
                max_fold_id = df_mapped.strat_fold.max() # unfortunately 1-based for PTB-XL; sometimes 100 (Ribeiro)
                df_train = df_mapped[df_mapped.strat_fold < max_fold_id]
                total_train = pd.concat([total_train, df_train], ignore_index = True)

            total_points = 0
            sum_samples = 0
            sum_square_samples = 0
            for i in range(len(total_train)):
                sum_samples += total_train["data_mean"][i] * total_train["data_length"][i]
                sum_square_samples += (total_train["data_std"][i] ** 2 + total_train["data_mean"][i] ** 2) * total_train["data_length"][i]
                total_points += total_train["data_length"][i]
            mean_global = sum_samples / total_points
            std_global = np.sqrt(sum_square_samples / total_points)
            tfms_mask = transforms.Compose([Normalize(mean_global, std_global), ToTensor()]) # a given mean/std calculated from certain datasets
        
        ### - training - ###
        train_datasets = []
        val_datasets = []
        
        for i, target_folder in enumerate(self.hparams_f.data):
            target_folder = Path(target_folder)           
            df_mapped, lbl_itos, _, _ = load_dataset(target_folder)
                                
            self.lbl_itos = lbl_itos
            max_fold_id = df_mapped.strat_fold.max() # unfortunately 1-based for PTB-XL; sometimes 100 (Ribeiro)
            
            df_train = df_mapped[df_mapped.strat_fold < max_fold_id]
            df_val = df_mapped[df_mapped.strat_fold == max_fold_id]
            
            train_datasets.append(
                TimeseriesDatasetCrops(
                    df_train,
                    self.hparams_f.input_size,
                    num_classes = len(lbl_itos),
                    data_folder = target_folder,
                    chunk_length = chunk_length_train,
                    min_chunk_length = self.hparams_f.input_size, 
                    stride = stride_train,
                    transforms = tfms_mask,
                    annotation = False,
                    col_lbl = None,
                    memmap_filename = target_folder/("memmap.npy"),
                    normalize_option = self.hparams_f.normalize,
                )
            )
            val_datasets.append(
                TimeseriesDatasetCrops(
                    df_val,
                    self.hparams_f.input_size,
                    num_classes = len(lbl_itos),
                    data_folder = target_folder,
                    chunk_length = chunk_length_valtest,
                    min_chunk_length = self.hparams_f.input_size, 
                    stride = stride_valtest,
                    transforms = tfms_mask,
                    annotation = False,
                    col_lbl = None,
                    memmap_filename = target_folder/("memmap.npy"),
                    normalize_option = self.hparams_f.normalize,
                ),
            )
            
            print("\n", target_folder)
            print("train dataset: ", len(train_datasets[-1]), "samples")
            print("val dataset: ", len(val_datasets[-1]), "samples")

        if (len(train_datasets) > 1): # multiple data folders
            print("\nCombined: ")
            self.train_dataset = ConcatDataset(train_datasets)
            self.val_dataset = ConcatDataset(val_datasets)
            print("train dataset: ", len(self.train_dataset), "samples")
            print("val dataset: ", len(self.val_dataset), "samples")
        
        else: # just a single data folder
            self.train_dataset = train_datasets[0]
            self.val_dataset = val_datasets[0]
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size = self.hparams_f.batch_size, 
            num_workers = 4, 
            shuffle = True, 
            drop_last = True, 
            worker_init_fn = seed_worker,
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size = self.hparams_f.batch_size, 
            num_workers = 4,
            worker_init_fn = seed_worker,
        )

    def configure_optimizers(self):
        if (self.hparams_f.optimizer == "sgd"):
            opt = torch.optim.SGD
            optimizer = opt(
                self.parameters(),
                self.lr,
                momentum = 0, 
                weight_decay = self.hparams_f.weight_decay,
            )
        elif (self.hparams_f.optimizer == "adam"):
            opt = torch.optim.AdamW
            optimizer = opt(
                self.parameters(), 
                self.lr, 
                betas = (0.9, 0.95), 
                eps = 1e-8, 
                weight_decay = self.hparams_f.weight_decay,
            )
        else:
            raise NotImplementedError("Unknown Optimizer.")
        
        scheduler = CustomLRScheduler(optimizer, self.hparams_f, len(self.train_dataloader()))
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def load_weights_from_checkpoint(self, checkpoint):
        """ Function that loads the weights from a given checkpoint file. 
        based on https://github.com/PyTorchLightning/pytorch-lightning/issues/525
        """
        checkpoint = torch.load(checkpoint, map_location = lambda storage, loc: storage,)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
            
        pretrained_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class extractEncoder(Callback):
    def __init__(self, monitor = 'val_loss', mode = 'min', save_path = 'best_models'):
        super(extractEncoder, self).__init__()
        self.monitor = monitor
        self.mode = mode
        self.save_path = save_path
        self.best_score = None
        self.best_epoch = -1
        os.makedirs(self.save_path, exist_ok=True)
        
    def on_validation_end(self, trainer, pl_module):
        metric_value = trainer.callback_metrics.get(self.monitor)
        if metric_value is None:
            return
        
        if self.best_score is None or (
            self.mode == 'min' and metric_value < self.best_score
        ) or (
            self.mode == 'max' and metric_value > self.best_score
        ):
            self.best_score = metric_value
            self.best_epoch = trainer.current_epoch

            # Save encoder
            encoder_state_dict = pl_module.model.encoder.state_dict()
            torch.save(encoder_state_dict, os.path.join(self.save_path, f'best_encoder.pth'))

            # # Save entire model
            # torch.save(pl_module, os.path.join(self.save_path, f'model_epoch_{self.best_epoch}_{self.best_score}.ckpt'))


#####################################################################################################
# ARGPARSERS
#####################################################################################################
def add_model_specific_args(parser):
    parser.add_argument("--input-channels", type = int, default = 12)
    parser.add_argument("--normalize", dest = "normalize", type = int, help = "select normalization mode", default = "0")
    return parser

def add_default_args():
    parser = argparse.ArgumentParser(description = 'PyTorch Lightning Masked Transformer Training')
    parser.add_argument('--data', metavar = 'DIR', type = str, help = 'path(s) to dataset',action = 'append')
    parser.add_argument('--epochs', default = 800, type = int, metavar = 'N', help = 'number of total epochs to run')
    parser.add_argument('--warmup-epochs', default = 5, type = int, metavar = 'N', help = 'number of total epochs to run')
    parser.add_argument(
        '--batch-size', default = 64, type = int, metavar = 'N',
        help = 'mini-batch size (default: 256), this is the total '
               'batch size of all GPUs on the current node when '
               'using Data Parallel or Distributed Data Parallel',
    )
    parser.add_argument('--lr', '--learning-rate', default = 0.0012, type = float, metavar = 'LR', help = 'initial learning rate', dest = 'lr')
    parser.add_argument('--wd', '--weight-decay', default = 0.01, type = float, metavar = 'W', help = 'weight decay (default: 0.)', dest = 'weight_decay')
    parser.add_argument('--resume', default = '', type = str, metavar = 'PATH', help = 'path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default = '', type = str, metavar = 'PATH', help = 'path to pretrained checkpoint (default: none)')
    parser.add_argument('--optimizer', default = 'adam', help = 'sgd/adam') # was sgd
    parser.add_argument('--output-path', default = '.', type = str,dest="output_path", help = 'output path')
    parser.add_argument('--metadata', default = '', type = str, help = 'metadata for output')
    parser.add_argument("--gpus", type = int, default = 1, help = "number of gpus")
    parser.add_argument("--num-nodes", dest = "num_nodes", type = int, default = 1, help = "number of compute nodes")
    parser.add_argument("--precision", type = int, default = 16, help = "16/32")
    parser.add_argument("--distributed-backend", dest = "distributed_backend", type = str, default = None, help = "None/ddp")
    parser.add_argument("--accumulate", type = int, default = 1, help = "accumulate grad batches (total-bs = accumulate - batches * bs)")
    parser.add_argument("--input-size", dest = "input_size", type = int, default = 1000)
    parser.add_argument("--patch-size", dest = "patch_size", type = int, default = 60)
    parser.add_argument("--model-selection", dest = "model_selection", type = int, help = "select model for supervised training", default = "0")
    parser.add_argument("--loss-select", dest = "loss_select", type = str, default = "mse")
    parser.add_argument("--discriminative-lr-factor", type = float, help = "factor by which the lr decreases per layer group during finetuning", default = 0.1)
    parser.add_argument("--lr-find", action = "store_true", help = "run lr finder before training run", default = False)
    
    return parser

             
###################################################################################################
# MAIN
###################################################################################################
if __name__ == '__main__':
    parser = add_default_args()
    parser = add_model_specific_args(parser)
    hparams = parser.parse_args()
    hparams.executable = "masked"

    if not os.path.exists(hparams.output_path):
        os.makedirs(hparams.output_path)
    
    model = LightningMasked(hparams)
    
    if (hparams.pretrained != ""):
        print("Loading pretrained weights from", hparams.pretrained)
        model.load_weights_from_checkpoint(hparams.pretrained)

    logger = TensorBoardLogger(
        save_dir = hparams.output_path,
        version = "version_0", # hparams.metadata.split(":")[0],
        name = "")
    print("Output directory:", logger.log_dir) 

    checkpoint_callback = ModelCheckpoint(
        dirpath = os.path.join(logger.log_dir, "best_models"), 
        filename = 'model_{epoch:02d}_{val_loss:.2f}',
        save_top_k = 1, # -1 keep all checkpoints
		save_last = True,
        verbose = True,
        monitor = 'val_loss', # val_loss/dataloader_idx_0
        mode = 'min',
    )

    # Custom callback to save the specific part of the model
    encoder_callback = extractEncoder(monitor = 'val_loss', mode = 'min', save_path = os.path.join(logger.log_dir, "best_models"))

    lr_monitor = LearningRateMonitor()

    trainer = pl.Trainer(
        # overfit_batches = 0.01,
        auto_lr_find = hparams.lr_find,
        max_epochs = hparams.epochs,
        min_epochs = hparams.epochs,
        
        default_root_dir = hparams.output_path,
        
        num_sanity_val_steps = 0,
        logger = logger,
        callbacks = [checkpoint_callback, encoder_callback, lr_monitor],
        benchmark = False,
    
        gpus = hparams.gpus,
        num_nodes = hparams.num_nodes,
        precision = hparams.precision,
        strategy = hparams.distributed_backend,
        accumulate_grad_batches = hparams.accumulate,
        
        enable_progress_bar = False,
        resume_from_checkpoint = None if hparams.resume == "" else hparams.resume,
        deterministic = True,
    )
        
    if (hparams.lr_find): # lr find
        trainer.tune(model)
        
    trainer.fit(model)   