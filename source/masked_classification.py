###############
# generic
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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import copy
import random

#################
# specific
from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import *
from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np
from models.masked.encoder import *
from clinical_ts.eval_utils_cafa import eval_scores, eval_scores_bootstrap
from torch.optim.lr_scheduler import _LRScheduler
import math
from torch.optim.lr_scheduler import _LRScheduler


# Ensuring deterministic DataLoader worker seeds
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): output (batch_size, num_classes).
            targets (torch.Tensor): target (batch_size, num_classes), one-hot or binary.
        
        Returns:
            torch.Tensor: Focal Loss.
        """
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction = "none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss


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
            if(freeze):
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
        assert ((state_dict[k].cpu() == state_dict_pre[k].cpu()).all()), '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")
  

class LightningSupervisedModel(pl.LightningModule):
    def __init__(self, hparams):
        super(LightningSupervisedModel, self).__init__()
        self.hparams_f = hparams
        self.lr = self.hparams_f.lr
        self.best_metric = float('-inf')
        if self.hparams_f.finetune_dataset.startswith("ptbxl"):
            self.criterion = F.cross_entropy if self.hparams_f.finetune_dataset == "thew" else F.binary_cross_entropy_with_logits
            # self.criterion = FocalLoss(alpha = 0.25, gamma = 2, reduction = 'mean')
            if (self.hparams_f.finetune_dataset == "thew"):
                num_classes = 5
            elif (self.hparams_f.finetune_dataset == "ptbxl_super"):
                num_classes = 5
            elif (self.hparams_f.finetune_dataset == "ptbxl_all"):
                num_classes = 71
            elif (self.hparams_f.finetune_dataset == "ptbxl_rhythm"):
                num_classes = 12
            elif (self.hparams_f.finetune_dataset == "ptbxl_diag_subclass"):
                num_classes = 23
        elif self.hparams_f.finetune_dataset.startswith("zheng"):
            self.criterion = F.cross_entropy if self.hparams_f.finetune_dataset == "thew" else F.binary_cross_entropy_with_logits
            if (self.hparams_f.finetune_dataset == "thew"):
                num_classes = 11
            elif (self.hparams_f.finetune_dataset == "zheng_rhythm"):
                num_classes = 11
            if (self.hparams_f.finetune_dataset == "zheng_all"):
                num_classes = 67
        elif self.hparams_f.finetune_dataset.startswith("cinc"):
            self.criterion = F.cross_entropy if self.hparams_f.finetune_dataset == "thew" else F.binary_cross_entropy_with_logits
            if (self.hparams_f.finetune_dataset == "thew"):
                num_classes = 111
            elif (self.hparams_f.finetune_dataset == "cinc"):
                num_classes = 111
        elif self.hparams_f.finetune_dataset.startswith("ribeiro"):
            self.criterion = F.cross_entropy if self.hparams_f.finetune_dataset == "thew" else F.binary_cross_entropy_with_logits
            if (self.hparams_f.finetune_dataset == "thew"):
                num_classes = 7
            elif (self.hparams_f.finetune_dataset == "ribeiro"):
                num_classes = 7
        elif self.hparams_f.finetune_dataset.startswith("code_15"):
            self.criterion = F.cross_entropy if self.hparams_f.finetune_dataset == "thew" else F.binary_cross_entropy_with_logits
            if (self.hparams_f.finetune_dataset == "code_15"):
                num_classes = 6

        if self.hparams_f.model_selection == 0: # masked ViT    
            self.model = ST_MEM_ViT(
                seq_len = self.hparams_f.input_size, patch_size = self.hparams_f.patch_size,
                num_leads = 12, num_classes = num_classes, width = 768, depth = 12,
                mlp_dim = 3072, heads = 12, dim_head = 64, qkv_bias = True,
                drop_out_rate = 0., attn_drop_out_rate = 0., drop_path_rate = 0.,
            )
        elif self.hparams_f.model_selection == 1: # masked layer-wise ViT
            self.model = ST_MEM_ViT_lw(
                seq_len = self.hparams_f.input_size, patch_size = self.hparams_f.patch_size,
                num_leads = 12, num_classes = num_classes, width = 768, depth = 12,
                mlp_dim = 3072, heads = 12, dim_head = 64, qkv_bias = True,
                drop_out_rate = 0., attn_drop_out_rate = 0., drop_path_rate = 0.,
            )
        elif self.hparams_f.model_selection == 2: # masked layer-wise loEViT 
            self.model = ST_MEM_ViT_MoL(
                seq_len = self.hparams_f.input_size, patch_size = self.hparams_f.patch_size,
                num_leads = 12, num_classes = num_classes, width = 768, depth = 12,
                mlp_dim = 3072, heads = 12, dim_head = 64, qkv_bias = True,
                drop_out_rate = 0., attn_drop_out_rate = 0., drop_path_rate = 0.,
            )
        
    def forward(self, x):
        return self.model(x)
        
    def _step(self, data_batch, batch_idx, train):      
        preds = self.forward(data_batch[0])
        loss = self.criterion(preds, data_batch[1])
        self.log("train_loss" if train else "val_loss", loss)
        return {'loss': loss, "preds": preds.detach(), "targs": data_batch[1]}
      
    def training_step(self, train_batch, batch_idx):
        if (self.hparams_f.linear_eval):
            _freeze_bn_stats(self)
        return self._step(train_batch, batch_idx, True)
        
    def validation_step(self, val_batch, batch_idx, dataloader_idx = 0):
        return self._step(val_batch, batch_idx, False)
        
    def validation_epoch_end(self, outputs_all):
        for dataloader_idx, outputs in enumerate(outputs_all): # multiple val dataloaders
            preds_all = torch.cat([x['preds'] for x in outputs])
            targs_all = torch.cat([x['targs'] for x in outputs])
            if (self.hparams_f.finetune_dataset == "thew"):
                preds_all = F.softmax(preds_all, dim = -1)
                targs_all = torch.eye(len(self.lbl_itos))[targs_all].to(preds.device) 
            else:
                preds_all = torch.sigmoid(preds_all)
            preds_all = preds_all.cpu().numpy()
            targs_all = targs_all.cpu().numpy()

            val_loss_monitor = np.mean([d.get("loss").to("cpu").numpy().astype(np.float64) for d in outputs_all[0] if "loss" in d])
            
            # instance level score
            res, classify_report, accuracy_instance, macro_mAP, std_macro_mAP, sample_mAP, std_sample_mAP, \
            macro_f1, std_macro_f1, sample_f1, std_sample_f1, sample_accuracy, std_sample_accuracy = \
                eval_scores(targs_all, preds_all, classes = self.lbl_itos)
            
            idmap = self.val_dataset.get_id_mapping()
            preds_all_agg, targs_all_agg = aggregate_predictions(preds_all, targs_all, idmap, aggregate_fn = np.mean)
            
            res_agg, classify_report_agg, accuracy_instance_agg, macro_mAP_agg, std_macro_mAP_agg, sample_mAP_agg, std_sample_mAP_agg, \
            macro_f1_agg, std_macro_f1_agg, sample_f1_agg, std_sample_f1_agg, sample_accuracy_agg, std_sample_accuracy_agg = \
                eval_scores(targs_all_agg, preds_all_agg, classes = self.lbl_itos)
            
            self.log_dict(
                {
                    "macro_auc_agg" + str(dataloader_idx): res_agg["label_AUC"]["macro"],
                    "samples_auc_agg" + str(dataloader_idx): res_agg["label_AUC"]["samples"],
                    "accuracy_instance_agg" + str(dataloader_idx): accuracy_instance_agg,
                    "sample_accuracy_agg" + str(dataloader_idx): sample_accuracy_agg,
                    "macro_f1_agg" + str(dataloader_idx): macro_f1_agg,
                    "sample_f1_agg" + str(dataloader_idx): sample_f1_agg,
                    "macro_mAP_agg" + str(dataloader_idx): macro_mAP_agg,
                    "sample_mAP_agg" + str(dataloader_idx): sample_mAP_agg,
                    #########
                    "macro_auc" + str(dataloader_idx): res["label_AUC"]["macro"],
                    "samples_auc" + str(dataloader_idx): res["label_AUC"]["samples"],
                    "accuracy_instance" + str(dataloader_idx): accuracy_instance,
                    "sample_accuracy" + str(dataloader_idx): sample_accuracy,
                    "macro_f1" + str(dataloader_idx): macro_f1,
                    "sample_f1" + str(dataloader_idx): sample_f1,
                    "macro_mAP" + str(dataloader_idx): macro_mAP,
                    "sample_mAP" + str(dataloader_idx): sample_mAP,
                }
            )

            print(
                "epoch", self.current_epoch,
                "macro_auc_agg" + str(dataloader_idx) + ":", res_agg["label_AUC"]["macro"],
                "macro_auc" + str(dataloader_idx) + ":", res["label_AUC"]["macro"],
            )

            # Access the validation loss from the logs
            if (dataloader_idx == 0) and (res_agg["label_AUC"]["macro"] >= self.best_metric):
                self.best_metric = res_agg["label_AUC"]["macro"]
                self.log("best_macro_auc_agg", self.best_metric)

    def on_fit_start(self):
        if (self.hparams_f.linear_eval):
            print("copying state dict before training for sanity check after training")   
            self.state_dict_pre = copy.deepcopy(self.state_dict().copy())
    
    def on_fit_end(self):
        if (self.hparams_f.linear_eval):
            sanity_check(self, self.state_dict_pre)
                              
    def setup(self, stage):
        # configure dataset params
        chunkify_train = False
        chunk_length_train = self.hparams_f.input_size if chunkify_train else 0
        stride_train = self.hparams_f.input_size
        
        chunkify_valtest = True
        chunk_length_valtest = self.hparams_f.input_size if chunkify_valtest else 0
        stride_valtest = self.hparams_f.input_size // 2

        tfms_mask = transforms.Compose([Normalize(np.zeros(12), np.ones(12)), ToTensor()]) # a given mean/std calculated from certain datasets
            
        ### - Calculate mean/std of the pre-trained model - ###
        if self.hparams_f.normalize == 2:
            if (self.hparams_f.pretrained != ""):
                predefined_mean = np.array(
                    [0.01280697, -0.07400763, -0.06765056, -0.07100317, -0.07507894, -0.12284327, -0.08932588, -0.11420273, -0.08601768, 0.024781, 0.04054207, -0.08625173]
                )
                predefined_std = np.array(
                    [1.0417495, 1.68107641, 1.33222465, 1.38948182, 1.52399162, 1.55477694, 1.39696444, 1.50415624, 1.68829511, 1.11405372, 1.11482145, 1.60335439]
                )
                tfms_mask = transforms.Compose([Normalize(predefined_mean, predefined_std), ToTensor()]) # using mean/std of the pre-trained model
            else:
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
        test_datasets = []
        
        for i, target_folder in enumerate(self.hparams_f.data):
            target_folder = Path(target_folder)           
            df_mapped, lbl_itos, mean_all, std_all = load_dataset(target_folder)
            
            if self.hparams_f.finetune_dataset.startswith("ptbxl"): # specific for PTB-XL
                # always use PTB-XL stats
                if (self.hparams_f.finetune_dataset == "ptbxl_super"):
                    ptb_xl_label = "label_diag_superclass"
                elif (self.hparams_f.finetune_dataset == "ptbxl_all"):
                    ptb_xl_label = "label_all"
                elif (self.hparams_f.finetune_dataset == "ptbxl_rhythm"):
                    ptb_xl_label = "label_rhythm"
                elif (self.hparams_f.finetune_dataset == "ptbxl_diag_subclass"):
                    ptb_xl_label = "label_diag_subclass"
                    
                lbl_itos = np.array(lbl_itos[ptb_xl_label])
                
                def multihot_encode(x, num_classes):
                    res = np.zeros(num_classes, dtype = np.float32)
                    for y in x:
                        res[y] = 1
                    return res
                
                df_mapped["label"] = df_mapped[ptb_xl_label + "_filtered_numeric"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
            
            elif self.hparams_f.finetune_dataset.startswith("zheng"):
                if (self.hparams_f.finetune_dataset == "zheng_rhythm"):
                    zheng_label = "rhythm"
                elif (self.hparams_f.finetune_dataset == "zheng_all"):
                    zheng_label = "all"
                
                lbl_itos = np.array(lbl_itos[zheng_label])
                
                def multihot_encode(x, num_classes):
                    res = np.zeros(num_classes, dtype = np.float32)
                    for y in x:
                        res[y] = 1
                    return res
                
                if zheng_label == "all":  
                    df_mapped["label"] = df_mapped["label"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
                elif zheng_label == "rhythm":
                    df_mapped["label"] = df_mapped["label_" + zheng_label].apply(lambda x: multihot_encode(x, len(lbl_itos)))

            elif self.hparams_f.finetune_dataset.startswith("cinc"):                   
                lbl_itos = np.array(lbl_itos)
                
                def multihot_encode(x, num_classes):
                    res = np.zeros(num_classes, dtype = np.float32)
                    for y in x:
                        res[y] = 1
                    return res

                df_mapped["label"] = df_mapped["label"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
                
            elif self.hparams_f.finetune_dataset.startswith("ribeiro"):                   
                lbl_itos = np.array(lbl_itos)
                
                def multihot_encode(x, num_classes):
                    res = np.zeros(num_classes, dtype = np.float32)
                    for y in x:
                        res[y] = 1
                    return res
                
                df_mapped["label"] = df_mapped['label'].apply(lambda x: [6] if x == [] else x)
                lbl_itos = np.append(lbl_itos, np.array(['N']))
                
                df_mapped["label"] = df_mapped["label"].apply(lambda x: multihot_encode(x, len(lbl_itos))) 

            elif self.hparams_f.finetune_dataset.startswith("code_15"):                   
                lbl_itos = np.array(lbl_itos)[0: -1]
                
                def multihot_encode(x, num_classes):
                    res = np.zeros(num_classes, dtype = np.float32)
                    for y in x:
                        res[y] = 1
                    return res
                
                df_mapped["label"] = df_mapped["ecg_labels"].apply(lambda x: multihot_encode(x, len(lbl_itos)))

            self.lbl_itos = lbl_itos
            max_fold_id = df_mapped.strat_fold.max() # unfortunately 1-based for PTB-XL; sometimes 100 (Ribeiro)
            
            df_train = df_mapped[df_mapped.strat_fold < (max_fold_id - 1)]
            df_val = df_mapped[df_mapped.strat_fold == (max_fold_id - 1)]
            df_test = df_mapped[df_mapped.strat_fold == max_fold_id]

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
                    col_lbl = "label",
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
                    col_lbl = "label",
                    memmap_filename = target_folder/("memmap.npy"),
                    normalize_option = self.hparams_f.normalize,
                )
            )
            test_datasets.append(
                TimeseriesDatasetCrops(
                    df_test,
                    self.hparams_f.input_size,
                    num_classes = len(lbl_itos),
                    data_folder = target_folder,
                    chunk_length = chunk_length_valtest,
                    min_chunk_length = self.hparams_f.input_size, 
                    stride = stride_valtest,
                    transforms = tfms_mask,
                    annotation = False,
                    col_lbl = "label",
                    memmap_filename = target_folder/("memmap.npy"),
                    normalize_option = self.hparams_f.normalize,
                )
            )
            
            print("\n", target_folder)
            print("train dataset: ", len(train_datasets[-1]), "samples")
            print("val dataset: ", len(val_datasets[-1]), "samples")
            print("test dataset: ", len(test_datasets[-1]), "samples")

        if (len(train_datasets) > 1): # multiple data folders
            print("\nCombined: ")
            self.train_dataset = ConcatDataset(train_datasets)
            self.val_dataset = ConcatDataset(val_datasets)
            print("train dataset: ", len(self.train_dataset), "samples")
            print("val dataset: ", len(self.val_dataset), "samples")
            self.test_dataset = ConcatDataset(test_datasets)
            print("test dataset: ", len(self.test_dataset), "samples")
        else: # just a single data folder
            self.train_dataset = train_datasets[0]
            self.val_dataset = val_datasets[0]
            self.test_dataset = test_datasets[0]
    
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
        return [
            DataLoader(self.val_dataset, batch_size = self.hparams_f.batch_size, num_workers = 4, worker_init_fn = seed_worker),
            DataLoader(self.test_dataset, batch_size = self.hparams_f.batch_size, num_workers = 4, worker_init_fn = seed_worker),
        ]

    ## Function to get parameters of specified blocks based on a condition
    def get_specific_block(self, model, condition):
        params = []
        for name, param in model.named_parameters():
            if condition(name):
                params.append(param)
        return params

    # Condition to identify special blocks (e.g., block names containing '2' or '4')
    def is_special_block(self, name):
        return 'head' in name

    # Function to get parameters of other blocks
    def get_blocks(self, model):
        params = []
        for name, param in model.named_parameters():
            if 'head' not in name:
                params.append(param)
        return params
    
    def configure_optimizers(self):
        if self.hparams_f.mode == "linprobe":
            ### - used for linear probing - ###
            if (self.hparams_f.optimizer == "sgd"):
                opt = torch.optim.SGD
            elif (self.hparams_f.optimizer == "adam"):
                opt = torch.optim.AdamW
            else:
                raise NotImplementedError("Unknown Optimizer.")

            # Create parameter groups dynamically
            default_params = self.get_blocks(self.model)
            special_params = self.get_specific_block(self.model, self.is_special_block)
            if (self.hparams_f.finetune and (self.hparams_f.linear_eval or self.hparams_f.train_head_only)):
                optimizer = opt(self.model.head.parameters(), self.lr, weight_decay = self.hparams_f.weight_decay) 
            elif (self.hparams_f.finetune and self.hparams_f.discriminative_lr_factor != 1.): # discrimative lrs
                optimizer = opt(
                    [
                        {
                            "params": default_params, 
                            "lr": self.lr * (self.hparams_f.discriminative_lr_factor ** 2),
                        },
                        {
                            "params": special_params, 
                            "lr": self.lr,
                        },
                    ],
                    self.hparams_f.lr, 
                    weight_decay = self.hparams_f.weight_decay,
                )
            else:
                optimizer = opt(self.parameters(), self.lr, weight_decay = self.hparams_f.weight_decay)
            return optimizer
        
        else: ### - used for fine tune and train from scratch - ###   
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
                    betas = (0.9, 0.999), 
                    eps = 1e-8, 
                    weight_decay = self.hparams_f.weight_decay,
                )
            else:
                raise NotImplementedError("Unknown Optimizer.")
            scheduler = CustomLRScheduler(optimizer, self.hparams_f, len(self.train_dataloader()))
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]


#####################################################################################################
# ARGPARSERS
#####################################################################################################
def add_model_specific_args(parser):
    parser.add_argument("--input-channels", type = int, default = 12)
    parser.add_argument("--normalize", dest = "normalize", type = int, help = "select normalization mode", default = "0")
    return parser


def add_default_args():
    parser = argparse.ArgumentParser(description='PyTorch Lightning Supervised Masked Training')
    parser.add_argument('--data', metavar = 'DIR',type = str, help = 'path(s) to dataset', action = 'append')
    parser.add_argument('--epochs', default = 10, type = int, metavar = 'N', help = 'number of total epochs to run') 
    parser.add_argument('--warmup-epochs', default = 5, type = int, metavar = 'N', help = 'number of total epochs to run')
    parser.add_argument(
        '--batch-size', default = 16, type = int, metavar = 'N',
        help = 'mini-batch size (default: 256), this is the total '
               'batch size of all GPUs on the current node when '
               'using Data Parallel or Distributed Data Parallel',
    )
    parser.add_argument('--lr', '--learning-rate', default = 1e-3, type = float, metavar = 'LR', help = 'initial learning rate', dest = 'lr')
    parser.add_argument('--wd', '--weight-decay', default = 0.05, type = float, metavar = 'W', help = 'weight decay (default: 0.)', dest = 'weight_decay')
    parser.add_argument('--resume', default = '', type = str, metavar = 'PATH', help = 'path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default = '', type = str, metavar = 'PATH', help = 'path to pretrained checkpoint (default: none)')
    parser.add_argument('--optimizer', default = 'adam', help = 'sgd/adam') # was sgd
    parser.add_argument('--output-path', default = '.', type = str, dest = "output_path", help = 'output path')
    parser.add_argument('--metadata', default = '', type = str, help = 'metadata for output')
    parser.add_argument("--gpus", type = int, default = 1, help = "number of gpus")
    parser.add_argument("--num-nodes", dest = "num_nodes", type = int, default = 1, help = "number of compute nodes")
    parser.add_argument("--precision", type = int, default = 32, help = "16/32")
    parser.add_argument("--distributed-backend", dest = "distributed_backend", type = str, default = None, help = "None/ddp")
    parser.add_argument("--accumulate", type = int, default = 1, help = "accumulate grad batches (total-bs=accumulate-batches*bs)")   
    parser.add_argument("--input-size", dest = "input_size", type = int, default = 16000)
    parser.add_argument("--patch-size", dest = "patch_size", type = int, default = 50)
    parser.add_argument("--model-selection", dest = "model_selection", type = int, help = "select model for supervised training", default = "0")
    parser.add_argument("--finetune-dataset", type = str, help = "thew/ptbxl_super/ptbxl_all", default = "thew")
    parser.add_argument("--linear-eval", action = "store_true", help = "linear evaluation instead of full finetuning", default = False)
    parser.add_argument("--mode", type = str, help = "select training mode", default = "finetune")
    parser.add_argument("--lr-find", action = "store_true", help = "run lr finder before training run", default = False)
    parser.add_argument("--finetune", action = "store_true", help = "finetuning (downstream classification task)",  default = False)
    parser.add_argument(
        "--train-head-only", action = "store_true", 
        help = "freeze everything except classification head (note: --linear-eval defaults to no hidden layer in classification head)",
    )
    parser.add_argument("--discriminative-lr-factor", type = float, help = "factor by which the lr decreases per layer group during finetuning", default = 0.1)
    
    return parser

             
###################################################################################################
# MAIN
###################################################################################################
if __name__ == '__main__':
    parser = add_default_args()
    parser = add_model_specific_args(parser)
    hparams = parser.parse_args()
    hparams.executable = "supervised_from_scratch"


    if not os.path.exists(os.path.join(hparams.output_path)):
        os.makedirs(os.path.join(hparams.output_path))

    model = LightningSupervisedModel(hparams)
    if (hparams.pretrained != ""):
        checkpoint_model = torch.load(hparams.pretrained, map_location = 'cpu')
        print(f"Load pre-trained checkpoint from: ", hparams.pretrained)
        state_dict = model.model.state_dict()
    
    if hparams.mode == "linprobe":
        if hparams.model_selection == 2:
            layer_keys = [
                'gating.net.0.weight', 'gating.net.0.bias', 'gating.net.1.weight', 'gating.net.1.bias', 
                'gating.net.4.weight', 'gating.net.4.bias', 'gating.net.7.weight', 'gating.net.7.bias', 
                'head.0.weight', 'head.0.bias', 'head.0.running_mean', 'head.0.running_var', 'head.0.num_batches_tracked', 
                'head.2.weight', 'head.2.bias', 'head.4.weight', 'head.4.bias', 'head.4.running_mean', 
                'head.4.running_var', 'head.4.num_batches_tracked', 'head.6.weight', 'head.6.bias',
            ]
        else:
            layer_keys = [
                'head.0.weight', 'head.0.bias', 'head.0.running_mean', 'head.0.running_var', 'head.0.num_batches_tracked', 
                'head.2.weight', 'head.2.bias', 'head.4.weight', 'head.4.bias', 'head.4.running_mean', 
                'head.4.running_var', 'head.4.num_batches_tracked', 'head.6.weight', 'head.6.bias',
            ]
        for k in layer_keys:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Remove key {k} from pre-trained checkpoint")
                del checkpoint_model[k]
        msg = model.model.load_state_dict(checkpoint_model, strict = False)

        if hparams.model_selection == 2:
            assert set(msg.missing_keys) == {
                'gating.net.0.weight', 'gating.net.0.bias', 'gating.net.1.weight', 'gating.net.1.bias', 
                'gating.net.4.weight', 'gating.net.4.bias', 'gating.net.7.weight', 'gating.net.7.bias',
                'head.0.weight', 'head.0.bias', 'head.0.running_mean', 'head.0.running_var', 
                'head.2.weight', 'head.2.bias', 'head.4.weight', 'head.4.bias', 'head.4.running_mean', 
                'head.4.running_var', 'head.6.weight', 'head.6.bias',
            }
        else:
            assert set(msg.missing_keys) == {
                'head.0.weight', 'head.0.bias', 'head.0.running_mean', 'head.0.running_var', 
                'head.2.weight', 'head.2.bias', 'head.4.weight', 'head.4.bias', 'head.4.running_mean', 
                'head.4.running_var', 'head.6.weight', 'head.6.bias',
            }
        for _, p in model.model.named_parameters():
            p.requires_grad = False
        for _, p in model.model.head.named_parameters():
            p.requires_grad = True
        if hparams.model_selection == 2:
            for _, p in model.model.gating.net.named_parameters():
                p.requires_grad = True
    elif hparams.mode == "finetune": 
        msg = model.load_state_dict(checkpoint_model['state_dict'], strict = True) 
        for _, p in model.model.named_parameters():
            p.requires_grad = True
        for _, p in model.model.head.named_parameters():
            p.requires_grad = True  

    logger = TensorBoardLogger(save_dir = os.path.join(hparams.output_path), name="")
    print("Output directory:", logger.log_dir) 

    checkpoint_callback = ModelCheckpoint(
        dirpath = os.path.join(logger.log_dir, "best_model"), # hparams.output_path
        save_top_k = 1,
		save_last = True,
        verbose = True,
        monitor = 'macro_auc_agg0', # val_loss / dataloader_idx_0
        mode = 'max',
        filename = '',
    )

    lr_monitor = LearningRateMonitor()

    trainer = pl.Trainer(
        auto_lr_find = hparams.lr_find,
        accumulate_grad_batches = hparams.accumulate,
        max_epochs = hparams.epochs,
        min_epochs = hparams.epochs,
        default_root_dir = os.path.join(hparams.output_path),
        num_sanity_val_steps = 0,
        logger = logger,
        callbacks = [checkpoint_callback], 
        benchmark = False,
    
        gpus = hparams.gpus,
        num_nodes = hparams.num_nodes,
        precision = hparams.precision,
        strategy = hparams.distributed_backend,

        enable_progress_bar = False,
        resume_from_checkpoint = None if hparams.resume == "" else hparams.resume,
        deterministic = True,
    )
        
    if (hparams.lr_find): # lr find
        trainer.tune(model)
        
    trainer.fit(model)