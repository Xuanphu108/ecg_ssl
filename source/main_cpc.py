###############
# generic
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.nn.functional as F
import os
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import copy

#################
# specific
from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import *
from pathlib import Path
import pandas as pd
import numpy as np
import random
from clinical_ts.eval_utils_cafa import eval_scores
from models.cpc import *


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


class LightningCPC(pl.LightningModule):

    def __init__(self, hparams, seed):
        super(LightningCPC, self).__init__()
        self.seed = seed
        setup_seed(self.seed)  # Call the setup seed here for consistency
        self.hparams_f = hparams
        self.lr = self.hparams_f.lr
        self.best_metric = float('-inf')
        # self.best_metric = float('inf')
        # these coincide with the adapted wav2vec2 params
        if (self.hparams_f.fc_encoder):
            strides = [1] * 4 
            kss = [1] * 4 
            features = [512] * 4
        else: # strided conv encoder
            strides = [2, 2, 2, 2] # original wav2vec2 [5, 2, 2, 2, 2, 2] original cpc [5, 4, 2, 2, 2]
            kss = [10, 4, 4, 4] # original wav2vec2 [10, 3, 3, 3, 3, 2] original cpc [18, 8, 4, 4, 4]
            features = [512] * 4 # wav2vec2 [512] * 6 original cpc [512] * 5
        
        if (self.hparams_f.finetune):
            if self.hparams_f.finetune_dataset.startswith("ptbxl"):
                self.criterion = F.cross_entropy if self.hparams_f.finetune_dataset == "thew" else F.binary_cross_entropy_with_logits
                if (self.hparams_f.finetune_dataset == "thew"):
                    num_classes = 5
                elif (self.hparams_f.finetune_dataset == "ptbxl_super"):
                    num_classes = 5
                elif (self.hparams_f.finetune_dataset == "ptbxl_all"):
                    num_classes = 71
                elif (self.hparams_f.finetune_dataset == "ptbxl_rhythm"):
                    num_classes = 12
            elif self.hparams_f.finetune_dataset.startswith("zheng"):
                self.criterion = F.cross_entropy if self.hparams_f.finetune_dataset == "thew" else F.binary_cross_entropy_with_logits
                if (self.hparams_f.finetune_dataset == "thew"):
                    num_classes = 11
                elif (self.hparams_f.finetune_dataset == "zheng_rhythm"):
                    num_classes = 11
                if (self.hparams_f.finetune_dataset == "zheng_all"):
                    num_classes = 67
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
        else:
            num_classes = None

        self.model_cpc = CPCModel(
            input_channels = self.hparams_f.input_channels, 
            strides = strides,
            kss = kss,
            features = features,
            n_hidden = self.hparams_f.n_hidden,
            n_layers = self.hparams_f.n_layers,
            mlp = self.hparams_f.mlp,
            lstm = not(self.hparams_f.gru),
            bias_proj = self.hparams_f.bias,
            num_classes = num_classes,
            skip_encoder = self.hparams_f.skip_encoder,
            bn_encoder = not(self.hparams_f.no_bn_encoder),
            lin_ftrs_head = [] if self.hparams_f.linear_eval else eval(self.hparams_f.lin_ftrs_head),
            ps_head = 0 if self.hparams_f.linear_eval else self.hparams_f.dropout_head,
            bn_head = False if self.hparams_f.linear_eval else not(self.hparams_f.no_bn_head),
        )
        
        target_fs = 100
        if (not(self.hparams_f.finetune)):
            print(
                "CPC pretraining:\ndownsampling factor:", self.model_cpc.encoder_downsampling_factor,
                "\nchunk length(s)", self.model_cpc.encoder_downsampling_factor/target_fs,
                "\npixels predicted ahead:", self.model_cpc.encoder_downsampling_factor*self.hparams_f.steps_predicted, 
                "\nseconds predicted ahead:", self.model_cpc.encoder_downsampling_factor*self.hparams_f.steps_predicted/target_fs, 
                "\nRNN input size:", self.hparams_f.input_size//self.model_cpc.encoder_downsampling_factor,
            )

    def forward(self, x):
        return self.model_cpc(x)
        
    def _step(self,data_batch, batch_idx, train):       
        if (self.hparams_f.finetune):
            preds = self.forward(data_batch[0])
            loss = self.criterion(preds, data_batch[1])
            self.log("train_loss" if train else "val_loss", loss)
            return {'loss': loss, "preds": preds.detach(), "targs": data_batch[1]}
        else:
            loss, acc = self.model_cpc.cpc_loss(
                data_batch[0],
                steps_predicted = self.hparams_f.steps_predicted,
                n_false_negatives = self.hparams_f.n_false_negatives, 
                negatives_from_same_seq_only = self.hparams_f.negatives_from_same_seq_only, 
                eval_acc = True,
            )
            self.log("loss" if train else "val_loss", loss)
            self.log("acc" if train else "val_acc", acc)
            return loss
      
    def training_step(self, train_batch, batch_idx):
        if(self.hparams_f.linear_eval):
            _freeze_bn_stats(self)
        return self._step(train_batch, batch_idx, True)
        
    def validation_step(self, val_batch, batch_idx, dataloader_idx = 0):
        return self._step(val_batch, batch_idx, False)
        
    def validation_epoch_end(self, outputs_all):
        if (self.hparams_f.finetune):
            for dataloader_idx, outputs in enumerate(outputs_all): # multiple val dataloaders
                preds_all = torch.cat([x['preds'] for x in outputs])
                targs_all = torch.cat([x['targs'] for x in outputs])
                if(self.hparams_f.finetune_dataset == "thew"):
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

        tfms_cpc = transforms.Compose([Normalize(np.zeros(12), np.ones(12)), ToTensor()]) # a given mean/std calculated from certain datasets
            
        ### - Calculate mean/std - ###
        if self.hparams_f.normalize == 2:
            if (self.hparams_f.pretrained != ""): # self.hparams_f.finetune and (self.hparams_f.discriminative_lr_factor != 1)
                predefined_mean = np.array(
                    [0.01280697, -0.07400763, -0.06765056, -0.07100317, -0.07507894, -0.12284327, -0.08932588, -0.11420273, -0.08601768, 0.024781, 0.04054207, -0.08625173]
                )
                predefined_std = np.array(
                    [1.0417495, 1.68107641, 1.33222465, 1.38948182, 1.52399162, 1.55477694, 1.39696444, 1.50415624, 1.68829511, 1.11405372, 1.11482145, 1.60335439]
                )
                tfms_cpc = transforms.Compose([Normalize(predefined_mean, predefined_std), ToTensor()]) # using mean/std of the pre-trained model
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
                tfms_cpc = transforms.Compose([Normalize(mean_global, std_global), ToTensor()]) # a given mean/std calculated from certain datasets

        ### - training - ###
        train_datasets = []
        val_datasets = []
        test_datasets = []
        for i, target_folder in enumerate(self.hparams_f.data):
            target_folder = Path(target_folder)           
            df_mapped, lbl_itos, mean_all, std_all = load_dataset(target_folder)
            
            # specific for PTB-XL
            if (self.hparams_f.finetune and self.hparams_f.finetune_dataset.startswith("ptbxl")):
                if (self.hparams_f.finetune_dataset == "ptbxl_super"):
                    ptb_xl_label = "label_diag_superclass"
                elif (self.hparams_f.finetune_dataset == "ptbxl_all"):
                    ptb_xl_label = "label_all"
                elif (self.hparams_f.finetune_dataset == "ptbxl_rhythm"):
                    ptb_xl_label = "label_rhythm"
                
                lbl_itos = np.array(lbl_itos[ptb_xl_label])
                
                def multihot_encode(x, num_classes):
                    res = np.zeros(num_classes, dtype = np.float32)
                    for y in x:
                        res[y] = 1
                    return res
                    
                df_mapped["label"] = df_mapped[ptb_xl_label + "_filtered_numeric"].apply(lambda x: multihot_encode(x, len(lbl_itos)))

            elif (self.hparams_f.finetune and self.hparams_f.finetune_dataset.startswith("zheng")):
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
                
            elif (self.hparams_f.finetune and self.hparams_f.finetune_dataset.startswith("ribeiro")):                   
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
            max_fold_id = df_mapped.strat_fold.max() #unfortunately 1-based for PTB-XL; sometimes 100 (Ribeiro)

            df_train = df_mapped[df_mapped.strat_fold < (max_fold_id - 1 if self.hparams_f.finetune else max_fold_id)]
            df_val = df_mapped[df_mapped.strat_fold == (max_fold_id - 1 if self.hparams_f.finetune else max_fold_id)]
            if (self.hparams_f.finetune):
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
                    transforms = tfms_cpc,
                    annotation = False,
                    col_lbl = "label" if self.hparams_f.finetune else None,
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
                    transforms = tfms_cpc,
                    annotation = False,
                    col_lbl = "label" if self.hparams_f.finetune else None,
                    memmap_filename = target_folder/("memmap.npy"),
                    normalize_option = self.hparams_f.normalize,
                ),
            )
            if (self.hparams_f.finetune):
                test_datasets.append(
                    TimeseriesDatasetCrops(
                        df_test,
                        self.hparams_f.input_size,
                        num_classes = len(lbl_itos),
                        data_folder = target_folder,
                        chunk_length = chunk_length_valtest,
                        min_chunk_length = self.hparams_f.input_size, 
                        stride = stride_valtest,
                        transforms = tfms_cpc,
                        annotation = False,
                        col_lbl = "label",
                        memmap_filename = target_folder/("memmap.npy"),
                        normalize_option = self.hparams_f.normalize,
                    ),
                )
            
            print("\n", target_folder)
            print("train dataset: ", len(train_datasets[-1]), "samples")
            print("val dataset: ", len(val_datasets[-1]), "samples")
            if (self.hparams_f.finetune):
                print("test dataset: ", len(test_datasets[-1]), "samples")

        if (len(train_datasets) > 1): # multiple data folders
            print("\nCombined: ")
            self.train_dataset = ConcatDataset(train_datasets)
            self.val_dataset = ConcatDataset(val_datasets)
            print("train dataset: ", len(self.train_dataset), "samples")
            print("val dataset: ", len(self.val_dataset), "samples")
            if (self.hparams_f.finetune):
                self.test_dataset = ConcatDataset(test_datasets)
                print("test dataset: ", len(self.test_dataset), "samples")
        else: # just a single data folder
            self.train_dataset = train_datasets[0]
            self.val_dataset = val_datasets[0]
            if (self.hparams_f.finetune):
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
        if (self.hparams_f.finetune): # multiple val dataloaders
            return [
                DataLoader(self.val_dataset, batch_size = self.hparams_f.batch_size, num_workers = 4, worker_init_fn = seed_worker),
                DataLoader(self.test_dataset, batch_size = self.hparams_f.batch_size, num_workers = 4, worker_init_fn = seed_worker),
            ]
        else:
            return DataLoader(self.val_dataset, batch_size = self.hparams_f.batch_size, num_workers = 4, worker_init_fn = seed_worker)

    def configure_optimizers(self):
        if (self.hparams_f.optimizer == "sgd"):
            opt = torch.optim.SGD
        elif (self.hparams_f.optimizer == "adam"):
            opt = torch.optim.AdamW
        else:
            raise NotImplementedError("Unknown Optimizer.")
        
        if (self.hparams_f.finetune and (self.hparams_f.linear_eval or self.hparams_f.train_head_only)):
            optimizer = opt(self.model_cpc.head.parameters(), self.lr, weight_decay = self.hparams_f.weight_decay)
        elif (self.hparams_f.finetune and self.hparams_f.discriminative_lr_factor != 1.): # discrimative lrs
            optimizer = opt(
                [
                    {
                        "params": self.model_cpc.encoder.parameters(), 
                        "lr": self.lr * self.hparams_f.discriminative_lr_factor * self.hparams_f.discriminative_lr_factor},
                    {
                        "params": self.model_cpc.rnn.parameters(), 
                        "lr": self.lr * self.hparams_f.discriminative_lr_factor,
                    },
                    {
                        "params": self.model_cpc.head.parameters(), 
                        "lr": self.lr,
                    },
                ],
                self.hparams_f.lr, 
                weight_decay = self.hparams_f.weight_decay,
            )
        else:
            optimizer = opt(self.parameters(), self.lr, weight_decay = self.hparams_f.weight_decay)

        return optimizer
        
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


#####################################################################################################
# ARGPARSERS
#####################################################################################################
def add_model_specific_args(parser):
    parser.add_argument("--input-channels", type = int, default = 12)
    parser.add_argument("--normalize", dest = "normalize", type = int, help = "select normalization mode", default = "0")
    parser.add_argument('--mlp', action = 'store_true', help = "False: original CPC True: as in SimCLR")
    parser.add_argument('--bias', action = 'store_true', help = "original CPC: no bias")
    parser.add_argument("--n-hidden", type = int, default = 512)
    parser.add_argument("--gru", action = "store_true")
    parser.add_argument("--n-layers", type = int, default = 2)
    parser.add_argument("--steps-predicted", dest = "steps_predicted", type = int, default = 12)
    parser.add_argument("--n-false-negatives", dest = "n_false_negatives", type = int, default = 128)
    parser.add_argument("--skip-encoder", action = "store_true", 
                        help = "disable the convolutional encoder i.e. just RNN; for testing")
    parser.add_argument("--fc-encoder", action = "store_true", 
                        help = "use a fully connected encoder (as opposed to an encoder with strided convs)")
    parser.add_argument("--negatives-from-same-seq-only", action = "store_true", 
                        help = "only draw false negatives from same sequence (as opposed to drawing from everywhere)")
    parser.add_argument("--no-bn-encoder", action = "store_true", help = "switch off batch normalization in encoder")
    parser.add_argument("--dropout-head", type = float, default = 0.5)
    parser.add_argument("--train-head-only", action = "store_true", 
                        help = "freeze everything except classification head (note: --linear-eval defaults to no hidden layer in classification head)")
    parser.add_argument("--lin-ftrs-head", type = str, default = "[512]", help = "hidden layers in the classification head")
    parser.add_argument('--no-bn-head', action = 'store_true', help = "use no batch normalization in classification head")
    return parser


def add_default_args():
    parser = argparse.ArgumentParser(description = 'PyTorch Lightning CPC Training')
    parser.add_argument('--data', metavar = 'DIR', type = str, help = 'path(s) to dataset',action = 'append')
    parser.add_argument('--epochs', default = 30, type = int, metavar = 'N', help = 'number of total epochs to run')
    parser.add_argument('--batch-size', default = 64, type = int, metavar = 'N',
                        help = 'mini-batch size (default: 256), this is the total '
                               'batch size of all GPUs on the current node when '
                               'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default = 1e-3, type = float, metavar = 'LR', help = 'initial learning rate', dest = 'lr')
    parser.add_argument('--wd', '--weight-decay', default = 1e-3, type = float, metavar = 'W', help = 'weight decay (default: 0.)', dest = 'weight_decay')
    parser.add_argument('--resume', default = '', type = str, metavar = 'PATH', help = 'path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default = '', type = str, metavar = 'PATH', help = 'path to pretrained checkpoint (default: none)')
    parser.add_argument('--optimizer', default = 'adam', help = 'sgd/adam') # was sgd
    parser.add_argument('--output-path', default = '.', type = str,dest="output_path", help = 'output path')
    parser.add_argument('--metadata', default = '', type = str, help = 'metadata for output')
    parser.add_argument("--gpus", type = int, default = 1, help = "number of gpus")
    parser.add_argument("--num-nodes", dest = "num_nodes", type = int, default = 1, help = "number of compute nodes")
    parser.add_argument("--precision", type = int, default = 16, help = "16/32")
    parser.add_argument("--distributed-backend", dest = "distributed_backend", type = str, default = None, help = "None/ddp")
    parser.add_argument("--accumulate", type = int, default = 1, help = "accumulate grad batches (total-bs = accumulate-batches * bs)")
    parser.add_argument("--input-size", dest = "input_size", type = int, default = 16000)
    parser.add_argument("--finetune", action = "store_true", help = "finetuning (downstream classification task)",  default = False)
    parser.add_argument("--linear-eval", action = "store_true", help = "linear evaluation instead of full finetuning",  default = False)
    parser.add_argument("--finetune-dataset", type = str, help = "thew/ptbxl_super/ptbxl_all", default = "thew")
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
    hparams.executable = "cpc"
    seed = 42
       
    if hparams.finetune:
        setup_seed(seed)

        if not os.path.exists(os.path.join(hparams.output_path)):
            os.makedirs(os.path.join(hparams.output_path))
        
        model = LightningCPC(hparams, seed)
    
        if (hparams.pretrained != ""):
            print("Loading pretrained weights from", hparams.pretrained)
            model.load_weights_from_checkpoint(hparams.pretrained)

        logger = TensorBoardLogger(
            save_dir = os.path.join(hparams.output_path),
            version = "version_0", # hparams.metadata.split(":")[0],
            name = "")
        print("Output directory:", logger.log_dir) 

        checkpoint_callback = ModelCheckpoint(
            dirpath = os.path.join(logger.log_dir,"best_model"), # hparams.output_path
            save_top_k = 1,
		    save_last = True,
            verbose = True,
            monitor = 'macro_auc_agg0' if hparams.finetune else 'val_loss', # val_loss/dataloader_idx_0
            mode = 'max' if hparams.finetune else 'min',
            filename = '',
        )
        lr_monitor = LearningRateMonitor()

        trainer = pl.Trainer(
            # overfit_batches = 0.01,
            auto_lr_find = hparams.lr_find,
            accumulate_grad_batches = hparams.accumulate,
            max_epochs = hparams.epochs,
            min_epochs = hparams.epochs,
            default_root_dir = os.path.join(hparams.output_path),
            num_sanity_val_steps = 0,
            logger = logger,
            callbacks = [checkpoint_callback], # lr_monitor],
            benchmark = False,
    
            gpus = hparams.gpus,
            num_nodes = hparams.num_nodes,
            precision = hparams.precision,
            strategy = hparams.distributed_backend,

            enable_progress_bar = False,
            resume_from_checkpoint = None if hparams.resume == "" else hparams.resume,
            deterministic = False,
        )
        
        if (hparams.lr_find): # lr find
            trainer.tune(model)
        
        trainer.fit(model)
    else:
        setup_seed(seed)
        model = LightningCPC(hparams, seed)

        logger = TensorBoardLogger(
            save_dir = hparams.output_path,
            version = "version_0", # hparams.metadata.split(":")[0],
            name = "")
        print("Output directory:", logger.log_dir)

        checkpoint_callback = ModelCheckpoint(
            dirpath = os.path.join(logger.log_dir, "best_model"), # hparams.output_path
            save_top_k = 1,
		    save_last = True,
            verbose = True,
            monitor = 'macro_auc_agg0' if hparams.finetune else 'val_loss', # val_loss/dataloader_idx_0
            mode = 'max' if hparams.finetune else 'min',
            filename = '',
        )
        lr_monitor = LearningRateMonitor()

        trainer = pl.Trainer(
            # overfit_batches = 0.01,
            auto_lr_find = hparams.lr_find,
            accumulate_grad_batches = hparams.accumulate,
            max_epochs = hparams.epochs,
            min_epochs = hparams.epochs,
            default_root_dir = hparams.output_path,
            num_sanity_val_steps = 0,
            logger = logger,
            callbacks = [checkpoint_callback], # lr_monitor],
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