import yaml
import tensorboard
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import sys
import csv
import argparse
import pickle
import random
from models.resnet_simclr import ResNetSimCLR
from models.cpc import CPCModel
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from clinical_ts.simclr_dataset_wrapper import SimCLRDataSetWrapper
from clinical_ts.eval_utils_cafa import eval_scores, eval_scores_bootstrap
from clinical_ts.timeseries_utils import aggregate_predictions
import pdb
from copy import deepcopy
from os.path import join, isdir
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def setup_seed(seed: int = 42):
    random.seed(seed)                # Python random
    np.random.seed(seed)              # Numpy random
    torch.manual_seed(seed)           # Torch CPU
    torch.cuda.manual_seed(seed)      # Torch GPU (if available)
    torch.cuda.manual_seed_all(seed)  # Torch all GPUs (if multi-GPU)
    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser("Finetuning tests")
    parser.add_argument("--model_file")
    parser.add_argument("--method")
    parser.add_argument("--dataset", nargs = "+", default = "./data/ptb_xl_fs100")
    parser.add_argument("--finetune-dataset", type = str, help = "thew/ptbxl_super/ptbxl_all", default = "thew")
    parser.add_argument("--batch_size", type = int, default = 512)
    parser.add_argument("--num_classes", type = int, default = 512)
    parser.add_argument("--discriminative_lr", default = False, action = "store_true")
    parser.add_argument("--num_workers", type = int, default = 8)
    parser.add_argument("--hidden", default = False, action = "store_true")
    parser.add_argument("--lr_schedule", default = "{}")
    parser.add_argument('--output-path', default = '.', type = str, dest = "output_path", help = 'output path')
    parser.add_argument("--use_pretrained", default = False, action = "store_true")
    parser.add_argument("--linear_evaluation", default = False, action = "store_true", help = "use linear evaluation")
    parser.add_argument("--test_noised", default = False, action = "store_true", help = "validate also on a noisy dataset")
    parser.add_argument("--noise_level", default = 1, type = int, help = "level of noise induced to the second validations set")
    parser.add_argument("--folds", default = 8, type = int, help = "number of folds used in finetuning (between 1-8)")
    parser.add_argument("--tag", default = "")
    parser.add_argument("--eval_only", action = "store_true", default = False, help = "only evaluate mode")
    parser.add_argument("--load_finetuned", action = "store_true", default = False)
    parser.add_argument("--test", action = "store_true", default = False)
    parser.add_argument("--verbose", action = "store_true", default = False)
    parser.add_argument("--cpc", action = "store_true", default = False)
    parser.add_argument("--model_location")
    parser.add_argument("--l_epochs", type = int, default = 0, help = "number of head-only epochs (these are performed first)")
    parser.add_argument("--f_epochs", type = int, default = 0, help = "number of finetuning epochs (these are perfomed after head-only training")
    parser.add_argument("--normalize", action = "store_true", default = False, help = "normalize dataset with ptbxl mean and std")
    parser.add_argument("--bn_head", action = "store_true", default = False)
    parser.add_argument("--ps_head", type = float, default = 0.0)
    parser.add_argument("--conv_encoder", action = "store_true", default = False)
    parser.add_argument("--base_model", default = "xresnet1d50")
    parser.add_argument("--widen", default = 1, type = int, help = "use wide xresnet1d50")
    args = parser.parse_args()
    return args


def get_new_state_dict(init_state_dict, lightning_state_dict, method = "simclr"):
    # in case of moco model
    from collections import OrderedDict
    # lightning_state_dict = lightning_state_dict["state_dict"]
    new_state_dict = OrderedDict()
    if method != "cpc":
        if method == "moco": 
            for key in init_state_dict:
                l_key = "encoder_q." + key
                if l_key in lightning_state_dict.keys():
                    new_state_dict[key] = lightning_state_dict[l_key]
        elif method == "simclr":
            for key in init_state_dict:
                if "features" in key:
                    l_key = key.replace("features", "encoder.features")
                if l_key in lightning_state_dict.keys():
                    new_state_dict[key] = lightning_state_dict[l_key]
        elif method == "swav":

            for key in init_state_dict:
                if "features" in key:
                    l_key = key.replace("features", "model.features")
                if l_key in lightning_state_dict.keys():
                    new_state_dict[key] = lightning_state_dict[l_key]
        elif method == "byol":
            for key in init_state_dict:
                l_key = "online_network.encoder." + key
                if l_key in lightning_state_dict.keys():
                    new_state_dict[key] = lightning_state_dict[l_key]
        else:
            raise("method unknown")
        new_state_dict["l1.weight"] = init_state_dict["l1.weight"]
        new_state_dict["l1.bias"] = init_state_dict["l1.bias"]
        if "l2.weight" in init_state_dict.keys():
            new_state_dict["l2.weight"] = init_state_dict["l2.weight"]
            new_state_dict["l2.bias"] = init_state_dict["l2.bias"]

        assert(len(init_state_dict) == len(new_state_dict))
    else:
        for key in init_state_dict:
            l_key = "model_cpc." + key
            if l_key in lightning_state_dict.keys():
                new_state_dict[key] = lightning_state_dict[l_key]
            if "head" in key:
                new_state_dict[key] = init_state_dict[key]
    return new_state_dict


def adjust(model, num_classes, hidden = False):
    in_features = model.l1.in_features
    last_layer = torch.nn.modules.linear.Linear(in_features, num_classes).to(device)
    if hidden:
        model.l1 = torch.nn.modules.linear.Linear(in_features, in_features).to(device)
        model.l2 = last_layer
    else:
        model.l1 = last_layer

    def def_forward(self):
        def new_forward(x):
            h = self.features(x)
            h = h.squeeze()

            x = self.l1(h)
            if hidden:
                x = F.relu(x)
                x = self.l2(x)
            return x
        return new_forward

    model.forward = def_forward(model)


def configure_optimizer(
        model, 
        batch_size, 
        head_only = False, 
        discriminative_lr = False, 
        base_model = "xresnet1d", 
        optimizer = "adam", 
        discriminative_lr_factor = 1,
    ):
    loss_fn = F.binary_cross_entropy_with_logits
    if base_model == "xresnet1d":
        wd = 1e-1
        if head_only:
            lr = (8e-3 * (batch_size / 256))
            optimizer = torch.optim.AdamW(model.l1.parameters(), lr = lr, weight_decay = wd)
        else:
            lr = 0.01
            if not discriminative_lr:
                optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = wd)
            else:
                param_dict = dict(model.named_parameters())
                keys = param_dict.keys()
                weight_layer_nrs = set()
                for key in keys:
                    if "features" in key:
                        # parameter names have the form features.x
                        weight_layer_nrs.add(key[9])
                weight_layer_nrs = sorted(weight_layer_nrs, reverse = True)
                features_groups = []
                while len(weight_layer_nrs) > 0:
                    if len(weight_layer_nrs) > 1:
                        features_groups.append(
                            list(
                                filter(
                                    lambda x: "features." + weight_layer_nrs[0] in x or "features." + weight_layer_nrs[1] in x, keys
                                )
                            )
                        )
                        del weight_layer_nrs[:2]
                    else:
                        features_groups.append(
                            list(filter(lambda x: "features." + weight_layer_nrs[0] in x, keys)))
                        del weight_layer_nrs[0]
                # filter linear layers
                linears = list(filter(lambda x: "l" in x, keys))
                groups = [linears] + features_groups
                optimizer_param_list = []
                tmp_lr = lr

                for layers in groups:
                    layer_params = [param_dict[param_name] for param_name in layers]
                    optimizer_param_list.append({"params": layer_params, "lr": tmp_lr})
                    tmp_lr /= 4
                optimizer = torch.optim.AdamW(optimizer_param_list, lr = lr, weight_decay = wd)

        print("lr", lr)
        print("wd", wd)
        print("batch size", batch_size)

    elif base_model == "cpc":
        if (optimizer == "sgd"):
            opt = torch.optim.SGD
        elif (optimizer == "adam"):
            opt = torch.optim.AdamW
        else:
            raise NotImplementedError("Unknown Optimizer.")
        lr = 1e-4
        wd = 1e-3
        if (head_only):
            lr = 1e-3
            print("Linear eval: model head", model.head)
            optimizer = opt(model.head.parameters(), lr, weight_decay = wd)
        elif (discriminative_lr_factor != 1.):  # discrimative lrs
            optimizer = opt(
                [
                    {
                        "params": model.encoder.parameters(), 
                        "lr": lr * discriminative_lr_factor * discriminative_lr_factor,
                    },
                    {
                        "params": model.rnn.parameters(), 
                        "lr": lr * discriminative_lr_factor,
                    }, 
                    {
                        "params": model.head.parameters(), 
                        "lr": lr,
                    },
                ], 
                lr, 
                weight_decay = wd,
            )
            print("Finetuning: model head", model.head)
            print("discriminative lr: ", discriminative_lr_factor)
        else:
            lr = 1e-3
            print("normal supervised training")
            optimizer = opt(model.parameters(), lr, weight_decay = wd)
    else:
        raise("model unknown")
    return loss_fn, optimizer


def load_model(
        linear_evaluation, 
        num_classes, 
        use_pretrained, 
        discriminative_lr = False, 
        hidden = False, 
        conv_encoder = False, 
        bn_head = False, 
        ps_head = 0.5, 
        location = "runs/pretrained/simclr/checkpoints/best_encoder.ckpt", 
        method = "simclr", 
        base_model = "xresnet1d50", 
        out_dim = 16, 
        widen = 1,
        f_epochs = 0,
        l_epochs = 0,
    ):
    # if f_epochs != 0 and l_epochs == 0:
    #     pretrained_folder = os.path.join(location, "runs/")
    #     pretrained_file = [filename for filename in os.listdir(pretrained_folder) if filename.startswith("linear")]
    #     location = pretrained_folder + pretrained_file[0]

    discriminative_lr_factor = 1
    if use_pretrained:
        print("load model from " + location)
        discriminative_lr_factor = 0.1
        if base_model == "cpc":
            lightning_state_dict = torch.load(location, map_location = device)

            # num_head = np.sum([1 if 'proj' in f else 0 for f in lightning_state_dict.keys()])
            if linear_evaluation:
                lin_ftrs_head = []
                bn_head = False
                ps_head = 0.0
            else:
                if hidden:
                    lin_ftrs_head = [512]
                else:
                    lin_ftrs_head = []

            if conv_encoder:
                strides = [2, 2, 2, 2]
                kss = [10, 4, 4, 4]
            else:
                strides = [1] * 4
                kss = [1] * 4

            model = CPCModel(
                input_channels = 12, 
                strides = strides, 
                kss = kss, 
                features = [512] * 4, 
                n_hidden = 512, 
                n_layers = 2, 
                mlp = False, 
                lstm = True, 
                bias_proj = False,
                num_classes = num_classes, 
                skip_encoder = False, 
                bn_encoder = True, 
                lin_ftrs_head = lin_ftrs_head, 
                ps_head = ps_head, 
                bn_head = bn_head,
            ).to(device)

            if "state_dict" in lightning_state_dict.keys():
                print("load pretrained model")
                model_state_dict = get_new_state_dict(
                    model.state_dict(), lightning_state_dict["state_dict"], 
                    method = "cpc",
                )
            else:
                print("load already finetuned model")
                model_state_dict = lightning_state_dict
            model.load_state_dict(model_state_dict)
        else:
            model = ResNetSimCLR(base_model, out_dim, hidden = hidden, widen = widen).to(device)
            model_state_dict = torch.load(location, map_location = device)
            if "state_dict" in model_state_dict.keys():
                model_state_dict = model_state_dict["state_dict"]
            if "l1.weight" in model_state_dict.keys():  # load already fine-tuned model
                model_classes = model_state_dict["l1.weight"].shape[0]
                if model_classes != num_classes:
                    raise Exception(
                        "Loaded model has different output dim ({}) than needed ({})".format(model_classes, num_classes)
                    )
                adjust(model, num_classes, hidden = hidden)
                if not hidden and "l2.weight" in model_state_dict:
                    del model_state_dict["l2.weight"]
                    del model_state_dict["l2.bias"]
                model.load_state_dict(model_state_dict)
            else:  # load pretrained model
                base_dict = model.state_dict()
                model_state_dict = get_new_state_dict(base_dict, model_state_dict, method = method)
                model.load_state_dict(model_state_dict)
                adjust(model, num_classes, hidden = hidden)

    else:
        if "xresnet1d" in base_model:
            model = ResNetSimCLR(base_model, out_dim, hidden = hidden, widen = widen).to(device)
            adjust(model, num_classes, hidden = hidden)
        elif base_model == "cpc":
            if linear_evaluation:
                lin_ftrs_head = []
                bn_head = False
                ps_head = 0.0
            else:
                if hidden:
                    lin_ftrs_head = [512]
                else:
                    lin_ftrs_head = []

            if conv_encoder:
                strides = [2, 2, 2, 2]
                kss = [10, 4, 4, 4]
            else:
                strides = [1] * 4
                kss = [1] * 4

            model = CPCModel(
                input_channels = 12, 
                strides = strides, 
                kss = kss, 
                features = [512] * 4, 
                n_hidden = 512, 
                n_layers = 2, 
                mlp = False, 
                lstm = True, 
                bias_proj = False,
                num_classes = num_classes, 
                skip_encoder = False, 
                bn_encoder = True, 
                lin_ftrs_head = lin_ftrs_head, 
                ps_head = ps_head, 
                bn_head = bn_head,
            ).to(device)
        else:
            raise Exception("model unknown")

    return model


def evaluate(model, dataloader, idmap, lbl_itos, cpc = False):
    preds, targs = eval_model(model, dataloader, cpc = cpc)
    res, classify_report, accuracy_instance, macro_mAP, std_macro_mAP, sample_mAP, std_sample_mAP, \
    macro_f1, std_macro_f1, sample_f1, std_sample_f1, sample_accuracy, std_sample_accuracy = \
        eval_scores(targs, preds, classes = lbl_itos, parallel = True)
    
    preds_agg, targs_agg = aggregate_predictions(preds, targs, idmap)
    res_agg, classify_report_agg, accuracy_instance_agg, macro_mAP_agg, std_macro_mAP_agg, sample_mAP_agg, std_sample_mAP_agg, \
    macro_f1_agg, std_macro_f1_agg, sample_f1_agg, std_sample_f1_agg, sample_accuracy_agg, std_sample_accuracy_agg = \
        eval_scores(targs_agg, preds_agg, classes = lbl_itos, parallel = True)

    macro_auc = res["label_AUC"]["macro"]
    std_macro_auc = res["label_AUC"]["macro_std"]
    sample_auc = res["label_AUC"]["samples"]
    std_sample_auc = res["label_AUC"]["samples_std"]
    macro_auc_agg = res_agg["label_AUC"]["macro"]
    std_macro_auc_agg = res_agg["label_AUC"]["macro_std"]
    sample_auc_agg = res_agg["label_AUC"]["samples"]
    std_sample_auc_agg = res_agg["label_AUC"]["samples_std"]

    return preds, classify_report, classify_report_agg, \
        macro_auc, std_macro_auc, sample_auc, std_sample_auc, accuracy_instance, macro_mAP, std_macro_mAP, sample_mAP, std_sample_mAP, \
        macro_f1, std_macro_f1, sample_f1, std_sample_f1, sample_accuracy, std_sample_accuracy, \
        macro_auc_agg, std_macro_auc_agg, sample_auc_agg, std_sample_auc_agg, accuracy_instance_agg, macro_mAP_agg, std_macro_mAP_agg, sample_mAP_agg, std_sample_mAP_agg, \
        macro_f1_agg, std_macro_f1_agg, sample_f1_agg, std_sample_f1_agg, sample_accuracy_agg, std_sample_accuracy_agg


def set_train_eval(model, cpc, linear_evaluation):
    if linear_evaluation:
        if cpc:
            model.encoder.eval()
        else:
            model.features.eval()
    else:
        model.train()


def train_model(
        model, 
        train_loader, 
        valid_loader, 
        test_loader, 
        epochs, 
        loss_fn, 
        optimizer, 
        head_only = True, 
        linear_evaluation = False, 
        percentage = 1, 
        lr_schedule = None, 
        path_downstream = None, 
        save_model_at = None, 
        val_idmap = None, 
        test_idmap = None, 
        lbl_itos = None, 
        cpc = False,
    ):
    if head_only:
        if linear_evaluation:
            print("linear evaluation for {} epochs".format(epochs))
        else:
            print("head-only for {} epochs".format(epochs))
    else:
        print("fine tuning for {} epochs".format(epochs))

    if head_only:
        for key, param in model.named_parameters():
            if "l1." not in key and "head." not in key:
                param.requires_grad = False
        print("copying state dict before training for sanity check after training")

    else:
        for param in model.parameters():
            param.requires_grad = True
    if cpc:
        data_type = model.encoder[0][0].weight.type()
    else:
        data_type = model.features[0][0].weight.type()

    set_train_eval(model, cpc, linear_evaluation)
    state_dict_pre = deepcopy(model.state_dict())
    print("epoch", "batch", "loss\n========================")
    loss_per_epoch = []
    macro_auc_agg_per_epoch = []
    max_batches = len(train_loader)
    break_point = int(percentage*max_batches)
    best_macro_auc = 0
    best_macro_auc_agg = 0
    best_epoch = 0
    best_preds = None

    test_macro_auc = 0
    test_sample_auc = 0
    test_accuracy_instance = 0
    test_macro_mAP = 0
    test_sample_mAP = 0
    test_macro_f1 = 0
    test_sample_f1 = 0 
    test_sample_accuracy = 0

    test_macro_auc_agg = 0
    test_sample_auc_agg = 0
    test_accuracy_instance_agg = 0
    test_macro_mAP_agg = 0
    test_sample_mAP_agg = 0
    test_macro_f1_agg = 0 
    test_sample_f1_agg = 0
    test_sample_accuracy_agg = 0

    test_classify_report = 0
    test_classify_report_agg = 0

    writer = SummaryWriter(path_downstream)
    for epoch in tqdm(range(epochs)):
        if type(lr_schedule) == dict:
            if epoch in lr_schedule.keys():
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= lr_schedule[epoch]
        total_loss_one_epoch = 0
        for batch_idx, samples in enumerate(train_loader):
            if batch_idx == break_point:
                print("break at batch nr.", batch_idx)
                break
            data = samples[0].to(device).type(data_type)
            labels = samples[1].to(device).type(data_type)
            optimizer.zero_grad()
            preds = model(data)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss_one_epoch += loss.item()
            if(batch_idx % 100 == 0):
                print(epoch, batch_idx, loss.item())
                writer.add_scalar('train_loss', loss / 100, epoch * len(train_loader) + batch_idx)
        loss_per_epoch.append(total_loss_one_epoch)

        preds, classify_report, classify_report_agg, \
        macro_auc, std_macro_auc, sample_auc, std_sample_auc, accuracy_instance, macro_mAP, std_macro_mAP, sample_mAP, std_sample_mAP, \
        macro_f1, std_macro_f1, sample_f1, std_sample_f1, sample_accuracy, std_sample_accuracy, \
        macro_auc_agg, std_macro_auc_agg, sample_auc_agg, std_sample_auc_agg, accuracy_instance_agg, macro_mAP_agg, std_macro_mAP_agg, sample_mAP_agg, std_sample_mAP_agg, \
        macro_f1_agg, std_macro_f1_agg, sample_f1_agg, std_sample_f1_agg, sample_accuracy_agg, std_sample_accuracy_agg = \
            evaluate(model, valid_loader, val_idmap, lbl_itos, cpc = cpc)
        
        macro_auc_agg_per_epoch.append(macro_auc_agg)
        writer.add_scalar('macro_auc_agg', macro_auc_agg, epoch)
        writer.add_scalar('sample_auc_agg', sample_auc_agg, epoch)
        writer.add_scalar('accuracy_instance_agg', accuracy_instance_agg, epoch)
        writer.add_scalar('macro_mAP_agg', macro_mAP_agg, epoch)
        writer.add_scalar('sample_mAP_agg', sample_mAP_agg, epoch)
        writer.add_scalar('macro_f1_agg', macro_f1_agg, epoch)
        writer.add_scalar('sample_f1_agg', sample_f1_agg, epoch)  
        writer.add_scalar('sample_accuracy_agg', sample_accuracy_agg, epoch)       

        print("loss: ", total_loss_one_epoch)
        print("aggregated macro: ", macro_auc_agg)
        if macro_auc_agg > best_macro_auc_agg:
            torch.save(model.state_dict(), save_model_at)
            best_macro_auc_agg = macro_auc_agg
            best_macro_auc = macro_auc
            best_epoch = epoch
            best_preds = preds
            _,  test_classify_report, test_classify_report_agg, \
            test_macro_auc, test_std_macro_auc, test_sample_auc, test_std_sample_auc, test_accuracy_instance, test_macro_mAP, test_std_macro_mAP, test_sample_mAP, test_std_sample_mAP, \
            test_macro_f1, test_std_macro_f1, test_sample_f1, test_std_sample_f1, test_sample_accuracy, test_std_sample_accuracy, \
            test_macro_auc_agg, test_std_macro_auc_agg, test_sample_auc_agg, test_std_sample_auc_agg, test_accuracy_instance_agg, test_macro_mAP_agg, test_std_macro_mAP_agg, test_sample_mAP_agg, test_std_sample_mAP_agg, \
            test_macro_f1_agg, test_std_macro_f1_agg, test_sample_f1_agg, test_std_sample_f1_agg, test_sample_accuracy_agg, test_std_sample_accuracy_agg = \
                evaluate(model, test_loader, test_idmap, lbl_itos, cpc = cpc)

        set_train_eval(model, cpc, linear_evaluation)
    writer.close()
    if epochs > 0:
        sanity_check(model, state_dict_pre, linear_evaluation, head_only)
    return loss_per_epoch, macro_auc_agg_per_epoch, best_macro_auc, best_macro_auc_agg, \
        test_classify_report, test_classify_report_agg, \
        test_macro_auc, test_sample_auc, test_accuracy_instance, test_macro_mAP, test_sample_mAP, test_macro_f1, test_sample_f1, test_sample_accuracy, \
        test_macro_auc_agg, test_sample_auc_agg, test_accuracy_instance_agg, test_macro_mAP_agg, test_sample_mAP_agg, test_macro_f1_agg, test_sample_f1_agg, test_sample_accuracy_agg, \
        best_epoch, best_preds


def sanity_check(model, state_dict_pre, linear_evaluation, head_only):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading state dict for sanity check")
    state_dict = model.state_dict()
    if linear_evaluation:
        for k in list(state_dict.keys()):
            # only ignore fc layer
            if 'fc.' in k or 'head.' in k or 'l1.' in k:
                continue

            equals = (state_dict[k].cpu() == state_dict_pre[k].cpu()).all()
            if (linear_evaluation != equals):
                raise Exception('=> failed sanity check in {}'.format("linear_evaluation"))
    elif head_only:
        for k in list(state_dict.keys()):
            # only ignore fc layer
            if 'fc.' in k or 'head.' in k:
                continue

            equals = (state_dict[k].cpu() == state_dict_pre[k].cpu()).all()
            if (equals and "running_mean" in k):
                raise Exception('=> failed sanity check in {}'.format("head-only"))

    print("=> sanity check passed.")


def eval_model(model, valid_loader, cpc = False):
    if cpc:
        data_type = model.encoder[0][0].weight.type()
    else:
        data_type = model.features[0][0].weight.type()
    model.eval()
    preds = []
    targs = []
    with torch.no_grad():
        for batch_idx, samples in tqdm(enumerate(valid_loader)):
            data = samples[0].to(device).type(data_type)
            preds_tmp = torch.sigmoid(model(data))
            targs.append(samples[1])
            preds.append(preds_tmp.cpu())
        preds = torch.cat(preds).numpy()
        targs = torch.cat(targs).numpy()

    return preds, targs


def get_dataset(
        batch_size, 
        num_workers, 
        target_folder, 
        apply_noise = False, 
        percentage = 1.0, 
        folds = 8, 
        t_params = None, 
        test = False, 
        normalize = False,
    ):
    target_folder = target_folder[0]
    if apply_noise:
        transformations = ["BaselineWander", "PowerlineNoise", "EMNoise", "BaselineShift"]
        if normalize:
            transformations.append("Normalize")
        dataset = SimCLRDataSetWrapper(
            batch_size,
            num_workers,
            None,
            "(12, 1000)",
            None,
            target_folder,
            [target_folder],
            None,
            None,
            mode = "downstream", 
            transformations = transformations, 
            percentage = percentage, 
            folds = folds, 
            t_params = t_params, 
            test = test, 
            dataset_label = args.finetune_dataset,
        )
    else:
        if normalize:
            # always use PTB-XL stats
            transformations = ["Normalize"]
            dataset = SimCLRDataSetWrapper(
                batch_size,
                num_workers,
                None,
                "(12, 1000)",
                None,
                target_folder,
                [target_folder],
                None,
                None,
                mode = "downstream", 
                percentage = percentage, 
                folds = folds, 
                test = test, 
                transformations = transformations, 
                dataset_label = args.finetune_dataset,
            )
        else:
            dataset = SimCLRDataSetWrapper(
                batch_size,
                num_workers,
                None,
                "(12, 1000)",
                None,
                target_folder,
                [target_folder],
                None,
                None,
                mode = "downstream", 
                percentage = percentage, 
                folds = folds, 
                test = test, 
                dataset_label = args.finetune_dataset,
            )

    train_loader, valid_loader = dataset.get_data_loaders()
    return dataset, train_loader, valid_loader


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(os.path.join(args.output_path, "runs/")):
        os.makedirs(os.path.join(args.output_path, "runs/"))
        
    setup_seed(42)
    dataset, train_loader, _ = get_dataset(
        args.batch_size, 
        args.num_workers, 
        args.dataset, 
        folds = args.folds, 
        test = args.test, 
        normalize = args.normalize,
    )
    _, _, valid_loader = get_dataset(
        args.batch_size, 
        args.num_workers, 
        args.dataset, 
        folds = args.folds, 
        test = False, 
        normalize = args.normalize,
    )
    val_idmap = dataset.val_ds_idmap
    dataset, _, test_loader = get_dataset(
        args.batch_size, 
        args.num_workers, 
        args.dataset, 
        test = True, 
        normalize = args.normalize,
    )
    test_idmap = dataset.val_ds_idmap
    lbl_itos = dataset.lbl_itos

    tag = "f=" + str(args.folds) + "_" + args.tag
    tag = tag if args.use_pretrained else "ran_" + tag
    tag = "eval_" + tag if args.eval_only else tag
    model_tag = "finetuned" if args.load_finetuned else "ckpt"
    if args.test_noised:
        t_params_by_level = {
            1: {"bw_cmax": 0.05, "em_cmax": 0.25, "pl_cmax": 0.1, "bs_cmax": 0.5},
            2: {"bw_cmax": 0.1, "em_cmax": 0.5, "pl_cmax": 0.2, "bs_cmax": 1},
            3: {"bw_cmax": 0.1, "em_cmax": 1, "pl_cmax": 0.2, "bs_cmax": 2},
            4: {"bw_cmax": 0.2, "em_cmax": 1, "pl_cmax": 0.4, "bs_cmax": 2},
            5: {"bw_cmax": 0.2, "em_cmax": 1.5, "pl_cmax": 0.4, "bs_cmax": 2.5},
            6: {"bw_cmax": 0.3, "em_cmax": 2, "pl_cmax": 0.5, "bs_cmax": 3},
        }
        if args.noise_level not in t_params_by_level.keys():
            raise("noise level does not exist")
        t_params = t_params_by_level[args.noise_level]
        dataset, _, noise_valid_loader = get_dataset(
            args.batch_size, 
            args.num_workers, 
            args.dataset, 
            apply_noise = True, 
            t_params = t_params, 
            test = args.test,
        )
    else:
        noise_valid_loader = None
    losses, macros_auc, predss, result_macros_auc, result_macros_auc_agg, \
    test_macros_auc, test_samples_auc, test_accuracies_instance, test_macros_mAP, test_samples_mAP, test_macros_f1, test_samples_f1, test_sample_accuracies, \
    test_macros_auc_agg, test_samples_auc_agg, test_accuracies_instance_agg, test_macros_mAP_agg, test_samples_mAP_agg, test_macros_f1_agg, test_samples_f1_agg, test_sample_accuracies_agg, \
    noised_macros_auc, noised_macros_auc_agg = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    ckpt_epoch_lin = 0
    ckpt_epoch_fin = 0
    
    path_downstream = os.path.join(args.output_path, "runs/")
    save_model_at = path_downstream + "best_checkpoint"
        
    filename_test = path_downstream + "results.txt"
    filename_report = path_downstream + "report.txt"

    model = load_model(
        args.linear_evaluation, 
        args.num_classes, 
        args.use_pretrained or args.load_finetuned, 
        hidden = args.hidden,
        location = args.model_file, 
        discriminative_lr = args.discriminative_lr, 
        method = args.method,
        f_epochs = args.f_epochs,
        l_epochs = args.l_epochs,
    )
    loss_fn, optimizer = configure_optimizer(
        model, 
        args.batch_size, 
        head_only = True, 
        discriminative_lr = args.discriminative_lr, 
        discriminative_lr_factor = 0.1 if args.use_pretrained and args.discriminative_lr else 1,
    )
    if not args.eval_only:
        print("train model...")
        if not isdir(save_model_at):
            os.mkdir(save_model_at)

        l1, m1, bm, bm_agg, \
        test_classify_report, test_classify_report_agg, \
        test_macro_auc, test_sample_auc, test_accuracy_instance, test_macro_mAP, test_sample_mAP, test_macro_f1, test_sample_f1, test_sample_accuracy, \
        test_macro_auc_agg, test_sample_auc_agg, test_accuracy_instance_agg, test_macro_mAP_agg, test_sample_mAP_agg, test_macro_f1_agg, test_sample_f1_agg, test_sample_accuracy_agg, \
        ckpt_epoch_lin, preds = \
            train_model(
                model, 
                train_loader, 
                valid_loader, 
                test_loader, 
                args.l_epochs, 
                loss_fn,
                optimizer, 
                head_only = True, 
                linear_evaluation = args.linear_evaluation, 
                lr_schedule = args.lr_schedule, 
                path_downstream = path_downstream, 
                save_model_at = join(save_model_at, "linear.pt"),
                val_idmap = val_idmap, 
                test_idmap = test_idmap, 
                lbl_itos = lbl_itos, 
                cpc = (args.method == "cpc"),
            )

        if bm != 0:
            print("best macro after head-only training:", bm_agg)
        l2 = []
        m2 = []
        if args.f_epochs != 0:
            if args.l_epochs != 0:
                model = load_model(
                    False, 
                    args.num_classes, 
                    True, 
                    hidden = args.hidden, 
                    location = join(save_model_at, "linear.pt"), 
                    discriminative_lr = args.discriminative_lr, 
                    method = args.method,
                    f_epochs = args.f_epochs,
                    l_epochs = args.l_epochs,
                )
            loss_fn, optimizer = configure_optimizer(
                model, 
                args.batch_size, 
                head_only = False, 
                discriminative_lr = args.discriminative_lr, 
                discriminative_lr_factor = 0.1 if args.use_pretrained and args.discriminative_lr else 1,
            )
            l2, m2, bm, bm_agg, \
            test_classify_report, test_classify_report_agg, \
            test_macro_auc, test_sample_auc, test_accuracy_instance, test_macro_mAP, test_sample_mAP, test_macro_f1, test_sample_f1, test_sample_accuracy, \
            test_macro_auc_agg, test_sample_auc_agg, test_accuracy_instance_agg, test_macro_mAP_agg, test_sample_mAP_agg, test_macro_f1_agg, test_sample_f1_agg, test_sample_accuracy_agg, \
            ckpt_epoch_lin, preds = \
                train_model(
                    model, 
                    train_loader, 
                    valid_loader, 
                    test_loader, 
                    args.f_epochs, 
                    loss_fn,
                    optimizer, 
                    head_only = False, 
                    linear_evaluation = False, 
                    lr_schedule = args.lr_schedule, 
                    path_downstream = path_downstream, 
                    save_model_at = join(save_model_at, "finetuned.pt"),
                    val_idmap = val_idmap, 
                    test_idmap = test_idmap, 
                    lbl_itos = lbl_itos, 
                    cpc = (args.method == "cpc"),
                )
                
        losses.append(l1 + l2)
        macros_auc.append(m1 + m2)

        test_macros_auc.append(test_macro_auc)
        test_samples_auc.append(test_sample_auc)
        test_accuracies_instance.append(test_accuracy_instance)
        test_macros_mAP.append(test_macro_mAP)
        test_samples_mAP.append(test_sample_mAP)
        test_macros_f1.append(test_macro_f1)
        test_samples_f1.append(test_sample_f1)
        test_sample_accuracies.append(test_sample_accuracy)
        test_macros_auc_agg.append(test_macro_auc_agg)
        test_samples_auc_agg.append(test_sample_auc_agg)
        test_accuracies_instance_agg.append(test_accuracy_instance_agg)
        test_macros_mAP_agg.append(test_macro_mAP_agg)
        test_samples_mAP_agg.append(test_sample_mAP_agg)
        test_macros_f1_agg.append(test_macro_f1_agg)
        test_samples_f1_agg.append(test_sample_f1_agg)
        test_sample_accuracies_agg.append(test_sample_accuracy_agg)

        result_macros_auc.append(bm)
        result_macros_auc_agg.append(bm_agg)

    else:
        preds, eval_classify_report, eval_classify_report_agg, \
        eval_macro_auc, eval_std_macro_auc, eval_sample_auc, eval_std_sample_auc, eval_accuracy_instance, eval_macro_mAP, eval_std_macro_mAP, eval_sample_mAP, eval_std_sample_mAP, \
        eval_macro_f1, eval_std_macro_f1, eval_sample_f1, eval_std_sample_f1, eval_sample_accuracy, eval_std_sample_accuracy, \
        eval_macro_auc_agg, eval_std_macro_auc_agg, eval_sample_auc_agg, eval_std_sample_auc_agg, eval_accuracy_instance_agg, eval_macro_mAP_agg, eval_std_macro_mAP_agg, eval_sample_mAP_agg, eval_std_sample_mAP_agg, \
        eval_macro_f1_agg, eval_std_macro_f1_agg, eval_sample_f1_agg, eval_std_sample_f1_agg, eval_sample_accuracy_agg, eval_std_sample_accuracy_agg = \
            evaluate(model, test_loader, test_idmap, lbl_itos, cpc = (args.method == "cpc"))
        result_macros_auc.append(eval_macro_auc)
        result_macros_auc_agg.append(eval_macro_auc_agg)
        if args.verbose:
            print("macro:", eval_macro_auc)
    predss.append(preds)

    if noise_valid_loader is not None:
        _, noise_classify_report, noise_classify_report_agg, \
        noise_macro_auc, noise_std_macro_auc, noise_sample_auc, noise_std_sample_auc, noise_accuracy_instance, noise_macro_mAP, noise_std_macro_mAP, noise_sample_mAP, noise_std_sample_mAP, \
        noise_macro_f1, noise_std_macro_f1, noise_sample_f1, noise_std_sample_f1, noise_sample_accuracy, noise_std_sample_accuracy, \
        noise_macro_auc_agg, noise_std_macro_auc_agg, noise_sample_auc_agg, noise_std_sample_auc_agg, noise_accuracy_instance_agg, noise_macro_mAP_agg, noise_std_macro_mAP_agg, noise_sample_mAP_agg, noise_std_sample_mAP_agg, \
        noise_macro_f1_agg, noise_std_macro_f1_agg, noise_sample_f1_agg, noise_std_sample_f1_agg, noise_sample_accuracy_agg, noise_std_sample_accuracy_agg = \
            evaluate(model, noise_valid_loader, val_idmap, lbl_itos)
        noised_macros_auc.append(noise_macro_auc)
        noised_macros_auc_agg.append(noise_macro_auc_agg)
    res = {
        "epochs": args.l_epochs + args.f_epochs, 
        "model_location": args.model_location,
        "losses": losses, 
        "macros": macros_auc, 
        "predss": predss, 
        "result_macros": result_macros_auc, 
        "result_macros_agg": result_macros_auc_agg,
        "noised_macros": noised_macros_auc, 
        "noised_macros_agg": noised_macros_auc_agg, 
        "ckpt_epoch_lin": ckpt_epoch_lin, 
        "ckpt_epoch_fin": ckpt_epoch_fin,
        "discriminative_lr": args.discriminative_lr, 
        "hidden": args.hidden, 
        "lr_schedule": args.lr_schedule,
        "use_pretrained": args.use_pretrained, 
        "linear_evaluation": args.linear_evaluation, 
        "loaded_finetuned": args.load_finetuned,
        "eval_only": args.eval_only, 
        "noise_level": args.noise_level, 
        "test_noised": args.test_noised, 
        "normalized": args.normalize,
    }

    with open(filename_test, 'a') as file:
        file.write(
            f"test_macros_auc: {test_macros_auc[0]}\n"
            f"test_samples_auc: {test_samples_auc[0]}\n"
            f"test_accuracies_instance: {test_accuracies_instance[0]}\n"
            f"test_sample_accuracies: {test_sample_accuracies[0]}\n"
            f"test_macros_f1: {test_macros_f1[0]}\n"
            f"test_samples_f1: {test_samples_f1[0]}\n"
            f"test_macros_mAP: {test_macros_mAP[0]}\n"
            f"test_samples_mAP: {test_samples_mAP[0]}\n\n\n\n"
            f"test_macros_auc_agg: {test_macros_auc_agg[0]}\n"
            f"test_samples_auc_agg: {test_samples_auc_agg[0]}\n"
            f"test_accuracies_instance_agg: {test_accuracies_instance_agg[0]}\n"
            f"test_sample_accuracies_agg: {test_sample_accuracies_agg[0]}\n"
            f"test_macros_f1_agg: {test_macros_f1_agg[0]}\n"
            f"test_samples_f1_agg: {test_samples_f1_agg[0]}\n"
            f"test_macros_mAP_agg: {test_macros_mAP_agg[0]}\n"
            f"test_samples_mAP_agg: {test_samples_mAP_agg[0]}\n",
        )
    with open(filename_report, 'a') as file_report:  # 'a' mode creates the file_report if it does not exist
        file_report.write(
            f"{test_classify_report}\n\n\n\n"
            f"{test_classify_report_agg}\n"
        )
    print("Done!")