"""
Vision Transformer adapted to 1D ECG signals

Intended fpr vanilla, supervised training
"""
import os
import re

import torch
from torch import nn
from transformers import PretrainedConfig
from vit_pytorch import ViT
from vit_pytorch.recorder import Recorder
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns

from ecg_transformer.util import *
from ecg_transformer.util.models import ModelOutput
import ecg_transformer.util.ecg as ecg_util
from ecg_transformer.preprocess import get_ptbxl_dataset
from ecg_transformer.chore import barplot


class EcgVitConfig(PretrainedConfig):
    pattern_model_name = re.compile(rf'^(?P<name>\S+)-(?P<size>\S+)$')

    def __init__(
            self,
            max_signal_length: int = 2560,
            patch_size: int = 64,
            num_channels: int = 12,
            hidden_size: int = 512,  # Default parameters are 2/3 of ViT base model sizes
            num_hidden_layers: int = 8,
            num_attention_heads: int = 8,
            intermediate_size: int = 2048,
            hidden_dropout_prob: float = 0.1,
            attention_probs_dropout_prob: float = 0.1,
            num_class: int = 71,  # specific to ECG supervised classification
            **kwargs
    ):
        self.max_signal_length = max_signal_length
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_class = num_class
        super().__init__(**kwargs)
        self.size = None

    @classmethod
    def from_defined(cls, model_name):
        """
        A few model sizes I defined
        """
        ca(model_name = model_name)
        conf = cls()
        m = cls.pattern_model_name.match(model_name)
        nm, size = m.group('name'), m.group('size')
        conf.size = size
        assert nm == 'ecg-vit'
        if size == 'debug':
            conf.hidden_size = 64
            conf.num_hidden_layers = 4
            conf.num_attention_heads = 4
            conf.intermediate_size = 256
        elif size == 'tiny':
            conf.hidden_size = 256
            conf.num_hidden_layers = 4
            conf.num_attention_heads = 4
            conf.intermediate_size = 1024
        elif size == 'small':
            conf.hidden_size = 512
            conf.num_hidden_layers = 8
            conf.num_attention_heads = 8
            conf.intermediate_size = 2048
        elif size == 'base':
            conf.hidden_size = 768
            conf.num_hidden_layers = 12
            conf.num_attention_heads = 12
            conf.intermediate_size = 3072
        elif size == 'large':
            conf.hidden_size = 1024
            conf.num_hidden_layers = 24
            conf.num_attention_heads = 16
            conf.intermediate_size = 4096
        return conf


class EcgVit(nn.Module):
    def __init__(self, num_class: int = 71, config = EcgVitConfig(), loss_reduction: str = 'mean'):
        super().__init__()
        hd_sz, n_head = config.hidden_size, config.num_attention_heads
        assert hd_sz % n_head == 0
        dim_head = hd_sz // n_head
        self.config = config
        _md_args = dict(
            image_size=(1, self.config.max_signal_length),  # height is 1
            patch_size=(1, self.config.patch_size),
            num_classes = num_class,
            dim = self.config.hidden_size,
            depth = self.config.num_hidden_layers,
            heads = self.config.num_attention_heads,
            mlp_dim = self.config.intermediate_size,
            pool = 'cls',
            channels = self.config.num_channels,
            dim_head = dim_head,
            dropout = self.config.hidden_dropout_prob,
            emb_dropout = self.config.attention_probs_dropout_prob
        )
        self.vit = ViT(**_md_args)
        self._loss_reduction = loss_reduction
        self.loss_fn = nn.BCEWithLogitsLoss(reduction = loss_reduction)  # TODO: more complex loss, e.g. weighting?
        self.loss_weight = None

        C, L = self.config.num_channels, self.config.max_signal_length
        cls_nm = self.__class__.__qualname__
        n_pch, n_l, n_h = L // self.config.patch_size, self.config.num_hidden_layers, self.config.num_attention_heads
        self.meta = {
            'name': cls_nm, 'input shape': f'{C} x {L}', '#patch': n_pch, '#layer': n_l, '#head': n_h
        }
        self.meta_str = log_dict_p({'nm': cls_nm, 'in-sp': f'{C}x{L}', '#p': n_pch, '#l': n_l, '#h': n_h})

    def to_str(self):
        return f'{self.__class__.__qualname__}, {self.config.size}'

    @property
    def loss_reduction(self):
        return self._loss_reduction

    @loss_reduction.setter
    def loss_reduction(self, r):
        self.loss_fn.reduction = self._loss_reduction = r

    def forward(self, sample_values: torch.FloatTensor, labels: torch.LongTensor = None):
        # import ipdb; ipdb.set_trace()
        logits = self.vit(sample_values.unsqueeze(-2))   # Add dummy height dimension
        loss = None
        if labels is not None:
            if self.loss_weight:  # modify the loss function each call
                weight = torch.tensor(self.loss_weight, device = labels.device)
                # Map weights by each label
                self.loss_fn = nn.BCEWithLogitsLoss(weight = weight[labels.long()], reduction = self.loss_reduction)
            loss = self.loss_fn(input = logits, target = labels)
        return ModelOutput(loss = loss, logits = logits)


def load_trained(model_key: str = 'ecg-vit-base'):
    model = EcgVit(config=EcgVitConfig.from_defined(model_key))

    fnm = 'model - model={nm=EcgVit, in-sp=12x2560, #p=40, #l=12, #h=12}, ' \
          'n=17441, a=0.0003, dc=0.01, bsz=256, n_ep=32, ep8.pt'
    checkpoint_path = os.path.join(PATH_BASE, DIR_PROJ, DIR_MDL, '2022-04-15_23-48-47', fnm)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=True)  # Need the pl wrapper cos that's how the model is saved
    model.eval()
    return model



if __name__ == '__main__':
    import pickle

    from icecream import ic

    ev = EcgVit()
    sigs = torch.randn(4, 12, 2560)
    print(ev.vit.to_patch_embedding(torch.randn(4, 12, 1, 2560)).shape)

    labels_ = torch.zeros(4, 71)
    labels_[[0, 0, 1, 2, 3, 3, 3], [0, 1, 2, 3, 4, 5, 6]] = 1
    print(labels_)
    loss_, logits_ = ev(sigs, labels_)
    print(sigs.shape, loss_, logits_.shape)

    # mdl = load_trained()