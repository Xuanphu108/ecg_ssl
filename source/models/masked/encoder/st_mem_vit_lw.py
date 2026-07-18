from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from models.masked.encoder.vit import TransformerBlock, Attention


__all__ = ['ST_MEM_ViT_lw']

class ST_MEM_ViT_lw(nn.Module):
    def __init__(
        self,
        seq_len: int,
        patch_size: int,
        num_leads: int,
        num_classes: Optional[int] = None,
        width: int = 768,
        depth: int = 12,
        mlp_dim: int = 3072,
        heads: int = 12,
        dim_head: int = 64,
        qkv_bias: bool = True,
        drop_out_rate: float = 0.,
        attn_drop_out_rate: float = 0.,
        drop_path_rate: float = 0.,
    ):
        super().__init__()
        assert seq_len % patch_size == 0, 'The sequence length must be divisible by the patch size.'
        self._repr_dict = {
            'seq_len': seq_len,
            'patch_size': patch_size,
            'num_leads': num_leads,
            'num_classes': num_classes if num_classes is not None else 'None',
            'width': width,
            'depth': depth,
            'mlp_dim': mlp_dim,
            'heads': heads,
            'dim_head': dim_head,
            'qkv_bias': qkv_bias,
            'drop_out_rate': drop_out_rate,
            'attn_drop_out_rate': attn_drop_out_rate,
            'drop_path_rate': drop_path_rate,
        }
        self.width = width
        self.depth = depth
        self.avg = nn.AvgPool1d(depth, depth)        
        # embedding layers
        num_patches = seq_len // patch_size
        patch_dim = patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b c n p', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, width),
            nn.LayerNorm(width),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, width))
        self.sep_embedding = nn.Parameter(torch.randn(width))
        self.lead_embeddings = nn.ParameterList(nn.Parameter(torch.randn(width)) for _ in range(num_leads))
        
        # transformer layers
        drop_path_rate_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(
                input_dim = width,
                output_dim = width,
                hidden_dim = mlp_dim,
                heads = heads,
                dim_head = dim_head,
                qkv_bias = qkv_bias,
                drop_out_rate = drop_out_rate,
                attn_drop_out_rate = attn_drop_out_rate,
                drop_path_rate = drop_path_rate_list[i],
            )
            self.add_module(f'block{i}', block)
        self.dropout = nn.Dropout(drop_out_rate)
        self.norm = nn.LayerNorm(width)

        # classifier head
        # self.head = nn.Identity() if num_classes is None else nn.Linear(width, num_classes)
        if num_classes is None:
            self.head = nn.Identity()
        else:
            layers_head = [
                nn.BatchNorm1d(self.width, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats =  True),
                nn.Dropout(p = 0.25, inplace = False),
                nn.Linear(in_features = self.width, out_features = 256, bias = True),
                nn.ReLU(inplace = True),
                nn.BatchNorm1d(256, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats =  True),
                nn.Dropout(p = 0.5, inplace = False),
                nn.Linear(in_features = 256, out_features = num_classes, bias = True)
            ]
            self.head = nn.Sequential(*layers_head)

    def reset_head(self, num_classes: Optional[int] = None):
        del self.head
        if num_classes is None:
            self.head = nn.Identity()
        else:
            layers_head = [
                nn.BatchNorm1d(self.width, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats =  True),
                nn.Dropout(p = 0.25, inplace = False),
                nn.Linear(in_features = self.width, out_features = 256, bias = True),
                nn.ReLU(inplace = True),
                nn.BatchNorm1d(256, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats =  True),
                nn.Dropout(p = 0.5, inplace = False),
                nn.Linear(in_features = 256, out_features = num_classes, bias = True)
            ]
            self.head = nn.Sequential(*layers_head)

    def forward_encoding(self, series):
        num_leads = series.shape[1]
        if num_leads > len(self.lead_embeddings):
            raise ValueError(f'Number of leads ({num_leads}) exceeds the number of lead embeddings')
        
        x = self.to_patch_embedding(series)
        b, _, n, _ = x.shape
        x = x + self.pos_embedding[:, 1:n + 1, :].unsqueeze(1)

        # lead indicating modules
        sep_embedding = self.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim = 2)
        lead_embeddings = torch.stack([lead_embedding for lead_embedding in self.lead_embeddings]).unsqueeze(0)
        lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, n + 2, -1)
        x = x + lead_embeddings
        x = rearrange(x, 'b c n p -> b (c n) p')
        x = self.dropout(x)
        layerout = []
        
        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x)
            # remove SEP embeddings
            x_rearrange = rearrange(x, 'b (c n) p -> b c n p', c = num_leads)
            x_wo_sep = x_rearrange[:, :, 1:-1, :]
            x_out = torch.mean(x_wo_sep, dim = (1, 2))
            layerout.append(x_out)
        return layerout

    def forward(self, series):
        x = self.forward_encoding(series)
        x = torch.stack(x, dim = 2)
        x = x.contiguous().view(x.size(0), -1)
        x = self.avg(x.unsqueeze(1))
        return self.head(self.norm(x.squeeze(1)))

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'{k}={v},\n'
        print_str += ')'
        return print_str