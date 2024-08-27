import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.model.components.pixel_vector_extractor import PixelVectorExtractor


class LocationEncoder(nn.Module): 
    def __init__(self, L_dims_encoded, bias=False):
        super().__init__()
        # self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=bias)
        # L_dims_encoded[0] *= 8

        self.ff_L = nn.Sequential()
        for i in range(len(L_dims_encoded)-1):
            self.ff_L.add_module(f'linear_{i}', nn.Linear(L_dims_encoded[i], L_dims_encoded[i+1], bias=bias))
            if i != len(L_dims_encoded)-2:
                self.ff_L.add_module(f'relu_{i}', nn.ReLU())

    def forward(self, x):
        NS, C, L = x.shape

        # Encode Locations
        # x = x.view(NS*C, 1, LH, LW) # [N, C, H, W]
        # x = self.conv(x) # [N, C, H, W]
        x = x.reshape(NS*C, L)
        x = self.ff_L(x).reshape(NS, C, -1) # [N*S*C, L]: 2. Location Encoding

        return x


class ReasonerNonColorEncoding(nn.Module):
    def __init__(self, VL_dim, L_dims_decoded, L_num_layers=6, L_n_head=None, L_dim_feedforward=1, C_num_layers=1, C_n_head=None, C_dim_feedforward=1, dropout=0.1, bias=False, n_class=10):
        super().__init__()
        L_dims_decoded =  [VL_dim] + L_dims_decoded

        self.attn_VL_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(VL_dim, L_n_head if L_n_head else VL_dim, L_dim_feedforward, dropout=dropout, batch_first=True, bias=bias),
            num_layers=L_num_layers,
            enable_nested_tensor=False,
        )
        self.attn_VC_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(n_class, C_n_head if C_n_head else n_class, C_dim_feedforward, dropout=dropout, batch_first=True, bias=bias),
            num_layers=C_num_layers,
            enable_nested_tensor=False,
        )

        self.decoder_VL = nn.Sequential()
        for i in range(len(L_dims_decoded)-1):
            self.decoder_VL.add_module(f'linear_{i}', nn.Linear(L_dims_decoded[i], L_dims_decoded[i+1], bias=bias))
            if i != len(L_dims_decoded)-2:
                self.decoder_VL.add_module(f'relu_{i}', nn.ReLU())

    def forward(self, x):
        NS, C, VL = x.shape

        # 1. Attention Across Vectors and Classify
        x = self.attn_VL_self(x) # [C, VL] <- [C, VL]
        x = self.attn_VC_self(x.transpose(1, 2)).transpose(1, 2)  # [VL, C] <- [VL, C]

        # 2. Determine Output Class (Decode Location)
        x = x.reshape(NS*C, VL)
        x = self.decoder_VL(x) # [NS*C, 1]

        return x


class PixelEachSubstitutorNonColorEncoding(nn.Module):
    def __init__(self, W_max=30, H_max=30, n_range_search=-1, W_kernel_max=61, H_kernel_max=61, L_dims_encoded=[512, 128, 64], L_dims_decoded=[32, 8, 1], pad_class_initial=0, pad_num_layers=1, pad_n_head=None, pad_dim_feedforward=1, L_encode=False, L_num_layers=6, L_n_head=None, L_dim_feedforward=1, C_num_layers=1, C_n_head=None, C_dim_feedforward=1, dropout=0.1, n_class=10, C_encode=None, C_dims_encoded=[]):
        super().__init__()
        assert n_range_search != -1 and W_kernel_max >= 1 + 2*n_range_search and H_kernel_max >= 1 + 2*n_range_search
        self.L_encode = L_encode

        self.abstractor = PixelVectorExtractor(
            W_max=W_max, 
            H_max=H_max, 
            n_range_search=n_range_search,
            W_kernel_max=W_kernel_max,
            H_kernel_max=H_kernel_max,
            pad_n_head=pad_n_head,
            pad_dim_feedforward=pad_dim_feedforward, 
            dropout=dropout,
            pad_num_layers=pad_num_layers,
            bias=False,
            pad_class_initial=pad_class_initial,
            n_class=n_class,
        )
        
        if self.L_encode:
            self.encoder = LocationEncoder(
                L_dims_encoded=L_dims_encoded,
                bias=False,
            )

        self.reasoner = ReasonerNonColorEncoding(
            VL_dim=L_dims_encoded[-1],
            L_dims_decoded=L_dims_decoded, 
            L_num_layers=L_num_layers,
            L_n_head=L_n_head,
            L_dim_feedforward=L_dim_feedforward,
            C_num_layers=C_num_layers,
            C_n_head=C_n_head,
            C_dim_feedforward=C_dim_feedforward,
            dropout=dropout,
            bias=False,
            n_class=n_class,
        )

    def forward(self, x, return_prob=False, **kwargs):
        N, C, H, W = x.shape

        x = self.abstractor(x)
        mem = self.encoder(x) if self.L_encode else x.reshape(N, C, -1)
        x = self.reasoner(mem)
        x = x.view(N, H, W, C).permute(0, 3, 1, 2) # [N, C, H, W]

        if return_prob:
            x = F.softmax(x, dim=1) # [NS, C_prob]

        return x
