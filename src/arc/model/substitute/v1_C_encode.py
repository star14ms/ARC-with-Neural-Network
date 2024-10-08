import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.model.components.pixel_vector_extractor import PixelVectorExtractor
from arc.model.components.cross_attn import MultiheadCrossAttentionLayer

from arc.utils.visualize import visualize_image_using_emoji


class ColorLocationEncoder(nn.Module): 
    def __init__(self, C_dims_encoded, L_dims_encoded, bias=False):
        super().__init__()
        L_dim = L_dims_encoded[0]

        self.attn_V_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(L_dim, L_dim, 1, dropout=0.1, batch_first=True, bias=bias),
            num_layers=1,
            enable_nested_tensor=False,
        )

        self.ff_C = nn.Sequential()
        for i in range(len(C_dims_encoded)-1):
            self.ff_C.add_module(f'linear_{i}', nn.Linear(C_dims_encoded[i], C_dims_encoded[i+1], bias=bias))
            if i != len(C_dims_encoded)-2:
                self.ff_C.add_module(f'relu_{i}', nn.ReLU())

    def forward(self, x):
        NS, C, LH, LW = x.shape

        # In     Out
        # 🔳🔳🔳  🟦🟦🟦
        # 🔳🔳🟩  🟦🟦🟨
        # 🔳🟩🟩  🟦🟨🟨

        # 🔳🔳🔳  🟦🟦🟦 
        # 🔳🔳🟧  🟦🟦🟦 
        # 🔳🟩🟩  🟦🟨🟨 

        # 1. Encode Colors depending on Location
        x = x.view(NS, C, LH*LW)
        x_VC = self.attn_V_self(x) # [C, L] < [C, L] # (🟧 -> 🟦)
        x_VC = self.ff_C(x_VC.transpose(1, 2)).transpose(1, 2) # [L, C] -> [L, VC]
        x_VC = F.softmax(x_VC, dim=1) # [VCp, L]

        return x_VC, x_VC


class Reasoner(nn.Module):
    def __init__(self, VC_dim, VL_dim, L_num_layers=1, L_n_head=None, L_dim_feedforward=1, C_num_layers=1, C_n_head=None, C_dim_feedforward=1, dropout=0.1, bias=False):
        super().__init__()

        self.attn_VL_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(VL_dim, L_n_head if L_n_head else VL_dim, L_dim_feedforward, dropout=dropout, batch_first=True, bias=bias),
            num_layers=L_num_layers,
            enable_nested_tensor=False,
        )
        self.attn_VC_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(VC_dim, C_n_head if C_n_head else VC_dim, C_dim_feedforward, dropout=dropout, batch_first=True, bias=bias),
            num_layers=C_num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, mem):
        # NS, VC, VL = mem.shape

        # In     Out
        # 🟦🟦🟦  🟦🟦🟦
        # 🟦🟦🟨  🟦🟨🟨
        # 🟦🟨🟨  🟦🟨🟨
        
        # 🟦🟦🟦  🟦🟦🟦
        # 🟦🟦🟦  🟦🟦🟦
        # 🟦🟨🟨  🟦🟨🟨

        # 2. Attention Across Location and Color
        mem = self.attn_VL_self(mem) # [VC, VL] < [VC, VL]
        mem = self.attn_VC_self(mem.transpose(1, 2)).transpose(1, 2)  # [VL, VC] < [VL, VC]

        # 3. Determine Encoded Output Class
        mem = F.softmax(mem, dim=1) # [VCp, VL]

        return mem


class ColorLocationDecoder(nn.Module):
    def __init__(self, C_dim, L_dim, L_dims_decoded, L_dim_feedforward=1, C_dim_feedforward=1, dropout=0.1, bias=False):
        super().__init__()

        self.attn_VC_L =  MultiheadCrossAttentionLayer(L_dim, L_dim, L_dim_feedforward, dropout=dropout, batch_first=True, bias=bias)
        
        self.attn_C_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(C_dim, C_dim, C_dim_feedforward, dropout=0.1, batch_first=True, bias=bias),
            num_layers=1,
            enable_nested_tensor=False,
        )

        self.attn_C_L =  MultiheadCrossAttentionLayer(L_dim, L_dim, C_dim_feedforward,  dropout=dropout, batch_first=True, bias=bias)

        self.ff_L = nn.Sequential()
        for i in range(len(L_dims_decoded)-1):
            self.ff_L.add_module(f'linear_{i}', nn.Linear(L_dims_decoded[i], L_dims_decoded[i+1], bias=bias))
            if i != len(L_dims_decoded)-2:
                self.ff_L.add_module(f'relu_{i}', nn.ReLU())

    def forward(self, x, mem, x_VC_mem):
        NS, C, HL, WL = x.shape

        # In     Out
        # 🟦🟦🟦  🔳🔳🔳
        # 🟦🟨🟨  🔳🟩🟩
        # 🟦🟨🟨  🔳🟩🟩
        
        # 🟦🟦🟦  🔳🔳🔳
        # 🟦🟦🟦  🔳🔳🟧
        # 🟦🟨🟨  🔳🟩🟩 
        # [VC, L] -> [C, L]

        # 4. Decode Color
        x = x.view(NS, C, HL*WL)
        x_VC_mem = self.attn_VC_L(x_VC_mem, mem) # [VC, L] < [VC, L]
        y = self.attn_C_L(x, x_VC_mem) # [C, L] < [VC, L] # (🟦 -> 🟧)
        y = self.attn_C_self(y.transpose(1, 2)).transpose(1, 2) # [L, C] < [L, C] # Detect Emerging Color
        y = self.ff_L(y) # [C, L] -> [C, 1]

        return y


class PixelEachSubstitutor(nn.Module):
    def __init__(self, W_max=30, H_max=30, n_range_search=-1, W_kernel_max=61, H_kernel_max=61, C_dims_encoded=[2], L_dims_encoded=[9], L_dims_decoded=[1], pad_class_initial=0, L_num_layers=6, L_n_head=None, L_dim_feedforward=1, C_num_layers=1, C_n_head=None, C_dim_feedforward=1, dropout=0.1, n_class=10, C_encode=None, L_encode=None):
        super().__init__()
        assert n_range_search != -1 and W_kernel_max >= 1 + 2*n_range_search and H_kernel_max >= 1 + 2*n_range_search
        L_dim = W_kernel_max * H_kernel_max

        self.abstractor = PixelVectorExtractor(
            W_max=W_max, 
            H_max=H_max, 
            n_range_search=n_range_search,
            W_kernel_max=W_kernel_max,
            H_kernel_max=H_kernel_max,
            dropout=dropout,
            bias=False,
            pad_class_initial=pad_class_initial,
            n_class=n_class,
        )

        self.encoder = ColorLocationEncoder(
            C_dims_encoded=C_dims_encoded,
            L_dims_encoded=L_dims_encoded,
            bias=False,
        )

        self.reasoner = Reasoner(
            VC_dim=C_dims_encoded[-1],
            VL_dim=L_dims_encoded[-1],
            L_num_layers=L_num_layers,
            L_n_head=L_n_head,
            L_dim_feedforward=L_dim_feedforward,
            C_num_layers=C_num_layers,
            C_n_head=C_n_head,
            C_dim_feedforward=C_dim_feedforward,
            dropout=dropout,
            bias=False,
        )
        
        self.decoder = ColorLocationDecoder(
            C_dim=n_class,
            L_dim=L_dim,
            L_dims_decoded=L_dims_decoded,
            L_dim_feedforward=L_dim_feedforward,
            C_dim_feedforward=C_dim_feedforward,
            dropout=dropout,
            bias=False,
        )

    def forward(self, x, return_prob=False, **kwargs):
        N, C, H, W = x.shape
        
        # x_origin = x.detach().clone()

        # Task: 22168020
        # Input  Encode Solve  Decode
        # 🔳🔳🔳  🟦🟦🟦  🟦🟦🟦  🔳🔳🔳
        # 🔳🔳🟩  🟦🟦🟨  🟦🟨🟨  🔳🟩🟩
        # 🔳🟩🟩  🟦🟨🟨  🟦🟨🟨  🔳🟩🟩
        
        # 🔳🔳🔳  🟦🟦🟦  🟦🟦🟦  🔳🔳🔳
        # 🔳🔳🟧  🟦🟦🟦  🟦🟦🟦  🔳🔳🟧
        # 🔳🟩🟩  🟦🟨🟨  🟦🟨🟨  🔳🟩🟩 

        x = self.abstractor(x)
        mem, x_VC_mem = self.encoder(x)
        mem = self.reasoner(mem)
        y = self.decoder(x, mem, x_VC_mem)

        # img = y.argmax(dim=1).view(N, H, W, -1).permute(0, 3, 1, 2)
        # visualize_image_using_emoji(x_origin[0], img[0])
        # breakpoint()
        y = y.view(N, H, W, C).permute(0, 3, 1, 2) # [N, C, H, W]

        if return_prob:
            y = F.softmax(y, dim=1) # [NS, C_prob]

        return y

