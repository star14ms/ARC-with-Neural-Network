import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.model.components.pixel_vector_extractor import PixelVectorExtractor
from arc.model.components.cross_attn import MultiheadCrossAttentionLayer


class ColorLocationEncoder(nn.Module): 
    def __init__(self, C_dims_encoded, L_dims_encoded, dropout=0.1, bias=False):
        super().__init__()
        L_dim = L_dims_encoded[0]

        self.attn_V_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(L_dim, L_dim, 1, dropout=dropout, batch_first=True, bias=bias),
            num_layers=1,
            enable_nested_tensor=False,
        )

        self.ff_C = nn.Sequential()
        for i in range(len(C_dims_encoded)-1):
            self.ff_C.add_module(f'linear_{i}', nn.Linear(C_dims_encoded[i], C_dims_encoded[i+1], bias=bias))
            if i != len(C_dims_encoded)-2:
                self.ff_C.add_module(f'relu_{i}', nn.ReLU())

        self.ff_L = nn.Sequential()
        for i in range(len(L_dims_encoded)-1):
            self.ff_L.add_module(f'linear_{i}', nn.Linear(L_dims_encoded[i], L_dims_encoded[i+1], bias=bias))
            if i != len(L_dims_encoded)-2:
                self.ff_L.add_module(f'relu_{i}', nn.ReLU())

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
        x = F.softmax(x_VC, dim=1) # [VCp, L]
        VC = x.shape[1]

        # 2. Encode Locations
        x = x.view(NS*VC, -1)
        x_VC_VL = self.ff_L(x) # [N*S*C, L]
        x_VC_VL = x_VC_VL.reshape(NS, VC, -1)

        return x_VC_VL, x_VC


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
        NS, VC, VL = mem.shape

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
    def __init__(self, VL_dim, VC_dim, L_dim, C_dim, L_dims_decoded, L_dim_feedforward=1, C_dim_feedforward=1, dropout=0.1, bias=False):
        super().__init__()

        self.attn_VL_VL =  MultiheadCrossAttentionLayer(VC_dim, VC_dim, L_dim_feedforward, dropout=dropout, batch_first=True, bias=bias)
        self.attn_L_VL =  MultiheadCrossAttentionLayer(VC_dim, VC_dim, L_dim_feedforward, dropout=dropout, batch_first=True, bias=bias)
        self.attn_VC_L =  MultiheadCrossAttentionLayer(L_dim, L_dim, L_dim_feedforward, dropout=dropout, batch_first=True, bias=bias)
        self.attn_C_L =  MultiheadCrossAttentionLayer(L_dim, L_dim, C_dim_feedforward,  dropout=dropout, batch_first=True, bias=bias)
        
        self.attn_C_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(C_dim, C_dim, C_dim_feedforward, dropout=dropout, batch_first=True, bias=bias),
            num_layers=1,
            enable_nested_tensor=False,
        )

        self.ff_L = nn.Sequential()
        for i in range(len(L_dims_decoded)-1):
            self.ff_L.add_module(f'linear_{i}', nn.Linear(L_dims_decoded[i], L_dims_decoded[i+1], bias=bias))
            if i != len(L_dims_decoded)-2:
                self.ff_L.add_module(f'relu_{i}', nn.ReLU())

    def forward(self, x, mem, x_VC_VL, x_VC):
        NS, C, HL, WL = x.shape

        # In     Out
        # 🟦🟦🟦  🔳🔳🔳
        # 🟦🟨🟨  🔳🟩🟩
        # 🟦🟨🟨  🔳🟩🟩
        
        # 🟦🟦🟦  🔳🔳🔳
        # 🟦🟦🟦  🔳🔳🟧
        # 🟦🟨🟨  🔳🟩🟩 
        # [VC, L] -> [C, L]

        # 4. Decode Location
        x_VC_VL = self.attn_L_VL(x_VC_VL.transpose(1, 2), mem.transpose(1, 2)) # [VL, VC] < [VL, VC]
        x_VC_mem = self.attn_VL_VL(x_VC.transpose(1, 2), x_VC_VL).transpose(1, 2) # [L, VC]
        x_VC_mem = F.softmax(x_VC_mem, dim=1) # [VC, L]

        # 5. Decode Color
        x = x.view(NS, C, HL*WL)
        x_VC_mem = self.attn_VC_L(x_VC, x_VC_mem) # [VC, L] < [VC, L]
        y = self.attn_C_L(x, x_VC_mem) # [C, L] < [VC, L] # (🟦 -> 🟧)
        y = self.attn_C_self(y.transpose(1, 2)).transpose(1, 2) # [L, C] -> [L, C] # Detect Emerging Color
        y = self.ff_L(y) # [C, L] -> [C, 1]

        return y


class PixelEachSubstitutor(nn.Module):
    def __init__(self, n_range_search=-1, max_width=61, max_height=61, C_dims_encoded=[2], L_dims_encoded=[9], L_dims_decoded=[1], pad_class_initial=0, pad_num_layers=1, pad_n_head=None, pad_dim_feedforward=1, L_num_layers=6, L_n_head=None, L_dim_feedforward=1, C_num_layers=1, C_n_head=None, C_dim_feedforward=1, dropout=0.1, n_class=10, C_encode=None, L_encode=None):
        super().__init__()
        assert n_range_search != -1 and max_width >= 1 + 2*n_range_search and max_height >= 1 + 2*n_range_search
        L_dim = max_width * max_height
        C_dims_encoded = [n_class] + C_dims_encoded
        L_dims_encoded = [L_dim] + L_dims_encoded
        L_dims_decoded =  [L_dim] + L_dims_decoded

        self.abstractor = PixelVectorExtractor(
            n_range_search=n_range_search,
            max_width=max_width,
            max_height=max_height,
            pad_n_head=pad_n_head,
            pad_dim_feedforward=pad_dim_feedforward, 
            dropout=dropout,
            pad_num_layers=pad_num_layers,
            bias=False,
            pad_class_initial=pad_class_initial,
            n_class=n_class,
        )

        self.encoder = ColorLocationEncoder(
            C_dims_encoded=C_dims_encoded,
            L_dims_encoded=L_dims_encoded,
            dropout=dropout,
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
            VL_dim=L_dims_encoded[-1],
            VC_dim=C_dims_encoded[-1],
            L_dim=L_dim,
            C_dim=n_class,
            L_dims_decoded=L_dims_decoded,
            L_dim_feedforward=L_dim_feedforward,
            C_dim_feedforward=C_dim_feedforward,
            dropout=dropout,
            bias=False,
        )

    def forward(self, x, return_prob=False, **kwargs):
        N, C, H, W = x.shape

        # Task: 22168020
        # Input  Encode Solve  Decode
        # 🔳🔳🔳  🟦🟦🟦  🟦🟦🟦  🔳🔳🔳
        # 🔳🔳🟩  🟦🟦🟨  🟦🟨🟨  🔳🟩🟩
        # 🔳🟩🟩  🟦🟨🟨  🟦🟨🟨  🔳🟩🟩
        
        # 🔳🔳🔳  🟦🟦🟦  🟦🟦🟦  🔳🔳🔳
        # 🔳🔳🟧  🟦🟦🟦  🟦🟦🟦  🔳🔳🟧
        # 🔳🟩🟩  🟦🟨🟨  🟦🟨🟨  🔳🟩🟩 

        x = self.abstractor(x)
        x_VC_VL, x_VC = self.encoder(x)
        mem = self.reasoner(x_VC_VL)
        y = self.decoder(x, mem, x_VC_VL, x_VC)

        y = y.view(N, H, W, C).permute(0, 3, 1, 2) # [N, C, H, W]

        if return_prob:
            y = F.softmax(y, dim=1) # [NS, C_prob]

        return y

