import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.model.components.pixel_vector_extractor import PixelVectorExtractor
from arc.model.components.cross_attn import MultiheadCrossAttentionLayer
from arc.utils.visualize import visualize_image_using_emoji


class ColorEncoder(nn.Module): 
    def __init__(self, C_dims_encoded, L_dim, L_dim_feedforward, memory_channel=False, n_class=10, dropout=0.1, bias=False):
        super().__init__()

        d_model = n_class+1 if memory_channel else n_class
        self.memory_channel = memory_channel
        self.attn_C_x = MultiheadCrossAttentionLayer(d_model, d_model, L_dim_feedforward, dropout=dropout, bias=bias, batch_first=True)
        self.attn_C_xs = MultiheadCrossAttentionLayer(d_model, d_model, L_dim_feedforward, dropout=dropout, bias=bias, batch_first=True)

        self.attn_L_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(L_dim, L_dim, L_dim_feedforward, dropout=dropout, batch_first=True, bias=bias),
            num_layers=1,
            enable_nested_tensor=False,
        )

        self.ff_C = nn.Sequential()
        for i in range(len(C_dims_encoded)-1):
            self.ff_C.add_module(f'linear_{i}', nn.Linear(C_dims_encoded[i], C_dims_encoded[i+1], bias=bias))
            if i != len(C_dims_encoded)-2:
                self.ff_C.add_module(f'relu_{i}', nn.ReLU())

    def forward(self, x, xs=None):
        NS, C, L = x.shape

        if xs is not None:
            N, C, H, W = xs.shape
            xs_L_sum = xs.view(N, C, H*W).sum(dim=0).repeat(NS, 1, 1) # [VC, L]

        # In     Out
        # 🔳🔳🔳  🟦🟦🟦
        # 🔳🔳🟩  🟦🟦🟨
        # 🔳🟩🟩  🟦🟨🟨

        # 🔳🔳🔳  🟦🟦🟦 
        # 🔳🔳🟧  🟦🟦🟦 
        # 🔳🟩🟩  🟦🟨🟨 
        
        if self.memory_channel:
            x, memory_channel = x[:, :-1], x[:, -1:] # [C+1, L] -> [C, L] [1, L]
        x_L_sum = x.sum(dim=2).unsqueeze(2) # [C, 1]

        # 1. Encode Colors depending on Location
        if xs is not None:
            x = self.attn_C_xs(x.transpose(2, 1), xs_L_sum.transpose(2, 1)).transpose(2, 1) # [L, C] < [L, C]
        x = self.attn_C_x(x.transpose(2, 1), x_L_sum.transpose(2, 1)).transpose(2, 1) # [L, C] < [1, C]

        if self.memory_channel:
            x = torch.cat([x, memory_channel], dim=1) # [C, L] -> [C+1, L]

        x_C = self.attn_L_self(x) # [C, L] < [C, L] # (🟧 -> 🟦)
        x_VC = self.ff_C(x_C.transpose(1, 2)).transpose(1, 2) # [L, C] -> [L, VC]
        x = x_VC.transpose(1, 0).softmax(dim=0).transpose(1, 0) # [VCp, L]

        return x, x_C, x_VC


class LocationEncoder(nn.Module): 
    def __init__(self, VC_dim, L_dims_encoded, C_dim_feedforward, dropout=0.1, bias=False):
        super().__init__()

        self.attn_C_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(VC_dim, VC_dim, C_dim_feedforward, dropout=dropout, batch_first=True, bias=bias),
            num_layers=1,
            enable_nested_tensor=False,
        )

        self.ff_L = nn.Sequential()
        for i in range(len(L_dims_encoded)-1):
            self.ff_L.add_module(f'linear_{i}', nn.Linear(L_dims_encoded[i], L_dims_encoded[i+1], bias=bias))
            if i != len(L_dims_encoded)-2:
                self.ff_L.add_module(f'relu_{i}', nn.ReLU())

    def forward(self, x):
        NS, VC, L = x.shape

        # In     Out
        # 🔳🔳🔳  🟦🟦🟦
        # 🔳🔳🟩  🟦🟦🟨
        # 🔳🟩🟩  🟦🟨🟨

        # 🔳🔳🔳  🟦🟦🟦 
        # 🔳🔳🟧  🟦🟦🟦 
        # 🔳🟩🟩  🟦🟨🟨 
        
        # xs_L_sum = x.sum(dim=0).repeat(NS, 1, 1) # [VC, L]

        # 2. Encode Locations
        # x = self.attn_V_xs(x, xs_L_sum) # [VC, L] < [C, L]
        x_VC_L = self.attn_C_self(x.transpose(1, 2)).transpose(1, 2) # [L, VC] < [L, VC]
        x = x_VC_L.reshape(NS*VC, L)
        x_VC_VL = self.ff_L(x) # [N*S*C, L]
        x_VC_VL = x_VC_VL.reshape(NS, VC, -1)

        return x_VC_L, x_VC_VL


class Encoder(nn.Module): 
    def __init__(self, C_dims_encoded, L_dims_encoded, L_dim_feedforward, C_dim_feedforward, memory_channel, n_class=10, dropout=0.1, bias=False):
        super().__init__()
        L_dim = L_dims_encoded[0]
        VC_dim = C_dims_encoded[-1]

        self.encoder_color = ColorEncoder(C_dims_encoded, L_dim, L_dim_feedforward, memory_channel=memory_channel, n_class=n_class, dropout=dropout, bias=bias)
        self.encoder_location = LocationEncoder(VC_dim, L_dims_encoded, C_dim_feedforward, dropout=dropout, bias=bias)

    def forward(self, x, xs=None):
        NS, C, L = x.shape

        # In     Out
        # 🔳🔳🔳  🟦🟦🟦
        # 🔳🔳🟩  🟦🟦🟨
        # 🔳🟩🟩  🟦🟨🟨

        # 🔳🔳🔳  🟦🟦🟦 
        # 🔳🔳🟧  🟦🟦🟦 
        # 🔳🟩🟩  🟦🟨🟨 
        # [C, L] -> [VC, L]

        x, x_C, x_VC = self.encoder_color(x, xs)
        x_VC_L, x_VC_VL = self.encoder_location(x)

        return x_VC_VL, x_VC_L, x_VC, x_C


class Reasoner(nn.Module):
    def __init__(self, VC_dim, VL_dim, L_num_layers=1, L_n_head=None, L_dim_feedforward=1, C_num_layers=1, C_n_head=None, C_dim_feedforward=1, x_inference=True, dropout=0.1, bias=False):
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

        if x_inference:
            self.attn_C_inference = MultiheadCrossAttentionLayer(VC_dim, VC_dim, L_dim_feedforward, dropout=dropout, batch_first=True, bias=bias)

    def forward(self, mem, x_inference=None):
        NS, VC, VL = mem.shape

        # In     Out
        # 🟦🟦🟦  🟦🟦🟦
        # 🟦🟦🟨  🟦🟨🟨
        # 🟦🟨🟨  🟦🟨🟨
        
        # 🟦🟦🟦  🟦🟦🟦
        # 🟦🟦🟦  🟦🟦🟦
        # 🟦🟨🟨  🟦🟨🟨

        # 3. Attention Across Location and Color
        if x_inference is not None:
            mem = self.attn_C_inference(mem.transpose(1, 2), x_inference.repeat(NS, 1, 1).transpose(1, 2)).transpose(1, 2) # [VC, VL] < [VC, VL]

        mem = self.attn_VL_self(mem) # [VC, VL] < [VC, VL]
        mem = self.attn_VC_self(mem.transpose(1, 2)).transpose(1, 2) # [VL, VC] < [VL, VC]

        # 4. Determine Encoded Output Class
        mem = mem.transpose(1, 0).softmax(dim=0).transpose(1, 0) # [VCp, VL]

        return mem


class LocationDecoder(nn.Module):
    def __init__(self, VC_dim, L_dim_feedforward=1, dropout=0.1, bias=False):
        super().__init__()

        self.attn_VL_VL = MultiheadCrossAttentionLayer(VC_dim, VC_dim, L_dim_feedforward, dropout=dropout, batch_first=True, bias=bias)
        self.attn_L_VL = MultiheadCrossAttentionLayer(VC_dim, VC_dim, L_dim_feedforward, dropout=dropout, batch_first=True, bias=bias)
        
    def forward(self, x_VC_L, x_VC_VL, mem):

        # 5. Decode Location
        x_VC_VL = self.attn_L_VL(x_VC_VL.transpose(1, 2), mem.transpose(1, 2)) # [VL, VC] < [VL, VC]
        x_VC_L = self.attn_VL_VL(x_VC_L.transpose(1, 2), x_VC_VL).transpose(1, 2) # [L, VC]
        x_VC_L = x_VC_L.transpose(1, 0).softmax(dim=0).transpose(1, 0) # [VC, L]

        return x_VC_L


class ColorDecoder(nn.Module):
    def __init__(self, L_dim, C_dim, L_dims_decoded, emerge_color=True, L_dim_feedforward=1, C_dim_feedforward=1, dropout=0.1, bias=False):
        super().__init__()

        self.attn_VC_L = MultiheadCrossAttentionLayer(L_dim, L_dim, L_dim_feedforward, dropout=dropout, batch_first=True, bias=bias)
        self.attn_C_L = MultiheadCrossAttentionLayer(L_dim, L_dim, L_dim_feedforward, dropout=dropout, batch_first=True, bias=bias)
        # self.attn_C_self = nn.MultiheadAttention(C_dim, C_dim, dropout=dropout, bias=bias)
        # self.attn_L_C = MultiheadCrossAttentionLayer(C_dim, C_dim, L_dim_feedforward, dropout=dropout, batch_first=True, bias=bias)

        self.emerge_color = emerge_color

        if emerge_color:
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

    def forward(self, x, x_C, x_VC, x_VC_mem):
        NS, C, L = x.shape

        # 6. Decode Color
        x = x.view(NS, C, L)
        x_VC = self.attn_VC_L(x_VC, x_VC_mem) # [VC, L] < [VC, L]
        # y = self.attn_C_self(x.transpose(1, 2))
        y = self.attn_C_L(x, x_VC) # [C, L] < [VC, L] # (🟦 -> 🟧)
        # y = self.attn_L_C(y.transpose(1, 2), x_C.transpose(1, 2)).transpose(1, 2) # [L, C] < [L, C] # (🟧 -> 🟦)

        if self.emerge_color:
            y = self.attn_C_self(y.transpose(1, 2)).transpose(1, 2) # [L, C] -> [L, C] # Detect Emerging Color

        y = self.ff_L(y) # [C, L] -> [C, 1]

        return y


class Decoder(nn.Module):
    def __init__(self, VL_dim, VC_dim, L_dim, C_dim, L_dims_decoded, emerge_color=True, L_dim_feedforward=1, C_dim_feedforward=1, dropout=0.1, bias=False):
        super().__init__()

        self.decoder_location = LocationDecoder(VC_dim, L_dim_feedforward=L_dim_feedforward, dropout=dropout, bias=bias)
        self.decoder_color = ColorDecoder(L_dim, C_dim, L_dims_decoded, emerge_color=emerge_color, L_dim_feedforward=L_dim_feedforward, C_dim_feedforward=C_dim_feedforward, dropout=dropout, bias=bias)

    def forward(self, x, mem, x_VC_VL, x_VC_L, x_VC, x_C):
        NS, C, L = x.shape

        # In     Out
        # 🟦🟦🟦  🔳🔳🔳
        # 🟦🟨🟨  🔳🟩🟩
        # 🟦🟨🟨  🔳🟩🟩
        
        # 🟦🟦🟦  🔳🔳🔳
        # 🟦🟦🟦  🔳🔳🟧
        # 🟦🟨🟨  🔳🟩🟩 
        # [VC, L] -> [C, L]

        x_VC_mem = self.decoder_location(x_VC_L, x_VC_VL, mem)
        y = self.decoder_color(x, x_C, x_VC, x_VC_mem)

        return y


class PixelEachSubstitutor(nn.Module):
    def __init__(self, n_range_search=-1, vec_abs=True, emerge_color=True, memory_channel=False, W_max=30, H_max=30, W_kernel_max=61, H_kernel_max=61, C_dims_encoded=[2], L_dims_encoded=[9], L_dims_decoded=[1], pad_class_initial=0, L_num_layers=1, L_n_head=None, L_dim_feedforward=1, C_num_layers=1, C_n_head=None, C_dim_feedforward=1, dropout=0.0, n_class=10, C_encode=None, L_encode=None, pad_num_layers=None, pad_n_head=None, pad_dim_feedforward=None):
        super().__init__()
        assert n_range_search != -1 and W_kernel_max >= 1 + 2*n_range_search and H_kernel_max >= 1 + 2*n_range_search
        self.memory_channel = memory_channel
        
        self.inferer = StateAnalyst(
            in_channels=n_class, 
            out_channels=10, 
            kernel_size=3, 
            dim_hidden=32, 
            dim_output=10, 
            x_size_max=15*15,
            VC_dim=C_dims_encoded[-1],
        )

        self.abstractor = PixelVectorExtractor(
            n_range_search=n_range_search,
            W_kernel_max=W_kernel_max,
            H_kernel_max=H_kernel_max,
            vec_abs=vec_abs,
            W_max=W_max,
            H_max=H_max,
            pad_class_initial=pad_class_initial,
            memory_channel=memory_channel,
        )

        self.encoder = Encoder(
            C_dims_encoded=C_dims_encoded,
            L_dims_encoded=L_dims_encoded,
            L_dim_feedforward=L_dim_feedforward,
            C_dim_feedforward=C_dim_feedforward,
            memory_channel=memory_channel,
            n_class=n_class,
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

        self.decoder = Decoder(
            VL_dim=L_dims_encoded[-1],
            VC_dim=C_dims_encoded[-1],
            L_dim=L_dims_encoded[0],
            C_dim=C_dims_encoded[0],
            L_dims_decoded=L_dims_decoded,
            emerge_color=emerge_color,
            L_dim_feedforward=L_dim_feedforward,
            C_dim_feedforward=C_dim_feedforward,
            dropout=dropout,
            bias=False,
        )

    def forward(self, x, xs=None, memory_channel=None, t=None, return_prob=False, return_observation=False, **kwargs):
        N, C, H, W = x.shape

        # Task: 22168020
        # Input  Encode Solve  Decode
        # 🔳🔳🔳  🟦🟦🟦  🟦🟦🟦  🔳🔳🔳
        # 🔳🔳🟩  🟦🟦🟨  🟦🟨🟨  🔳🟩🟩
        # 🔳🟩🟩  🟦🟨🟨  🟦🟨🟨  🔳🟩🟩
        
        # 🔳🔳🔳  🟦🟦🟦  🟦🟦🟦  🔳🔳🔳
        # 🔳🔳🟧  🟦🟦🟦  🟦🟦🟦  🔳🔳🟧
        # 🔳🟩🟩  🟦🟨🟨  🟦🟨🟨  🔳🟩🟩 
        
        x_inference = self.inferer(x) # [N, C, V]

        x = self.abstractor(x, memory_channel) # [N*H*W, C+1, H_max*W_max]
        C = x.shape[1]

        x_VC_VL, x_VC_L, x_VC, x_C = self.encoder(x, xs)
        mem = self.reasoner(x_VC_VL, x_inference)
        y = self.decoder(x, mem, x_VC_VL, x_VC_L, x_VC, x_C)
        
        if self.memory_channel:
            y = y[:, :-1] # Remove the padding class
        y = y.view(N, H, W, -1).permute(0, 3, 1, 2) # [N, C, H, W]

        if return_prob:
            y = y.transpose(1, 0).softmax(dim=0).transpose(1, 0) # [NS, C_prob]

        if return_observation:
            return x, y
        else:
            return y



class StateAnalyst(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dim_hidden=32, dim_output=8, x_size_max=15*15, VC_dim=3):
        super().__init__()
        self.x_size_max = x_size_max
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        
        self.ff = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(x_size_max, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_output),
        )
        
        self.encoder_C = nn.Sequential(
            nn.Linear(in_channels, VC_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        N, C, H, W = x.shape
        y = torch.zeros([N, 10, self.x_size_max])
        x = self.conv(x)

        x = x.flatten(2)
        y[:, :, :H*W] = x

        y = self.ff(y.view(N*C, -1)).view(N, C, -1)
        y = self.encoder_C(y.transpose(1, 2)).transpose(1, 2)

        return y