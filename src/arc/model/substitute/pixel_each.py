import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.model.substitute.cross_attn import MultiheadCrossAttentionLayer


class PixelVectorExtractor(nn.Module):
    def __init__(self, n_range_search, max_width, max_height, pad_class_initial=0, pad_n_head=None, pad_dim_feedforward=1, dropout=0.1, pad_num_layers=1, bias=False, n_class=10):
        super().__init__()
        self.n_range_search = n_range_search
        self.max_width = max_width
        self.max_height = max_height
        self.pad_class_initial = pad_class_initial

        self.attn_C = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(n_class+1, pad_n_head if pad_n_head is not None else n_class+1, pad_dim_feedforward, dropout=dropout, batch_first=True, bias=bias),
            num_layers=pad_num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, x):
        N, C, H, W = x.shape
        H_PAD = self.n_range_search if self.n_range_search != -1 else H - 1
        W_PAD = self.n_range_search if self.n_range_search != -1 else W - 1
        HL = 1 + 2*H_PAD
        WL = 1 + 2*W_PAD
        
        if self.n_range_search != 0:
            x_C10 = torch.zeros_like(x[:,:1], dtype=x.dtype, device=x.device)
            x_C10 = F.pad(x_C10, (W_PAD, W_PAD, H_PAD, H_PAD), mode='constant', value=1)
            
            if self.pad_class_initial == 0:
                # add one more color dimension meaning padding or not
                x_C0 = F.pad(x[:, self.pad_class_initial:self.pad_class_initial+1], (W_PAD, W_PAD, H_PAD, H_PAD), mode='constant', value=1)
                x_C1to9 = F.pad(x[:, self.pad_class_initial+1:], (W_PAD, W_PAD, H_PAD, H_PAD), mode='constant', value=0)
                x = torch.cat([x_C0, x_C1to9, x_C10], dim=1)
            elif self.pad_class_initial == -1:
                x = F.pad(x, (W_PAD, W_PAD, H_PAD, H_PAD), mode='constant', value=0)
                x = torch.cat([x, x_C10], dim=1)

            ### TODO: if self.pad_class_initial != 0:

        x = x.permute(0, 2, 3, 1)
        C = x.shape[3]
        x = torch.stack([x[n].unfold(0, HL, 1).unfold(1, WL, 1) for n in range(N)], dim=0) # [N, H, W, C, HL, WL]
        x = x.view(N*H*W*C, HL, WL) # [N, S, C, L] where S = H*W

        # Padding to Max Length
        x_max = torch.full([N*H*W*C, self.max_height, self.max_width], fill_value=0, dtype=x.dtype, device=x.device)
        x_max[:,:HL,:WL] = x
        x = x_max.view(N*H*W, C, -1)

        # Predict Padding Colors
        if self.n_range_search != 0:
            x = x.transpose(1, 2)
            mask_to_inf = (x == 0).all(dim=2)
            mask_to_0 = (x[:, :, -1:] == 1)
            colored_padding = self.attn_C(x, src_key_padding_mask=mask_to_inf) # [L, C+1] <- [L, C+1]
            colored_padding = colored_padding[:, :, :-1].softmax(dim=2)
            colored_padding = colored_padding * mask_to_0
            x = x[:, :, :-1] + colored_padding
            x = x.transpose(1, 2)
            C = x.shape[1]

        # if self.n_range_search != 0:
        #     x = x[:, :-1] # [N*H*W, C, L]
        #     C = x.shape[1]

        ### TODO: PixelVector = Pixels Located Relatively + Pixels Located Absolutely (363442ee, 63613498, aabf363d) + 3 Pixels with point-symmetric and line-symmetric relationships (3631a71a, 68b16354)
        ### TODO: PixelEachSubstitutorOverTime (045e512c, 22168020, 22eb0ac0, 3bd67248, 508bd3b6, 623ea044), 

        return x.view(N*H*W, C, HL, WL)


class ColorLocationEncoder(nn.Module): 
    def __init__(self, C_dims_encoded, L_dims_encoded, bias=False):
        super().__init__()
        L_dim = L_dims_encoded[0]
        
        self.attn_V_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(L_dim, L_dim, 1, dropout=0.1, batch_first=True, bias=bias),
            num_layers=1,
            enable_nested_tensor=False,
        )

        self.ff_L = nn.Sequential()
        for i in range(len(L_dims_encoded)-1):
            self.ff_L.add_module(f'linear_{i}', nn.Linear(L_dims_encoded[i], L_dims_encoded[i+1], bias=bias))
            if i != len(L_dims_encoded)-2:
                self.ff_L.add_module(f'relu_{i}', nn.ReLU())

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
        x_mem = self.attn_V_self(x) # [C, L] < [C, L] # (🟦 -> 🟧)
        x_mem = self.ff_C(x_mem.transpose(1, 2)).transpose(1, 2) # [L, C] -> [L, VC]
        mem = F.softmax(x_mem, dim=1) # [VCp, L]

        VC = mem.shape[1]
        mem = mem.reshape(NS, VC, LH*LW)

        return mem, x_mem


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

        # 1. Attention Across Location and Color
        mem = self.attn_VL_self(mem) # [VC, VL] < [VC, VL]
        mem = self.attn_VC_self(mem.transpose(1, 2)).transpose(1, 2)  # [VL, VC] < [VL, VC]

        # 2. Determine Encoded Output Class
        mem = F.softmax(mem, dim=1) # [VCp, VL]

        return mem


class ColorLocationDecoder(nn.Module):
    def __init__(self, L_dim, L_dims_decoded, L_dim_feedforward=1, C_dim_feedforward=1, dropout=0.1, bias=False):
        super().__init__()

        self.attn_VC_L =  MultiheadCrossAttentionLayer(L_dim, L_dim, L_dim_feedforward, dropout=dropout, batch_first=True, bias=bias)
        self.attn_C_L =  MultiheadCrossAttentionLayer(L_dim, L_dim, C_dim_feedforward,  dropout=dropout, batch_first=True, bias=bias)

        self.decoder_L = nn.Sequential()
        for i in range(len(L_dims_decoded)-1):
            self.decoder_L.add_module(f'linear_{i}', nn.Linear(L_dims_decoded[i], L_dims_decoded[i+1], bias=bias))
            if i != len(L_dims_decoded)-2:
                self.decoder_L.add_module(f'relu_{i}', nn.ReLU())

    def forward(self, x, mem, x_mem):
        NS, C, HL, WL = x.shape

        # In     Out
        # 🟦🟦🟦  🔳🔳🔳
        # 🟦🟨🟨  🔳🟩🟩
        # 🟦🟨🟨  🔳🟩🟩
        
        # 🟦🟦🟦  🔳🔳🔳
        # 🟦🟦🟦  🔳🔳🟧
        # 🟦🟨🟨  🔳🟩🟩 
        # [VC, L] -> [C, L]

        # 3. Decode Color
        x = x.view(NS, C, HL*WL)
        x_mem = self.attn_VC_L(x_mem, mem) # [C, L] < [VC, L]
        y = self.attn_C_L(x, x_mem) # [C, L] < [C, L] # (🟦 -> 🟧)
        y = self.decoder_L(y) # [C, L] -> [C, 1]

        return y

class PixelEachSubstitutor(nn.Module):
    def __init__(self, n_range_search=-1, max_width=61, max_height=61, C_dims_encoded=[2], L_dims_encoded=[9], L_dims_decoded=[1], pad_class_initial=0, pad_num_layers=1, pad_n_head=None, pad_dim_feedforward=1, L_num_layers=6, L_n_head=None, L_dim_feedforward=1, C_num_layers=1, C_n_head=None, C_dim_feedforward=1, dropout=0.1, n_class=10):
        super().__init__()
        assert n_range_search != -1 and max_width >= 1 + 2*n_range_search and max_height >= 1 + 2*n_range_search
        L_dim = max_width * max_height
        C_dims_encoded = [n_class] + C_dims_encoded
        L_dims_encoded = [L_dim] + L_dims_encoded
        L_dims_decoded =  [L_dims_encoded[-1]] + L_dims_decoded

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
            L_dim=L_dim,
            L_dims_decoded=L_dims_decoded,
            L_dim_feedforward=L_dim_feedforward,
            C_dim_feedforward=C_dim_feedforward,
            dropout=dropout,
            bias=False,
        )

    def forward(self, x, return_prob=False, **kwargs):
        N, C, H, W = x.shape

        # Task: 22168020
        # In     Encode  Reason  Decode
        # 🔳🔳🔳  🟦🟦🟦  🟦🟦🟦  🔳🔳🔳
        # 🔳🔳🟩  🟦🟦🟨  🟦🟨🟨  🔳🟩🟩
        # 🔳🟩🟩  🟦🟨🟨  🟦🟨🟨  🔳🟩🟩
        
        # 🔳🔳🔳  🟦🟦🟦  🟦🟦🟦  🔳🔳🔳
        # 🔳🔳🟧  🟦🟦🟦  🟦🟦🟦  🔳🔳🟧
        # 🔳🟩🟩  🟦🟨🟨  🟦🟨🟨  🔳🟩🟩 

        x = self.abstractor(x)
        mem, x_mem = self.encoder(x)
        mem = self.reasoner(mem)
        y = self.decoder(x, mem, x_mem)

        y = y.view(N, H, W, C).permute(0, 3, 1, 2) # [N, C, H, W]

        if return_prob:
            y = F.softmax(y, dim=1) # [NS, C_prob]

        return y
