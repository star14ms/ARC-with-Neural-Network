import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.utils.visualize import visualize_image_using_emoji


class PixelVectorExtractor(nn.Module):
    def __init__(self, n_range_search, max_width, max_height, pad_class_initial=0, pad_n_head=None, pad_dim_feedforward=1, dropout=0.1, pad_num_layers=1, bias=False, n_class=10):
        super().__init__()
        self.n_range_search = n_range_search
        self.max_width = max_width
        self.max_height = max_height
        self.pad_class_initial = pad_class_initial

        # self.attn_C = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(n_class+1, pad_n_head if pad_n_head is not None else n_class+1, pad_dim_feedforward, dropout=dropout, batch_first=True, bias=bias),
        #     num_layers=pad_num_layers,
        #     enable_nested_tensor=False,
        # )

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
        # if self.n_range_search != 0:
        #     x = x.transpose(1, 2)
        #     mask_to_inf = (x == 0).all(dim=2)
        #     mask_to_0 = (x[:, :, -1:] == 1)
        #     colored_padding = self.attn_C(x, src_key_padding_mask=mask_to_inf) # [L, C+1] <- [L, C+1]
        #     colored_padding = colored_padding[:, :, :-1].softmax(dim=2)
        #     colored_padding = colored_padding * mask_to_0
        #     x = x[:, :, :-1] + colored_padding
        #     x = x.transpose(1, 2)

        if self.n_range_search != 0:
            x = x[:, :-1] # [N*H*W, C, L]

        ### TODO: PixelVector = Pixels Located Relatively + Pixels Located Absolutely (363442ee, 63613498, aabf363d) + 3 Pixels with point-symmetric and line-symmetric relationships (3631a71a, 68b16354)
        ### TODO: PixelEachSubstitutorOverTime (045e512c, 22168020, 22eb0ac0, 3bd67248, 508bd3b6, 623ea044), 

        return x


class RelatedPixelSampler(nn.Module): 
    def __init__(self, L_dims_encoded, bias=False):
        super().__init__()

        self.ff_L = nn.Sequential()
        for i in range(len(L_dims_encoded)-1):
            self.ff_L.add_module(f'linear_{i}', nn.Linear(L_dims_encoded[i], L_dims_encoded[i+1], bias=bias))
            if i != len(L_dims_encoded)-2:
                self.ff_L.add_module(f'relu_{i}', nn.ReLU())

    def forward(self, x):
        NS, C, L = x.shape

        # Encode Locations
        x = self.ff_L(x.reshape(NS*C, L)).reshape(NS, C, -1) # [N*S*C, L]: 2. Location Encoding

        return x


class Reasoner(nn.Module):
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
        NS, VC, VL = x.shape

        # 1. Attention Across Vectors and Classify
        x = self.attn_VL_self(x) # [VC, VL] <- [VC, VL]
        x = self.attn_VC_self(x.transpose(1, 2)).transpose(1, 2)  # [VL, VC] <- [VL, VC]

        # 2. Determine Output Class
        x = x.reshape(NS*VC, VL)
        x = self.decoder_VL(x) # [NS*VC, 1]
        x = x.view(NS, VC, -1)

        return x


class PixelEachSubstitutor(nn.Module):
    def __init__(self, n_range_search=-1, max_width=61, max_height=61, L_dims_encoded=[512, 128, 64], L_dims_decoded=[32, 8, 1], pad_class_initial=0, pad_num_layers=1, pad_n_head=None, pad_dim_feedforward=1, skip_sampler=False, L_num_layers=6, L_n_head=None, L_dim_feedforward=1, C_num_layers=1, C_n_head=None, C_dim_feedforward=1, dropout=0.1, n_class=10):
        super().__init__()
        assert n_range_search != -1 and max_width >= 1 + 2*n_range_search and max_height >= 1 + 2*n_range_search
        self.skip_sampler = skip_sampler
        L_dim = max_width * max_height
        L_dims_encoded = [L_dim] + L_dims_encoded

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

        self.pixel_sampler = RelatedPixelSampler(
            L_dims_encoded=L_dims_encoded,
            bias=False,
        )

        self.reasoner = Reasoner(
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
        x = self.pixel_sampler(x) if not self.skip_sampler else x
        y = self.reasoner(x)
        y = y.view(N, H, W, C).permute(0, 3, 1, 2) # [N, C, H, W]

        if return_prob:
            y = F.softmax(y, dim=1) # [NS, C_prob]

        return y
