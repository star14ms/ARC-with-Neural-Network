import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.utils.visualize import visualize_image_using_emoji


class PixelVectorExtractor(nn.Module):
    def __init__(self, pad_size, max_width, max_height, pad_class_initial=0, pad_n_head=None, pad_dim_feedforward=1, dropout=0.1, pad_num_layers=1, bias=False, num_classes=10):
        super().__init__()
        self.pad_size = pad_size
        self.max_width = max_width
        self.max_height = max_height
        self.pad_class_initial = pad_class_initial

        self.attn_C = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(num_classes+1, pad_n_head if pad_n_head else num_classes+1, pad_dim_feedforward, dropout=dropout, batch_first=True, bias=bias),
            num_layers=pad_num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, x):
        N, C, H, W = x.shape
        H_PAD = self.pad_size if self.pad_size != -1 else H - 1
        W_PAD = self.pad_size if self.pad_size != -1 else W - 1
        HL = 1 + 2*H_PAD
        WL = 1 + 2*W_PAD
        
        if self.pad_size != 0:
            x_C10 = torch.zeros_like(x[:,:1], dtype=x.dtype, device=x.device)
            x_C10 = F.pad(x_C10, (W_PAD, W_PAD, H_PAD, H_PAD), mode='constant', value=1)

            # add one more color dimension meaning padding or not
            x_C0 = F.pad(x[:, self.pad_class_initial:self.pad_class_initial+1], (W_PAD, W_PAD, H_PAD, H_PAD), mode='constant', value=1)
            x_C1to9 = F.pad(x[:, self.pad_class_initial+1:], (W_PAD, W_PAD, H_PAD, H_PAD), mode='constant', value=0)
            x = torch.cat([x_C0, x_C1to9, x_C10], dim=1)

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
        if self.pad_size != 0:
            x = x.transpose(1, 2)
            mask_to_inf = (x == 0).all(dim=2)
            mask_to_0 = (x[:, :, -1:] == 1)
            colored_padding = self.attn_C(x, src_key_padding_mask=mask_to_inf) # [L, C+1] <- [L, C+1]
            colored_padding = colored_padding[:, :, :-1].softmax(dim=2)
            colored_padding = colored_padding * mask_to_0
            x = x[:, :, :-1] + colored_padding
            x = x.transpose(1, 2)

        return x


class RelatedPixelSampler(nn.Module): 
    def __init__(self, in_dim, dims_reduced, bias=False):
        super().__init__()
        d_reduced_L_list = [in_dim] + dims_reduced

        self.ff_L = nn.Sequential()
        for i in range(len(d_reduced_L_list)-1):
            self.ff_L.add_module(f'linear_{i}', nn.Linear(d_reduced_L_list[i], d_reduced_L_list[i+1], bias=bias))
            if i != len(d_reduced_L_list)-2:
                self.ff_L.add_module(f'relu_{i}', nn.ReLU())
        
    def forward(self, x):
        NS, C, L = x.shape

        # Remain Significant Relative Locations from Each Pixel
        x = self.ff_L(x.reshape(NS*C, L)).reshape(NS, C, -1) # [N*S*C, L]

        return x


class Reasoner(nn.Module):
    def __init__(self, n_dim, dims_decoded, L_num_layers=6, L_n_head=None, L_dim_feedforward=1, C_num_layers=1, C_n_head=None, C_dim_feedforward=1, dropout=0.1, bias=False, num_classes=10):
        super().__init__()
        dims_decoded =  [n_dim] + dims_decoded

        # TODO: EncoderLayer_1head -> EncoderLayer_nhead -> DecoderLayer_nhead -> DecoderLayer_1head or DecoderLayer_1head -> DecoderLayer_nhead
        self.attn_L = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(n_dim, L_n_head if L_n_head else n_dim, L_dim_feedforward, dropout=dropout, batch_first=True, bias=bias),
            num_layers=L_num_layers,
            enable_nested_tensor=False,
        )
        self.attn_C = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(num_classes, C_n_head if C_n_head else num_classes, C_dim_feedforward, dropout=dropout, batch_first=True, bias=bias),
            num_layers=C_num_layers,
            enable_nested_tensor=False,
        )

        self.decoder = nn.Sequential()
        for i in range(len(dims_decoded)-1):
            self.decoder.add_module(f'linear_{i}', nn.Linear(dims_decoded[i], dims_decoded[i+1], bias=bias))
            if i != len(dims_decoded)-2:
                self.decoder.add_module(f'relu_{i}', nn.ReLU())

    def forward(self, x):
        NS, C, L = x.shape

        # Attention across Locations and Colors
        x = self.attn_L(x) + 0.00001 # [C, L] <- [C, L]
        x = self.attn_C(x.transpose(1, 2)).transpose(1, 2)  # [L, C] <- [L, C]

        # Decode Predicted Colors
        x = x.reshape(NS*C, L)
        x = self.decoder(x) # [NS, 1]

        return x.view(NS, C)


class PixelEachSubstitutor(nn.Module):
    def __init__(self, pad_size=-1, max_width=61, max_height=61, dims_reduced=[512, 128, 64], dims_decoded=[32, 8, 1], pad_class_initial=0, pad_num_layers=1, pad_n_head=None, pad_dim_feedforward=1, L_num_layers=6, L_n_head=None, L_dim_feedforward=1, C_num_layers=1, C_n_head=None, C_dim_feedforward=1, dropout=0.1, num_classes=10):
        super().__init__()
        assert pad_size != -1 and max_width >= 1 + 2*pad_size and max_height >= 1 + 2*pad_size
        self.skip_sampler = True if dims_reduced[-1] == max_width * max_height else False

        self.abstractor = PixelVectorExtractor(
            max_width=max_width,
            max_height=max_height,
            pad_size=pad_size,
            pad_n_head=pad_n_head,
            pad_dim_feedforward=pad_dim_feedforward, 
            dropout=dropout,
            pad_num_layers=pad_num_layers,
            bias=False,
            pad_class_initial=pad_class_initial,
            num_classes=num_classes,
        )

        self.pixel_sampler = RelatedPixelSampler(
            in_dim=max_width * max_height,
            dims_reduced=dims_reduced,
            bias=False,
        )

        self.reasoner = Reasoner(
            n_dim=dims_reduced[-1], 
            dims_decoded=dims_decoded, 
            L_num_layers=L_num_layers,
            L_n_head=L_n_head,
            L_dim_feedforward=L_dim_feedforward,
            C_num_layers=C_num_layers,
            C_n_head=C_n_head,
            C_dim_feedforward=C_dim_feedforward,
            dropout=dropout,
            bias=False,
            num_classes=num_classes
        )

    def forward(self, x):
        N, C, H, W = x.shape

        x = self.abstractor(x)
        if not self.skip_sampler:
            x = self.pixel_sampler(x)
        y = self.reasoner(x)

        return y.view(N, H, W, C).permute(0, 3, 1, 2) # [N, C, H, W]
