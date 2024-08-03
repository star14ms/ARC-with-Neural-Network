import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.utils.visualize import visualize_image_using_emoji


class PixelVectorExtractor(nn.Module):
    def __init__(self, max_width, max_height, dim_feedforward, num_layers=1, bias=False, pad_value=0, num_classes=10):
        super().__init__()
        self.pad_value = pad_value
        self.max_width = max_width
        self.max_height = max_height

        self.C_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(num_classes+1, 1, dim_feedforward, batch_first=True, bias=bias),
            num_layers=num_layers,
            enable_nested_tensor=False
        )

    def forward(self, x):
        N, C, H, W = x.shape

        # H_PAD = H - 1
        # W_PAD = W - 1
        # HL = H + H_PAD
        # WL = W + W_PAD

        H_PAD = 3
        W_PAD = 3
        HL = 1 + 2*H_PAD
        WL = 1 + 2*W_PAD

        x_C1 = torch.zeros_like(x[:,:1])
        x_C1 = F.pad(x_C1, (W_PAD, W_PAD, H_PAD, H_PAD), mode='constant', value=1)

        # add one more color dimension meaning padding or not
        x = F.pad(x, (W_PAD, W_PAD, H_PAD, H_PAD), mode='constant', value=self.pad_value)
        x = torch.cat([x, x_C1], dim=1)
        x = x.permute(0, 2, 3, 1)

        x = torch.stack([x[n].unfold(0, HL, 1).unfold(1, WL, 1) for n in range(N)], dim=0) # [N, H, W, C, HL, WL]
        x = x.view(N*H*W*(C+1), HL, WL) # [N, S, C, L] where S = H*W

        # Padding to Max Length
        x_max = torch.full([N*H*W*(C+1), self.max_height, self.max_width], fill_value=self.pad_value, dtype=x.dtype, device=x.device)
        x_max[:,:HL,:WL] = x

        # Predict Padding Colors
        x_max = x_max.view(N*H*W, C+1, -1)
        x_max = self.C_self_attn(x_max.transpose(1, 2)).transpose(1, 2)[:,:-1] # [L, C] <- [L, C]

        return x_max


class RelatedPixelSampler(nn.Module): 
    def __init__(self, n_dim, d_reduced_L_list, dropout=0.1, bias=False):
        super().__init__()
        self.L_weight = nn.Parameter(torch.ones([n_dim]))
        self.dropout = nn.Dropout(dropout)
        self.n_dim = n_dim

        self.ff_L = nn.Sequential()
        for i in range(len(d_reduced_L_list)-1):
            self.ff_L.add_module(f'linear_{i}', nn.Linear(d_reduced_L_list[i], d_reduced_L_list[i+1], bias=bias))
            if i != len(d_reduced_L_list)-2:
                self.ff_L.add_module(f'relu_{i}', nn.ReLU())

    def forward(self, x):
        # Drop random pixels
        x = x * self.dropout(self.L_weight)

        # Remain Significant Relative Locations from Each Pixel
        x = self.ff_L(x.reshape(-1, self.n_dim)) # [N*S*C, L]

        return x


class Reasoner(nn.Module):
    def __init__(self, n_dim, d_reduced_V_list, dim_feedforward=1, num_layers=1, bias=False, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        self.C_weight = nn.Parameter(torch.randn(num_classes))
        self.attn_C = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(num_classes, num_classes, dim_feedforward, batch_first=True, bias=bias),
            num_layers=num_layers
        )

        self.L_weight = nn.Parameter(torch.randn(n_dim))
        self.attn_L = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(n_dim, n_dim, dim_feedforward, batch_first=True, bias=bias),
            num_layers=num_layers
        )

        self.decoder = nn.Sequential()
        for i in range(len(d_reduced_V_list)-1):
            self.decoder.add_module(f'linear_{i}', nn.Linear(d_reduced_V_list[i], d_reduced_V_list[i+1], bias=bias))
            if i != len(d_reduced_V_list)-2:
                self.decoder.add_module(f'relu_{i}', nn.ReLU())
                
    def forward(self, x):
        NSC, L = x.shape
        C = self.num_classes
        NS = NSC // C

        # Attention across Locations and Colors
        x = x.reshape(NS, C, L)
        x = self.attn_L(x, self.L_weight.repeat(NS, 1, 1)) # [C, L] <- [1, L]
        x = self.attn_C(x.transpose(1, 2), self.C_weight.repeat(NS, 1, 1)).transpose(1, 2) # [L, C] <- [C, C]

        # Decode Predicted Colors
        x = x.reshape(NS*C, L)
        x = self.decoder(x) # [N*S, C]
        
        return x


class PixelEachSubstitutor(nn.Module):
    def __init__(self, d_reduced_L_list=[49, 32], d_reduced_V_list=[32, 1], dim_feedforward=1, num_layers=1, pad_value=0, dropout=0.1, num_classes=10):
        super().__init__()
        self.max_width = 7
        self.max_height = 7
        max_L = self.max_width * self.max_height

        self.abstractor = PixelVectorExtractor(
            max_width=self.max_width,
            max_height=self.max_height,
            dim_feedforward=dim_feedforward, 
            num_layers=num_layers,
            bias=False,
            pad_value=pad_value,
            num_classes=num_classes,
        )

        self.pixel_sampler = RelatedPixelSampler(
            n_dim=max_L,
            d_reduced_L_list=d_reduced_L_list,
            dropout=dropout,
            bias=False,
        )

        self.reasoner = Reasoner(
            n_dim=d_reduced_L_list[-1], 
            d_reduced_V_list=d_reduced_V_list, 
            dim_feedforward=dim_feedforward, 
            num_layers=1,
            bias=False,
            num_classes=num_classes
        )

    def forward(self, x):
        N, C, H, W = x.shape

        y = self.abstractor(x)
        y = self.pixel_sampler(y)
        y = self.reasoner(y)

        return y.view(N, H, W, C).permute(0, 3, 1, 2) # [N, C, H, W]
