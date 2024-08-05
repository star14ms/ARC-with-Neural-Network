import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.utils.visualize import visualize_image_using_emoji


class PixelVectorExtractor(nn.Module):
    def __init__(self, max_width, max_height, pad_size, pad_n_head=None, pad_dim_feedforward=1, pad_num_layers=1, bias=False, pad_value=0, num_classes=10):
        super().__init__()
        self.pad_value = pad_value
        self.max_width = max_width
        self.max_height = max_height
        self.pad_size = pad_size

        self.attn_C = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(num_classes+1, pad_n_head if pad_n_head else num_classes+1, pad_dim_feedforward, batch_first=True, bias=bias),
            num_layers=pad_num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, x):
        N, C, H, W = x.shape

        H_PAD = self.pad_size if self.pad_size != -1 else H - 1
        W_PAD = self.pad_size if self.pad_size != -1 else W - 1
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
        x = x_max.view(N*H*W, C+1, -1).transpose(1, 2)
        mask_to_inf = (x == 0).all(dim=2)
        mask_to_0 = (x[:, :, -1:] == 1)
        colored_padding = self.attn_C(x, src_key_padding_mask=mask_to_inf) # [L, C+1] <- [L, C+1]
        colored_padding = colored_padding[:, :, :-1].softmax(dim=2)
        colored_padding = colored_padding * mask_to_0
        x = x[:, :, :-1] + colored_padding
        x = x.transpose(1, 2)

        return x


class RelatedPixelSampler(nn.Module): 
    def __init__(self, in_dim, out_dim, dropout=0.1, dim_feedforward=1, num_layers=1, bias=False, num_classes=10):
        super().__init__()
        # self.out_dim = out_dim

        # self.attn_L = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(in_dim, in_dim, dim_feedforward, batch_first=True, bias=bias),
        #     num_layers=1,
        #     enable_nested_tensor=False,
        # )
        # self.out = nn.Parameter(torch.zeros([out_dim, num_classes]), requires_grad=False)
        # self.attn_C = nn.TransformerDecoder(
        #     nn.TransformerDecoderLayer(num_classes, num_classes, dim_feedforward, batch_first=True, bias=bias),
        #     num_layers=1,
        # )

        # self.L_weight = nn.Parameter(torch.zeros([in_dim]) + 0.00001)
        # self.C_weight = nn.Parameter(torch.zeros([num_classes]) + 0.00001)
        # self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # NS, C, L = x.shape

        # x_prob = self.attn_L(x) # [C, L] <- [C, L]
        # x_prob = self.attn_C(x.transpose(1, 2)).transpose(1, 2) # [L, C] <- [L, C]

        # x_prob = x.sum(dim=1) # [N*S, L]
        # x_prob = x_prob.repeat(C, 1, 1).permute(1, 0, 2).reshape(NS*C, L) # [N*S, C, L]

        # # Drop random pixels
        # x_prob = x * self.L_weight # [N*S, C, L]
        # x_prob = x_prob.transpose(1, 2)
        # x_prob = x_prob * self.C_weight # [N*S, L, C]
        # x_prob = x_prob.transpose(1, 2)
        # x = self.attn_C(self.out.repeat(NS, 1, 1), x.transpose(1, 2)).transpose(1, 2) # [L_small, C] <- [L, C]

        # Remain Significant Relative Locations from Each Pixel
        # indices = torch.topk(x_prob, self.out_dim, dim=1)[1]
        # x = torch.gather(x.view(NS*C, L), 1, indices).view(NS, C, -1)

        return x


class Reasoner(nn.Module):
    def __init__(self, n_dim, d_reduced_V_list, L_n_head=None, L_dim_feedforward=1, L_num_layers=3, C_n_head=None, C_dim_feedforward=1, C_num_layers=1, bias=False, num_classes=10):
        super().__init__()

        self.attn_L = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(n_dim, L_n_head if L_n_head else n_dim, L_dim_feedforward, batch_first=True, bias=bias),
            num_layers=L_num_layers,
            enable_nested_tensor=False,
        )
        self.attn_C = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(num_classes, C_n_head if C_n_head else num_classes, C_dim_feedforward, batch_first=True, bias=bias),
            num_layers=C_num_layers,
            enable_nested_tensor=False,
        )

        self.decoder = nn.Sequential()
        for i in range(len(d_reduced_V_list)-1):
            self.decoder.add_module(f'linear_{i}', nn.Linear(d_reduced_V_list[i], d_reduced_V_list[i+1], bias=bias))
            if i != len(d_reduced_V_list)-2:
                self.decoder.add_module(f'relu_{i}', nn.ReLU())

    def forward(self, x):
        NS, C, L = x.shape

        # Attention across Locations and Colors
        x = self.attn_L(x) # [C, L] <- [C, L]
        x = self.attn_C(x.transpose(1, 2)).transpose(1, 2)  # [L, C] <- [L, C]

        # Decode Predicted Colors
        x = x.reshape(NS*C, L)
        x = self.decoder(x) # [NS, 1]

        return x.view(NS, C)


class PixelEachSubstitutor(nn.Module):
    def __init__(self, max_width=7, max_height=7, out_dim=49, d_reduced_V_list=[49, 1], pad_size=1, pad_n_head=None, pad_dim_feedforward=32, pad_num_layers=6, L_n_head=None, L_dim_feedforward=1, L_num_layers=6, C_n_head=None, C_dim_feedforward=1, C_num_layers=1, pad_value=0, dropout=0.0, num_classes=10):
        super().__init__()
        assert out_dim == d_reduced_V_list[0], f'out_dim ({out_dim}) should be same as d_reduced_V_list[0] ({d_reduced_V_list[0]})'
        assert pad_size != -1 and max_width >= 1 + 2*pad_size 

        self.abstractor = PixelVectorExtractor(
            max_width=max_width,
            max_height=max_height,
            pad_size=pad_size,
            pad_n_head=pad_n_head,
            pad_dim_feedforward=pad_dim_feedforward, 
            pad_num_layers=pad_num_layers,
            bias=False,
            pad_value=pad_value,
            num_classes=num_classes,
        )

        # self.pixel_sampler = RelatedPixelSampler(
        #     in_dim=max_width * max_height,
        #     out_dim=out_dim,
        #     dropout=dropout,
        #     dim_feedforward=dim_feedforward, 
        #     num_layers=num_layers,
        #     bias=False,
        #     num_classes=num_classes,
        # )

        self.reasoner = Reasoner(
            n_dim=out_dim, 
            d_reduced_V_list=d_reduced_V_list, 
            L_n_head=L_n_head,
            L_dim_feedforward=L_dim_feedforward,
            L_num_layers=L_num_layers,
            C_n_head=C_n_head,
            C_dim_feedforward=C_dim_feedforward,
            C_num_layers=C_num_layers,
            bias=False,
            num_classes=num_classes
        )

    def forward(self, x):
        N, C, H, W = x.shape

        x = self.abstractor(x)
        # x = self.pixel_sampler(x)
        y = self.reasoner(x)

        return y.view(N, H, W, C).permute(0, 3, 1, 2) # [N, C, H, W]
