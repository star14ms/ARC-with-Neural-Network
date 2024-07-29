import torch
import torch.nn as nn
import torch.nn.functional as F

from arc_prize.utils.visualize import visualize_image_using_emoji


class PixelVectorExtractor(nn.Module):
    def __init__(self, pad_value=0):
        super().__init__()
        self.pad_value = pad_value

    def forward(self, x):
        N, C, H, W = x.shape

        # H_PAD = H - 1
        # W_PAD = W - 1
        # HL = H + H_PAD
        # WL = W + W_PAD
        
        H_PAD = 1
        W_PAD = 1
        HL = 3
        WL = 3
        
        # if not self.training:
        #     x_ = x.detach()
        #     breakpoint()

        x_C1 = torch.ones_like(x[:,:1])
        x_C1 = F.pad(x_C1, (W_PAD, W_PAD, H_PAD, H_PAD), mode='constant', value=1)

        x = F.pad(x, (W_PAD, W_PAD, H_PAD, H_PAD), mode='constant', value=self.pad_value)
        # visualize_image_using_emoji(x_[0], x[0])

        # add one more color dimension meaning padding or not
        x = torch.cat([x, x_C1], dim=1)
        x = x.permute(0, 2, 3, 1)

        x_expanded = torch.stack([x[n].unfold(0, HL, 1).unfold(1, WL, 1) for n in range(N)], dim=0) # [N, H, W, C, HL, WL]
        # visualize_image_using_emoji(x_expanded[0,0,6])
        x_expanded = x_expanded.view(N, H*W, C+1, HL*WL) # [N, S, C, V] where S = H*W, V = HL*WL

        return x_expanded


class PixelEachSubstitutor(nn.Module):
    def __init__(self, num_classes=10, d_reduced_V_list=[81, 32], d_reduced_V2_list=[32, 8, 1], dim_feedforward=1, num_layers=1):
        super().__init__()
        self.V_extract = PixelVectorExtractor()
        max_width = 9
        max_height = 9
        self.max_V = max_width * max_height
        self.V_feature = nn.Parameter(torch.randn([num_classes, d_reduced_V_list[-1]]))

        self.C_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(num_classes+1, 1, dim_feedforward, batch_first=True, bias=False),
            num_layers=num_layers
        )

        self.ff_V = nn.Sequential()
        for i in range(len(d_reduced_V_list)-1):
            self.ff_V.add_module(f'linear_{i}', nn.Linear(d_reduced_V_list[i], d_reduced_V_list[i+1], bias=False))
            if i != len(d_reduced_V_list)-2:
                self.ff_V.add_module(f'relu_{i}', nn.ReLU())

        self.attn_V = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_reduced_V_list[-1], d_reduced_V_list[-1], dim_feedforward, batch_first=True, bias=False),
            num_layers=num_layers
        )

        self.decoder = nn.Sequential()
        for i in range(len(d_reduced_V2_list)-1):
            self.decoder.add_module(f'linear_{i}', nn.Linear(d_reduced_V2_list[i], d_reduced_V2_list[i+1], bias=False))
            if i != len(d_reduced_V2_list)-2:
                self.decoder.add_module(f'relu_{i}', nn.ReLU())
        
    def forward(self, x):
        N, C, H, W = x.shape
        feature = self.V_extract(x)
        N, S, C_IN, V_IN = feature.shape

        feature = feature.view(N*S, C_IN, V_IN)
        feature = F.pad(feature, (0, self.max_V-feature.size(2)), mode='constant', value=0)

        # Effect: Determine Which Color is used for padding
        feature = self.C_self_attn(feature.transpose(2, 1)).transpose(2, 1) # [N*S, V, C]

        feature = self.ff_V(feature) # [N*S*C, V]
        V = feature.shape[-1]

        y = self.V_feature.repeat(N*S, 1, 1)
        y = self.attn_V(y, feature) # [N*S, C, V], [N*S, C_IN, V]
        y = y.view(N*S*C, V)

        y = self.decoder(y) # [N*S*C, 1]
        y = y.view(N, H, W, C).permute(0, 3, 1, 2) # [N, C, H, W]

        return y
