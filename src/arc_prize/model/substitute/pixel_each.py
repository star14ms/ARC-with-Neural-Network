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
    def __init__(self, num_classes=10, d_reduced_V_list=[9, 1], dim_feedforward=4, num_layers=1):
        super().__init__()
        self.V_extract = PixelVectorExtractor()
        max_width = 3
        max_height = 3
        self.max_V = max_width * max_height
        self.C1 = nn.Parameter(torch.randn([1, num_classes]))
        self.V1 = nn.Parameter(torch.randn([1, self.max_V]))

        self.C_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(num_classes+1, 1, dim_feedforward, batch_first=True, bias=False),
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.attn_C = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(num_classes, num_classes, dim_feedforward, batch_first=True, bias=False),
            num_layers=num_layers
        )
        self.attn_V = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(self.max_V, self.max_V, dim_feedforward, batch_first=True, bias=False),
            num_layers=num_layers
        )

        self.decoder = nn.Sequential()
        for i in range(len(d_reduced_V_list)-1):
            self.decoder.add_module(f'linear_{i}', nn.Linear(d_reduced_V_list[i], d_reduced_V_list[i+1], bias=False))
            if i != len(d_reduced_V_list)-2:
                self.decoder.add_module(f'relu_{i}', nn.ReLU())
        
    def forward(self, x):
        N, C, H, W = x.shape
        feature_L = self.V_extract(x)
        N, S, C_IN, L = feature_L.shape

        feature_L = feature_L.view(N*S, C_IN, L)
        feature_L = F.pad(feature_L, (0, self.max_V-feature_L.size(2)), mode='constant', value=0)

        feature_L = self.C_self_attn(feature_L.transpose(1, 2)).transpose(1, 2)[:,:-1] # [L, C] <- [L, C]

        feature_L = feature_L.reshape(N*S, C, L)
        feature = self.attn_V(feature_L, self.V1.repeat(N*S, 1, 1)) # [C, L] <- [1, L]
        feature = self.attn_C(feature.transpose(1, 2), self.C1.repeat(N*S, 1, 1)).transpose(1, 2) # [L, C] <- [C, C]

        y = feature.reshape(N*S*C, L)
        y = self.decoder(y) # [N*S, C]
        y = y.view(N, H, W, C).permute(0, 3, 1, 2) # [N, C, H, W]

        return y
