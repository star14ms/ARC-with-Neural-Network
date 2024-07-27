import torch
from torch import nn

from arc_prize.model.components.convfixedkernel import Conv2dEncoderLayer
from arc_prize.model.components.attention import ReductiveAttention


class ConvFeatureExtractor(nn.Module):
    def __init__(self, pad_value=-1, reduced_channels_encoder=[512, 128, 32], reduced_channels_decoder=[128, 32], recursion=3):
        super().__init__()
        self.recursion = recursion
        self.encoder = Conv2dEncoderLayer(1, reduced_channels_encoder, pad_value=pad_value, fixed_kernel=True)
        self.extender = Conv2dEncoderLayer(reduced_channels_encoder[-1], reduced_channels_decoder, pad_value=pad_value)
        self.decoder = Conv2dEncoderLayer(reduced_channels_decoder[-1], reduced_channels_decoder, pad_value=pad_value, out_one_channel=True)
        self.attn_reduction = ReductiveAttention()
        self.attn_h = nn.Parameter(torch.randn(reduced_channels_decoder[-1]))

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.transpose(1, 0) # [C, N, H, W]

        x_list = []
        for i, x_c in enumerate(x):
            if not torch.any(x_c == 1):
                x_c = x_c.unsqueeze(0) + 1 # .fill_(1) # default value is 0
                x_list.append(x_c)
                continue

            x_c = x_c.view(N, 1, H, W)
            x_c = self.encoder(x_c) # [N, V, H, W]
            x_c_seqs = []
            for _ in range(self.recursion): ### Hyperparameter
                x_c = self.extender(x_c) # [N, V, H, W]
                x_c_seqs.append(x_c.unsqueeze(0))
            x_c_seqs = torch.cat(x_c_seqs) # [S, N, V, H, W]

            S, N, V, H, W = x_c_seqs.shape
            x_c_seqs = x_c_seqs.permute(1, 3, 4, 0, 2).reshape(N*H*W, S, V)
            x_c = self.attn_reduction(x_c_seqs, self.attn_h.repeat(N*H*W, 1)).view(N, H, W, V).permute(0, 3, 1, 2)
            x_c = self.decoder(x_c).view(N, H, W) # [1, N, H, W]
            x_list.append(x_c.unsqueeze(0))

        x = torch.cat(x_list) 

        return x.transpose(1, 0) # [N, C, H, W]
        
    def to(self, *args, **kwargs):
        self.encoder = self.encoder.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class FillerKeepInputIgnoreColor(nn.Module):
    def __init__(self, pad_value=-1, reduced_channels_encoder=[512, 128, 32], reduced_channels_decoder=[128, 32]):
        super().__init__()
        self.feature_extractor = ConvFeatureExtractor(pad_value, reduced_channels_encoder, reduced_channels_decoder)

    def forward(self, x):
        N, C, H, W = x.shape

        x = self.feature_extractor(x) # [N, C, H, W]
        y = x[:, 1:].sum(dim=1).view(N, H, W)

        return y

    def to(self, *args, **kwargs):
        self.feature_extractor = self.feature_extractor.to(*args, **kwargs)
        return super().to(*args, **kwargs)
