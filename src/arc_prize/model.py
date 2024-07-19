import torch
from torch import nn

from arc_prize.model_components.convfixedkernel import Conv2dEncoderLayer
from utils.visualize import plot_kernels_and_outputs
from arc_prize.model_components.attention import ReductiveAttention


class ConvSameColorFeatureExtractor(nn.Module):
    def __init__(self, pad_value=-1, reduced_channels_encoder=[512, 32], reduced_channels_decoder=[512, 32], out_dim=4):
        super().__init__()
        self.V = reduced_channels_decoder[-1]
        self.out_dim = out_dim
        self.encoder = Conv2dEncoderLayer(1, reduced_channels_encoder, pad_value=pad_value, fixed_kernel=True)
        self.extender = Conv2dEncoderLayer(reduced_channels_encoder[-1], reduced_channels_decoder, pad_value=pad_value)
        self.attn_reduction = ReductiveAttention()
        self.attn_h = nn.Parameter(torch.randn(self.V))

        self.decoder = nn.Sequential(
            nn.Linear(self.V, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, out_dim, bias=True),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.transpose(1, 0) # [C, N, H, W]

        x_list = []
        for i, x_c in enumerate(x):
            if not torch.any(x_c == 1):
                x_c = (x_c.view(1, N, 1, H, W) + 1).repeat(1, 1,  self.out_dim, 1, 1) # .fill_(1) # default value is 0
                x_list.append(x_c)
                continue

            x_c = x_c.view(N, 1, H, W)
            x_c = self.encoder(x_c) # [N, V, H, W]
            x_c_seqs = []
            for _ in range(3): ### Hyperparameter
                x_c = self.extender(x_c) # [N, V, H, W]
                x_c_seqs.append(x_c.unsqueeze(0))
            x_c_seqs = torch.cat(x_c_seqs) # [S, N, V, H, W]

            S, N, V, H, W = x_c_seqs.shape
            x_c_seqs = x_c_seqs.permute(1, 3, 4, 0, 2).view(N*H*W, S, V)
            x_c = self.attn_reduction(x_c_seqs, self.attn_h.repeat(N*H*W, 1)).view(N, H, W, V).permute(0, 3, 1, 2)

            x_c = self.decoder(x_c.permute(0, 2, 3, 1).reshape(N*H*W, V)).view(N, H, W, -1).permute(0, 3, 1, 2)
            x_list.append(x_c.unsqueeze(0)) # [1, N, V, H, W]

        x = torch.cat(x_list) # [C, N, V, H, W]

        return x
        
    def to(self, *args, **kwargs):
        self.encoder = self.encoder.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class ShapeStableSolver(nn.Module):
    def __init__(self, pad_value=-1, reduced_channels_encoder=[512, 32], reduced_channels_decoder=[128, 32], hidden_size=1, num_classes=10, feature_dim=1, color_dim=1):
        super().__init__()
        self.feature_extractor = ConvSameColorFeatureExtractor(pad_value, reduced_channels_encoder, reduced_channels_decoder, out_dim=feature_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=num_classes, nhead=1, dim_feedforward=128, batch_first=True, bias=True),
            num_layers=1,
        )

        self.attn_input = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=num_classes, nhead=1, dim_feedforward=1, batch_first=True, bias=False),
            num_layers=1,
        )

        self.decoder = nn.Sequential(
            nn.Linear(color_dim, 1, bias=False),
            nn.BatchNorm1d(1),
        )

        # self.color_vector = nn.Parameter(torch.randn(num_classes, feature_dim)) # dim-1 because the first channel is the input

    def forward(self, x):
        N, C, H, W = x.shape
        y = self.feature_extractor(x) # [C, N, V, H, W]
        y = y.permute(1, 3, 4, 0, 2).reshape(N*H*W, C, -1)

        # y = self.color_vector.repeat(N*H*W, 1, 1) # [N*H*W, C, V]
        # y = torch.cat([y], dim=2) # [N*H*W, C, V+FV]
        # V = self.color_vector.shape[1] + y.shape[2]
        V = y.shape[2]
        y = y.transpose(2, 1) # [N*H*W, V, C]
        y = self.encoder(y)

        x = x.permute(0, 2, 3, 1).reshape(N*H*W, 1, C).repeat(1, V, 1) # [N*H*W, V, C]
        y = self.attn_input(y, x)
        y = y.transpose(2, 1) # [N*H*W, C, V]

        y = y.reshape(N*H*W*C, V)
        y = self.decoder(y) # [N*H*W*C, 1]
        y = y.view(N, H, W, C).permute(0, 3, 1, 2) # [N, C, H, W]

        return y

    def to(self, *args, **kwargs):
        self.feature_extractor = self.feature_extractor.to(*args, **kwargs)
        return super().to(*args, **kwargs)
