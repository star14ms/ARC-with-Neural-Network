import torch
from torch import nn

from arc_prize.model_components.convfixedkernel import Conv2dEncoderLayer
from utils.visualize import plot_kernels_and_outputs
from arc_prize.model_components.attention import ReductiveAttention


class ConvSameColorFeatureExtractor(nn.Module):
    def __init__(self, pad_value=-1, reduced_channels_encoder=[512, 32], reduced_channels_decoder=[512, 32], hidden_dim=16, out_dim=4):
        super().__init__()
        self.out_dim = out_dim
        self.V = reduced_channels_encoder[-1]
        self.encoder = Conv2dEncoderLayer(1, reduced_channels_encoder, pad_value=pad_value, fixed_kernel=True)
        self.extender = Conv2dEncoderLayer(reduced_channels_encoder[-1], reduced_channels_decoder, pad_value=pad_value)
        self.attn_reduction = ReductiveAttention()
        self.attn_h = nn.Parameter(torch.randn(self.V))
        
        # self.feature_extractor = nn.Sequential(
        #     nn.Linear(self.V, hidden_dim, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, self.V, bias=False),
        #     nn.ReLU(),
        # )

        # self.attn_conv = nn.TransformerDecoder(
        #     nn.TransformerDecoderLayer(d_model=1, nhead=1, dim_feedforward=4, batch_first=True, bias=False),
        #     num_layers=1,
        # )

        self.decoder = nn.Sequential(
            nn.Linear(self.V, out_dim, bias=True),
        )
        # self.decoder_initial = nn.Sequential(
        #     nn.Linear(self.V, out_dim, bias=False),
        # )
        # self.feature_final = nn.Sequential(
        #     nn.Linear(out_dim, out_dim, bias=False),
        # )

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

            features_time = []
            features_time.append(x_c.unsqueeze(0))

            for _ in range(max(H, W)//2+1): ### Varialble (Depends on Input Shape)
                x_c = self.extender(x_c) # [N, V, H, W]
                features_time.append(x_c.unsqueeze(0))
            features_time = torch.cat(features_time) # [S, N, V, H, W]

            S, N, V, H, W = features_time.shape
            features_time = features_time.permute(1, 3, 4, 0, 2).view(N*H*W, S, V)
            feature = self.attn_reduction(features_time, self.attn_h.repeat(N*H*W, 1))

            # feature = x_c.permute(0, 2, 3, 1).view(N*H*W, self.V, 1)
            # # feature = self.decoder_initial(feature).view(N*H*W, -1, 1) # [N, V2, H, W]
            # V2 = feature.shape[1]

            # for _ in range(max(H, W)//2+1): ### Varialble (Depends on Input Shape)
            #     x_c = self.extender(x_c) # [N, V, H, W]

            #     feature_moment = x_c.permute(0, 2, 3, 1).view(N*H*W, self.V, 1)
            #     # feature_moment = self.decoder(feature_moment).view(N*H*W, V2, 1) # [N, V2, H, W]
            #     feature = self.attn_conv(feature, feature_moment)
            # # feature = self.feature_final(feature.view(N*H*W, V2)).view(N, H, W, V2).permute(0, 3, 1, 2) # [N, V2, H, W]
            # feature = feature.view(N, H, W, V2).permute(0, 3, 1, 2) # [N, V2, H, W]

            x_c = self.decoder(feature)  # [N*H*W, V]
            x_c = x_c.view(N, H, W, self.V).permute(0, 3, 1, 2)
            x_list.append(x_c.unsqueeze(0)) # [1, N, V, H, W]

        x = torch.cat(x_list) # [C, N, V, H, W]
        return x
        
    def to(self, *args, **kwargs):
        self.encoder = self.encoder.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class ShapeStableSolver(nn.Module):
    def __init__(self, pad_value=-1, reduced_channels_encoder=[512, 32], reduced_channels_decoder=[128, 32], num_classes=10, feature_dim=32):
        super().__init__()
        self.feature_extractor = ConvSameColorFeatureExtractor(pad_value, reduced_channels_encoder, reduced_channels_decoder, out_dim=feature_dim)

        self.attn_input = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=num_classes, nhead=1, dim_feedforward=1, batch_first=True, bias=False),
            num_layers=1,
        )
        # self.attn_feature = nn.TransformerDecoder(
        #     nn.TransformerDecoderLayer(d_model=feature_dim, nhead=1, dim_feedforward=1, batch_first=True, bias=True),
        #     num_layers=2,
        # )
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=num_classes, nhead=1, dim_feedforward=128, batch_first=True, bias=True),
            num_layers=1,
        )

        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 1, bias=False),
            nn.BatchNorm1d(1),
        )

        # self.color_vector = nn.Parameter(torch.randn(num_classes, feature_dim)) # Task_specific color vector

    def forward(self, x):
        N, C, H, W = x.shape
        feature = self.feature_extractor(x) # [C, N, V, H, W]
        V = feature.shape[2]
        feature = feature.permute(1, 3, 4, 0, 2).reshape(N*H*W, C, V) # [N*H*W, C, V]

        y = feature
        y = y.transpose(2, 1)
        y = self.encoder(y)
        # y = self.color_vector.repeat(N*H*W, 1, 1) # [N*H*W, C, V]
        # y = self.attn_feature(y, feature)

        x = x.permute(0, 2, 3, 1).reshape(N*H*W, 1, C).repeat(1, V, 1) # [N*H*W, V, C]
        y = self.attn_input(y, x) # [N*H*W, V, C]
        y = y.transpose(2, 1)

        y = y.reshape(N*H*W*C, V)
        y = self.decoder(y) # [N*H*W*C, 1]
        y = y.view(N, H, W, C).permute(0, 3, 1, 2) # [N, C, H, W]

        return y

    def to(self, *args, **kwargs):
        self.feature_extractor = self.feature_extractor.to(*args, **kwargs)
        return super().to(*args, **kwargs)
