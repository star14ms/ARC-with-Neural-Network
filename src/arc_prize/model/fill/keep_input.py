import torch
from torch import nn

from arc_prize.model.components.convfixedkernel import Conv2dEncoderLayer


class ConvSameColorFeatureExtractor(nn.Module):
    def __init__(self, pad_value=-1, reduced_channels_encoder=[512, 32], reduced_channels_decoder=[512, 32], d_feature=32):
        super().__init__()
        self.d_feature = d_feature
        self.V = reduced_channels_encoder[-1]
        self.encoder = Conv2dEncoderLayer(1, reduced_channels_encoder, pad_value=pad_value, fixed_kernel=True)
        self.extender = Conv2dEncoderLayer(reduced_channels_encoder[-1], reduced_channels_decoder, pad_value=1)
        self.decoder_initial = nn.Sequential(
            nn.Linear(self.V, d_feature, bias=False),
        )

        self.attn_conv = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.V, nhead=1, dim_feedforward=32, batch_first=True, bias=False),
            num_layers=1,
        )
        self.decoder_secondary_feature = nn.Sequential(
            nn.Linear(self.V, 1, bias=True),
        )
        self.decoder_final = nn.Sequential(
            nn.Linear(d_feature, 1, bias=True),
        )
        self.feature_map = nn.Parameter(torch.randn(d_feature, self.V))

    def forward(self, x, n_recurrance_feature_extraction=None):
        N, C, H, W = x.shape
        x = x.transpose(1, 0) # [C, N, H, W]
        n_recurrance_feature_extraction = n_recurrance_feature_extraction or max(H, W)

        x_list = []
        for i, x_c in enumerate(x):
            if not torch.any(x_c == 1):
                x_c = (x_c.view(1, N, 1, H, W) + 1).repeat(1, 1, 1, 1, 1) # .fill_(1) # default value is 0
                x_list.append(x_c)
                continue

            x_c = x_c.view(N, 1, H, W)
            x_c = self.encoder(x_c) # [N, V, H, W]
            V = x_c.shape[1]

            feature = x_c.permute(0, 2, 3, 1).reshape(N*H*W, self.V)
            feature = self.decoder_initial(feature).view(N*H*W, -1, 1).repeat(1, 1, V) # [N*H*W, V2, V]
            V2 = feature.shape[1]

            for _ in range(n_recurrance_feature_extraction): ### Varialble (Depends on Input Shape)
                x_c = self.extender(x_c) # [N, V, H, W]

                feature = self.attn_conv(feature, x_c.permute(0, 2, 3, 1).reshape(N*H*W, 1, V))
            feature = feature.view(N, H, W, V2, V).view(N*H*W*V2, V)

            x_c = self.decoder_secondary_feature(feature) # [N*H*W*V2, 1]
            x_c = self.decoder_final(x_c.view(N*H*W, V2)) # [N*H*W, 1]
            x_c = x_c.view(N, H, W, 1).permute(0, 3, 1, 2)
            x_list.append(x_c.unsqueeze(0)) # [1, N, V, H, W]

        x = torch.cat(x_list) # [C, N, V, H, W]
        return x
        
    def to(self, *args, **kwargs):
        self.encoder = self.encoder.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class FillerKeepInput(nn.Module):
    def __init__(self, pad_value=-1, reduced_channels_encoder=[512, 32], reduced_channels_decoder=[128, 32], num_classes=10, d_feature=16, d_color_feature=32):
        super().__init__()
        self.feature_extractor = ConvSameColorFeatureExtractor(pad_value, reduced_channels_encoder, reduced_channels_decoder, d_feature=d_feature)
        self.d_color_feature = d_color_feature
    
        self.attn_input = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=num_classes, nhead=1, dim_feedforward=1, batch_first=True, bias=False),
            num_layers=1,
        )
        self.attn_feature = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=num_classes, nhead=1, dim_feedforward=1, batch_first=True, bias=True),
            num_layers=1,
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_color_feature, 1, bias=False),
            nn.BatchNorm1d(1),
        )
        self.color_vector = nn.Parameter(torch.randn(num_classes, d_color_feature)) # Task_specific color vector

    def forward(self, x, n_recurrance_feature_extraction=None):
        N, C, H, W = x.shape
        V = self.d_color_feature
        feature = self.feature_extractor(x, n_recurrance_feature_extraction) # [C, N, V, H, W]
        feature = feature.permute(1, 3, 4, 0, 2).reshape(N*H*W, C, -1) # [N*H*W, C, V]

        y = self.color_vector.repeat(N*H*W, 1, 1)
        feature = feature.transpose(2, 1).repeat(1, V, 1) # [N*H*W, V, C]

        y = y.transpose(2, 1) # [N*H*W, V, C]
        y = self.attn_feature(y, feature)
        x = x.permute(0, 2, 3, 1).reshape(N*H*W, 1, C)
        y = self.attn_input(y, x)
        y = y.transpose(2, 1) # [N*H*W, C, V]

        y = y.reshape(N*H*W*C, V)
        y = self.decoder(y) # [N*H*W*C, 1]
        y = y.view(N, H, W, C).permute(0, 3, 1, 2) # [N, C, H, W]

        return y

    def to(self, *args, **kwargs):
        self.feature_extractor = self.feature_extractor.to(*args, **kwargs)
        return super().to(*args, **kwargs)
