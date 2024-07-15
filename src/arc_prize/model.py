import torch
import torch.nn.functional as F
from torch import nn

from arc_prize.model_custom import Conv2dFixedKernel
from utils.visualize import plot_kernels_and_outputs


class Conv2dEncoderLayer(nn.Module):
    def __init__(self, in_channels=1, reduced_channels=[512, 128, 32], fixed_kernel=False, out_one_channel=False, kernel_size=(3, 3), stride=1, padding=1, pad_value=-1):
        super().__init__()
        self.padding = padding
        self.pad_value = pad_value
        self.out_one_channel = out_one_channel

        if fixed_kernel:
            self.conv = Conv2dFixedKernel(in_channels, kernel_size=kernel_size, stride=stride, padding=0)
        else:
            self.conv = nn.Conv2d(in_channels, reduced_channels[0], kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        self.activation = nn.ReLU()

        self.linear_layers = nn.Sequential()
        for i in range(len(reduced_channels)-1):
            self.linear_layers.add_module(f'linear_{i}', nn.Linear(reduced_channels[i], reduced_channels[i+1], bias=False))
            self.linear_layers.add_module(f'relu_{i}', nn.ReLU())
        self.linear_layers.add_module(f'norm', nn.BatchNorm1d(reduced_channels[-1]))

        if out_one_channel:
            self.out = nn.Linear(reduced_channels[-1], 1, bias=False)

    def forward(self, x, **kwargs):
        N, H, W = x.shape[0], x.shape[2], x.shape[3]
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=self.pad_value)
        x = self.activation(self.conv(x)) # [N, C, H, W]
        x = x.permute(0, 2, 3, 1).reshape(N*H*W, -1) # [N*H*W, C]
        x = self.linear_layers(x)

        if self.out_one_channel:
            x = self.out(x)
            return x.view(N, H, W)
        else:
            return x.view(N, H, W, -1).permute(0, 3, 1, 2)

    def to(self, *args, **kwargs):
        self.conv = self.conv.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    
class ConvFeatureExtractor(nn.Module):
    def __init__(self, pad_value=-1, reduced_channels_encoder=[512, 128, 32], reduced_channels_decoder=[128, 32]):
        super().__init__()
        self.encoder = Conv2dEncoderLayer(1, reduced_channels_encoder, pad_value=pad_value, fixed_kernel=True)
        self.extender = Conv2dEncoderLayer(reduced_channels_encoder[-1], reduced_channels_decoder, pad_value=pad_value)
        self.decoder = Conv2dEncoderLayer(reduced_channels_decoder[-1], reduced_channels_decoder, pad_value=pad_value, out_one_channel=True)

    def forward(self, x):
        x = x.transpose(1, 0) # [C, N, H, W]

        x_list = []
        for i, x_one in enumerate(x):
            if not torch.any(x_one == 1):
                x_one = x_one.unsqueeze(0) # .fill_(1)
                x_list.append(x_one)
                continue

            x_one = x_one.view(1, *x_one.shape)
            x_one = self.encoder(x_one)
            for _ in range(5):
                x_one = self.extender(x_one) # [N, H, W]
            x_one = self.decoder(x_one).unsqueeze(0)
            x_list.append(x_one)

        x = torch.cat(x_list) + 1 # default value is 0

        return x.transpose(1, 0) # [N, C, H, W]
        
    def to(self, *args, **kwargs):
        self.encoder = self.encoder.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class ShapeStableSolver(nn.Module):
    def __init__(self, pad_value=-1, reduced_channels_encoder=[512, 128, 32], reduced_channels_decoder=[128, 32], n_classes=10, hidden_size=64):
        super().__init__()
        self.feature_extractor = ConvFeatureExtractor(pad_value, reduced_channels_encoder, reduced_channels_decoder)
        
        self.source_finder = nn.Sequential(
            nn.Linear(n_classes, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False),
            nn.BatchNorm1d(1),
        )
        # self.target_finder = nn.Sequential(
        #     nn.Linear(n_classes, hidden_size, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, n_classes, bias=False),
        #     nn.BatchNorm1d(n_classes),
        # )

    def forward(self, x):
        N, C, H, W = x.shape

        x = self.feature_extractor(x) # [N, C, H, W]
        y_source = x[:, 1:].sum(dim=1).view(N, H, W)

        # x = x.permute(0, 2, 3, 1).reshape(H*W, C)
        # y_source = self.source_finder(x).view(H, W)
        # y_target = self.target_finder(x).view(H, W, C).permute(2, 0, 1) # [C, H, W]

        return y_source, None

    def to(self, *args, **kwargs):
        self.feature_extractor = self.feature_extractor.to(*args, **kwargs)
        return super().to(*args, **kwargs)
