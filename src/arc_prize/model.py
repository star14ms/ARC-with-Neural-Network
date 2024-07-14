import torch
import torch.nn.functional as F
from torch import nn

from arc_prize.model_custom import Conv2dFixedKernel
from utils.visualize import plot_kernels_and_outputs


class Conv2dEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2dFixedKernel(1, kernel_size=(3, 3), stride=1, padding=0)
        self.activation = nn.ReLU()

        self.linear_layers = nn.Sequential(
            nn.Linear(512, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 1, bias=False),
        )
        
    def forward(self, x, **kwargs):
        if not torch.any(x == 1):
            return x.squeeze(1) # .fill_(1)

        N, H, W = x.shape[0], x.shape[2], x.shape[3]
        x = F.pad(x, (1, 1, 1, 1), mode='constant', value=-1)
        x = self.activation(self.conv(x)) # [N, C, H, W]

        x = x.permute(0, 2, 3, 1).reshape(N*H*W, -1) # [N*H*W, C]
        x = self.linear_layers(x)
        x = x.view(N, H, W)
        return x
    
    def to(self, *args, **kwargs):
        self.conv = self.conv.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    
class ShapeStableSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Conv2dEncoder()
        self.norm = nn.BatchNorm2d(1)
        # self.decoder = nn.TransformerDecoder
        
    def forward(self, x):
        x_origin = torch.argmax(x, dim=0).float()
        x = torch.cat([self.encoder(x_single_color.view(1, 1, *x_single_color.shape), x_origin=x_origin, color_idx=i) for i, x_single_color in enumerate(x)])
        x = x[1:].sum(dim=0)
        x = self.norm(x.view(1, 1, *x.shape)).squeeze()
        # x = self.decoder(x)
        return x
    
    def to(self, *args, **kwargs):
        self.encoder = self.encoder.to(*args, **kwargs)
        return super().to(*args, **kwargs)
