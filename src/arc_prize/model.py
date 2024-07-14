import torch
from torch import nn

from arc_prize.model_custom import Conv2dFixedKernel
from utils.visualize import plot_kernels_and_outputs


class Conv2dEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2dFixedKernel(1, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.linear_layers = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64, bias=False),
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            nn.Linear(16, 1, bias=False),
            nn.ReLU(),
        )
        
    def forward(self, x, **kwargs):
        N, H, W = x.shape[0], x.shape[2], x.shape[3]
        x = self.conv(x)

        x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
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
        # self.decoder = nn.TransformerDecoder
        
    def forward(self, x):
        x_origin = torch.argmax(x, dim=0).float()
        x = torch.cat([self.encoder(x_single_color.view(1, 1, *x_single_color.shape), x_origin=x_origin, color_idx=i) for i, x_single_color in enumerate(x)])
        x = x.sum(dim=0)
        # x = self.decoder(x)
        return x
    
    def to(self, *args, **kwargs):
        self.encoder = self.encoder.to(*args, **kwargs)
        return super().to(*args, **kwargs)


if __name__ == '__main__':
    x = torch.randn(1, 1, 10, 10)
    model = ShapeStableSolver((5, 5))
    
    y = model(x)
    print(y.shape)