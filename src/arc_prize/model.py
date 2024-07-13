import torch
from torch import nn

from model_custom import Conv2dFixedKernel


class ARCSameShapeConv(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv = Conv2dFixedKernel(1, (3, 3), padding=1, update=False, bias=False)
        self.linear = nn.Linear(512*input_size[0]*input_size[1], 1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten()
        x = self.linear(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 1, 10, 10)
    model = ARCSameShapeConv((5, 5))
    
    y = model(x)
    print(y.shape)