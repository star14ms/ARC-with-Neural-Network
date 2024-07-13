import torch
from torch import nn
from itertools import product
from torch.nn.modules.conv import _size_2_t, Union


class Conv2dFixedKernel(nn.Conv2d):
    def __init__(
        self, 
        in_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        weight=None, 
        update=False
    ):
        weight, out_channels = self.generate_all_possible_NxM_kernels(kernel_size) if weight is None else (weight, weight.shape[0])
        super().__init__(
            in_channels, out_channels, 
            kernel_size, stride, padding, dilation, groups, bias, padding_mode, 
            device, dtype
        )

        del self._parameters['weight']
        
        if update:
            param = nn.Parameter(weight)
            self.register_parameter('weight', param) # Replace the weight tensor
        else:
            self.weight = weight # Stable weight (cannot be updateed)
        
    @staticmethod
    def generate_all_possible_NxM_kernels(kernel_size=(3, 3)):
        '''Generate all possible n x m kernels'''
        weight_values = [0, 1]
        repeat = kernel_size[0] * kernel_size[1]
        weight_custom = torch.cat(
            [torch.tensor(kernel, dtype=torch.float32).reshape(1, 1, kernel_size[0], kernel_size[1]) for kernel in product(weight_values, repeat=repeat)]
        , dim=0)

        out_channels = len(weight_values) ** repeat
        
        return weight_custom, out_channels


def test_backward(model, x):
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    opt.zero_grad()
    y = model(x)
    loss = loss_fn(y, torch.ones_like(y))
    loss.backward()
    opt.step()


if __name__ == '__main__':
    class SampleModel(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.conv = Conv2dFixedKernel(1, (3, 3), padding=1, update=False, bias=False)
            self.linear = nn.Linear(512*input_size[0]*input_size[1], 1, bias=False)

        def forward(self, x):
            x = self.conv(x)
            x = x.flatten()
            x = self.linear(x)

            return x

    input_size = (5, 5)
    model = SampleModel(input_size=input_size)
    x = torch.randn(1, 1, *input_size)
    test_backward(model, x)
