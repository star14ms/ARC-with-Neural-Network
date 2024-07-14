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
        bias: bool = False,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        weight=None, 
        update=False
    ):
        if weight is None:
            weight, out_channels, biases = self.generate_all_possible_NxM_kernels(kernel_size, device=device, dtype=dtype)
        else:
            out_channels = weight.shape[0]
            biases = self.generate_biases(weight)

        super().__init__(
            in_channels, out_channels, 
            kernel_size, stride, padding, dilation, groups, bias, padding_mode, 
            device, dtype
        )

        # Remove the default weight parameter
        del self._parameters['weight']
        del self._parameters['bias']
        
        if update:
            # Register weight as a parameter for updates
            param = nn.Parameter(weight)
            self.register_parameter('weight', param)
            
            self.bias = nn.Parameter(biases, requires_grad=update)
            self.register_parameter('bias', self.bias)
        else:
            # Set fixed weight
            self.weight = weight
            self.bias = biases

    def to(self, *args, **kwargs):
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @staticmethod
    def generate_all_possible_NxM_kernels(kernel_size=(3, 3), device=None, dtype=None):
        '''Generate all possible n x m kernels with values 0 and 1'''
        weight_values = [0.0, 1.0]
        repeat = kernel_size[0] * kernel_size[1]
        # Generate all possible combinations of kernel values
        weight_custom = torch.cat(
            [torch.tensor(kernel, dtype=dtype, device=device).reshape(1, 1, kernel_size[0], kernel_size[1]) 
             for kernel in product(weight_values, repeat=repeat)], dim=0)

        out_channels = len(weight_values) ** repeat
        
        # Generate corresponding biases
        biases = -(torch.sum(weight_custom, dim=(2, 3)) - 1).reshape(out_channels).to(device=device, dtype=dtype)
        
        return weight_custom, out_channels, biases

    @staticmethod
    def generate_biases(weight):
        '''Generate biases based on weight patterns'''
        return -(torch.sum(weight, dim=(2, 3)) - 1).reshape(weight.shape[0])


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
