import torch
from torch import nn
import torch.nn.functional as F
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
        bias=False,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        weight=None, 
        requires_grad=False,
        generate_all_possible_binary_kernels=False,
    ):
        if generate_all_possible_binary_kernels:
            weight, out_channels, biases = self.generate_all_possible_NxM_kernels(kernel_size, device=device, dtype=dtype)
        else:
            if weight is None:
                raise ValueError('Weight must be provided if generate_all_possible_binary_kernels is False')
            else:
                out_channels = weight.shape[0]

            if bias is not False:
                biases = self.generate_biases(weight) if not isinstance(bias, torch.Tensor) else bias

        super().__init__(
            in_channels, out_channels, 
            kernel_size, stride, padding, dilation, groups, bias, padding_mode, 
            device, dtype
        )

        # Remove the default weight parameter
        del self._parameters['weight']
        self.weight = nn.Parameter(weight, requires_grad=requires_grad)
        self.register_parameter('weight', self.weight)

        if bias is not False:
            del self._parameters['bias']
            self.bias = nn.Parameter(biases, requires_grad=requires_grad)
            self.register_parameter('bias', self.bias)

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


class Conv2dEncoderLayer(nn.Module):
    def __init__(self, in_channels=1, reduced_channels=[512, 32], fixed_kernel=False, out_one_channel=False, kernel_size=(3, 3), stride=1, padding=1, pad_value=-1):
        super().__init__()
        self.padding = padding
        self.pad_value = pad_value
        self.fixed_kernel = fixed_kernel

        if fixed_kernel:
            self.conv = Conv2dFixedKernel(in_channels, kernel_size=kernel_size, stride=stride, padding=0, generate_all_possible_binary_kernels=True)
        else:
            self.conv = nn.Conv2d(in_channels, reduced_channels[0], kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.activation = nn.ReLU()

        self.linear_layers = nn.Sequential()
        for i in range(len(reduced_channels)-1):
            self.linear_layers.add_module(f'linear_{i}', nn.Linear(reduced_channels[i], reduced_channels[i+1], bias=False))
            self.linear_layers.add_module(f'relu_{i}', nn.ReLU())

        if out_one_channel:
            self.linear_layers.add_module('out', nn.Linear(reduced_channels[-1], 1, bias=False))

        self.norm = nn.InstanceNorm2d(reduced_channels[-1] if not out_one_channel else 1)

    def forward(self, x):
        N, H, W = x.shape[0], x.shape[2], x.shape[3]
        if self.fixed_kernel:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=self.pad_value)
        x = self.activation(self.conv(x)) # [N, C, H, W]
        x = x.permute(0, 2, 3, 1).reshape(N*H*W, -1) # [N*H*W, C]
        x = self.linear_layers(x)
        x = x.view(N, H, W, -1).permute(0, 3, 1, 2)
        x = self.norm(x)

        return x


def test_backward(model, x):
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    opt.zero_grad()
    y = model(x)
    loss = loss_fn(y, torch.ones_like(y))
    print(loss)
    loss.backward()
    opt.step()


if __name__ == '__main__':
    from rich import print

    class SampleModel(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.conv = Conv2dFixedKernel(1, (3, 3), padding=1)
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
