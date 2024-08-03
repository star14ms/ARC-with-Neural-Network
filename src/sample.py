import torch
import torch.nn as nn
from pytorch_lightning.trainer.connectors.accelerator_connector import _AcceleratorConnector
from rich import print

from data import ARCDataset
from arc.model.fill.keep_input import FillerKeepInput
from arc.model.substitute.pixel_each import PixelEachSubstitutor
from arc.constants import get_challenges_solutions_filepath
from arc.utils.visualize import plot_task


def profile_model(model, inputs):
    from thop import profile
    from rich import print
    import sys
    macs, params = profile(model, inputs=inputs, verbose=False)
    print('모델 생성 완료! (Input {}: MACs: {} M | Params: {} K)'.format(
        sys.getsizeof(x.storage()),
        round(macs/1000/1000, 2), 
        round(params/1000, 2),
    ))


if __name__ == '__main__':
    challenges, solutions = get_challenges_solutions_filepath('train')
    dtype = torch.float16
    device = torch.device(_AcceleratorConnector._choose_auto_accelerator())

    model = PixelEachSubstitutor()
    print(model)
    model.to(device, dtype)

    # Example 1: Measure MACs and Parameters of the model with different input sizes
    max_size = 10
    for i in range(1, max_size+1):
        x = torch.randn(1, 10, i, i, dtype=dtype, device=device)
        profile_model(model, (x, ))

    # # Example 2: Using ARC Dataset
    # dataset = ARCDataset(challenges, solutions)
    # plot_task(dataset, 0, data_category='train')
    # x = dataset[0][0][0].unsqueeze(0).to(device, dtype)
    # t = dataset[0][1][0].unsqueeze(0).to(device, dtype)

    # y = model(x)
    # loss_fn = nn.CrossEntropyLoss()
    # loss = loss_fn(y, t)

    # print(loss)