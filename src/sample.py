from torch import nn
from rich import print

from data import ARCDataset
from arc_prize.model.fill.keep_input import FillerKeepInput
from arc_prize.constants import get_challenges_solutions_filepath
from arc_prize.utils.visualize import plot_task


if __name__ == '__main__':
    challenges, solutions = get_challenges_solutions_filepath('train')
    dataset_train = ARCDataset(challenges, solutions, train=True)
    dataset_test = ARCDataset(challenges, solutions, train=False)
    # plot_task(dataset_train, dataset_test, 0)

    model = FillerKeepInput()
    print(model)
    x = dataset_train[0][0][0].unsqueeze(0)
    t = dataset_train[0][1][0]

    loss_fn = nn.BCEWithLogitsLoss()

    # x_origin = torch.argmax(x, dim=0).float()

    y_source = model(x)
    loss = loss_fn(y_source[0], t)
    
    print(loss)
    