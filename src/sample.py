from data import ARCDataset

from utils.visualize import plot_task, plot_kernels_and_outputs
from arc_prize.model import Conv2dFixedKernel
from arc_prize.preprocess import one_hot_encode, one_hot_encode_changes, reconstruct_t_from_one_hot

from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from constants import COLORS

from arc_prize.model import ShapeStableSolver

    
if __name__ == '__main__':
    from rich import print
    base_path = './data/arc-prize-2024/'
    challenges = base_path + 'arc-agi_training_challenges.json'
    solutions = base_path + 'arc-agi_training_solutions.json'

    dataset_train = ARCDataset(challenges, solutions, train=True)
    dataset_test = ARCDataset(challenges, solutions, train=False)
    # plot_task(dataset_train, dataset_test, 0)


    model = ShapeStableSolver()
    x = dataset_train[0][0][0]
    source_one_hot, target_one_hot = dataset_train[0][1][0]
    source_one_hot = source_one_hot
    target_one_hot = target_one_hot

    loss_fn = nn.CrossEntropyLoss()

    # x_origin = torch.argmax(x, dim=0).float()

    y = model(x)
    y_prob = torch.softmax(y, dim=0)
    loss = loss_fn(y, source_one_hot)
    
    print(loss)
    
    kernels = model.encoder.conv.weight.detach().cpu().numpy()
    