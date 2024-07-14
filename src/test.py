import torch
import torch.nn.functional as F
import os

import warnings
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from rich.traceback import install
from rich import print
install()

from arc_prize import (
    get_model_class,
    DataConfig, 
    TrainConfig, 
    ShapeStableSolverConfig,
)
from data import ARCDataset
from utils.visualize import plot_input_predicted_answer
from arc_prize.preprocess import reconstruct_t_from_one_hot


def _test(config, model, dataset, device):

    for inputs, outputs in dataset:
        for input, output in zip(inputs, outputs):
            source_one_hot, target_one_hot = output
            x = input.to(device)
            source_one_hot = source_one_hot.to(device)
            target_one_hot = target_one_hot.to(device)
            print(input.shape, source_one_hot.shape, target_one_hot.shape)

            y = model(x)
            print(y)
            y_prob = F.sigmoid(y)
            # y_prob = F.softmax(y.view(-1), dim=0).view(y.size())
            print((y_prob*100).int())
            y_pred = torch.round(y_prob)
            
            x_origin = torch.argmax(x, dim=0).long()
            source = reconstruct_t_from_one_hot(x_origin, target_one_hot) 

            # visualize
            plot_input_predicted_answer(x_origin.detach().cpu(), y_pred.detach().cpu(), source.detach().cpu())


def test(config):
    hparams_data = OmegaConf.to_container(config.data.params, resolve=True)
    hparams_model = OmegaConf.to_container(config.model.params, resolve=True)
    del hparams_data['batch_size']

    hparams_shared = {
    }

    model_class = get_model_class(config.model.name)
    model = model_class(**hparams_model, **hparams_shared)
    print(OmegaConf.to_yaml(config))
    
    if config.model_path.endswith('.pth'):
        state_dict = torch.load(config.model_path)
    elif config.model_path.endswith('.ckpt'):
        state_dict = torch.load(config.model_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    
    # Reading files
    base_path = './data/arc-prize-2024/'
    challenges = base_path + 'arc-agi_training_challenges.json'
    solutions = base_path + 'arc-agi_training_solutions.json'

    dataset = ARCDataset(challenges, solutions, **hparams_data)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_printoptions(sci_mode=False, precision=1)

    with torch.no_grad():
        _test(config, model, dataset, device)


cs = ConfigStore.instance()
cs.store(group="data", name="base_data", node=DataConfig, package="data")
cs.store(group="train", name="base_train", node=TrainConfig, package="train")
cs.store(group="model", name="base_ShapeStableSolver", node=ShapeStableSolverConfig, package="arc_prize")


@hydra.main(config_path=os.path.join('..', "configs"), config_name="test", version_base=None)
def main(config: DictConfig) -> None:
    # warnings.filterwarnings('ignore')
    test(config)


if __name__ == '__main__':
    main()