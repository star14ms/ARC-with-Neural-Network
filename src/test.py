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
from utils.visualize import print_image_with_probs, plot_xyt, plot_xyts
from arc_prize.preprocess import reconstruct_t_from_one_hot


def _test(config, model, dataset_train, dataset_test, device):
    for i, (inputs, outputs) in enumerate(dataset_train):
        task_result = []
        key = dataset_train.task_key(i)
        
        inputs += dataset_test[i][0]
        outputs += dataset_test[i][1]
        
        for j, (input, output) in enumerate(zip(inputs, outputs)):
            source_one_hot, target_one_hot = output
            x = input.to(device)
            source_one_hot = source_one_hot.to(device)
            target_one_hot = target_one_hot.to(device)
            # print(input.shape, source_one_hot.shape, target_one_hot.shape)

            y_source = model(x.unsqueeze(0))
            y_source_prob = F.sigmoid(y_source)
            y_pred = torch.where(y_source_prob > 0.5, 1, -1).squeeze(0) # [C, H, W]

            # print(y_source - y_source.min())
            print('Task: {}, index: {}'.format(key, j+1))
            print_image_with_probs(y_source_prob.squeeze(0).detach().cpu(), y_pred.squeeze(0).detach().cpu(), source_one_hot.squeeze(0).detach().cpu())

            x_origin = torch.argmax(x, dim=0).long() # [H, W]
            t = reconstruct_t_from_one_hot(x_origin, target_one_hot) 

            # overwrite x_origin with the predicted values
            # H, W = target_one_hot.shape[1], target_one_hot.shape[2]
            # y_pred = x_origin.clone().view(-1)
            # for i, (x, y_pixels) in enumerate(zip(x_origin.view(-1), y_pred.permute(1, 2, 0).view(H*W, -1))):
            #     if y_pixels.sum() == 0:
            #         continue
            #     y_pred[i] = y_pixels.argmax()
            # y_pred = y_pred.view(H, W)
            
            # visualize
            if config.verbose_single:
                plot_xyt(x_origin.detach().cpu(), y_pred.detach().cpu(), t.detach().cpu())
                continue

            task_result.append((x_origin.detach().cpu(), y_pred.detach().cpu(), t.detach().cpu()))

        plot_xyts(task_result, title_prefix=key)


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

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    # Reading files
    base_path = './data/arc-prize-2024/'
    challenges = base_path + 'arc-agi_training_challenges.json'
    solutions = base_path + 'arc-agi_training_solutions.json'

    dataset_train = ARCDataset(challenges, solutions, train=True, **hparams_data)
    dataset_test = ARCDataset(challenges, solutions, train=False, **hparams_data)
    torch.set_printoptions(sci_mode=False, precision=1)

    with torch.no_grad():
        _test(config, model, dataset_train, dataset_test, device)


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
