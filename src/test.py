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
        
        for j, (x, t) in enumerate(zip(inputs, outputs)):
            x = x.to(device)
            t = t.to(device)
            # print(x.shape, t.shape)

            y = model(x.unsqueeze(0))
            y_prob = F.sigmoid(y)
            
            x_origin = torch.argmax(x, dim=0).long() # [H, W]

            if config.data.params.ignore_color:
                y_origin = torch.where(y_prob > 0.5, 1, -1).squeeze(0) # [C, H, W]
                t0 = torch.zeros_like(x)
                t0[4:5] += t
                t = t0
            else:
                y_origin = torch.argmax(y_prob[0], dim=0).long() # [H, W]
                

            t = reconstruct_t_from_one_hot(x_origin, t)
            correct_ratio = (y_origin == t).sum().float() / t.numel()
            
            print('Task: {}, index: {}, correct {}%'.format(key, j+1, correct_ratio*100))
            # print(y - y.min())
            # print_image_with_probs(y_prob.squeeze(0).detach().cpu(), y_pred.squeeze(0).detach().cpu(), t.squeeze(0).detach().cpu())

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
                plot_xyt(x_origin.detach().cpu(), y_origin.detach().cpu(), t.detach().cpu())
                continue

            task_result.append((x_origin.detach().cpu(), y_origin.detach().cpu(), t.detach().cpu()))

        plot_xyts(task_result, title_prefix=key)


def test(config, model=None):
    hparams_data = OmegaConf.to_container(config.data.params, resolve=True)
    hparams_model = OmegaConf.to_container(config.model.params, resolve=True)
    base_path = hparams_data.pop('base_path')
    del hparams_data['batch_size']

    hparams_shared = {
        'ignore_color': hparams_data['ignore_color'],
    }

    model_class = get_model_class(config.model.name)
    model = model_class(**hparams_model, **hparams_shared, model=model)
    # print(OmegaConf.to_yaml(config))
    
    if config.model_path.endswith('.pth'):
        state_dict = torch.load(config.model_path)
    elif config.model_path.endswith('.ckpt'):
        state_dict = torch.load(config.model_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    # Reading files
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
