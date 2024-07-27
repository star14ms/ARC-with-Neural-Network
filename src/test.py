import torch
import torch.nn.functional as F
from pytorch_lightning.trainer.connectors.accelerator_connector import _AcceleratorConnector
import os

import warnings
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from rich import print
from rich.traceback import install
install()

from arc_prize.model import (
    get_model_class,
    DataConfig, 
    TestConfig, 
    FillerKeepInputConfig,
    FillerKeepInputIgnoreColorConfig
)
from arc_prize.utils.visualize import print_image_with_probs, plot_xyt, plot_xyts, visualize_image_using_emoji
from arc_prize.preprocess import reconstruct_t_from_one_hot
from data import ARCDataset


def _test(config, model, dataset_train, device):
    n_recurrance_feature_extraction = config.test.params.get('n_recurrance_feature_extraction', None)
    kwargs = dict(n_recurrance_feature_extraction=n_recurrance_feature_extraction) if n_recurrance_feature_extraction else {}

    for i, (inputs, outputs, inputs_test, outputs_test, key) in enumerate(dataset_train):
        task_result = []
        len_train = len(inputs)
        
        inputs += inputs_test
        outputs += outputs_test
        
        for j, (x, t) in enumerate(zip(inputs, outputs)):
            x = x.to(device).unsqueeze(0)
            t = t.to(device).unsqueeze(0)
            # print(x.shape, t.shape)

            y = model(x, **kwargs)
            x_origin = torch.argmax(x, dim=1).long() # [H, W]

            if config.data.params.ignore_color:
                y_prob = F.sigmoid(y).squeeze(0)
                y_origin = torch.where(y_prob > 0.5, 1, 0) # [C, H, W]
                t0 = torch.zeros_like(x).squeeze(0)
                t0[4:5] += t
                t_origin = reconstruct_t_from_one_hot(x_origin, t0)

                # xy_construct = x_origin.squeeze(0).repeat(1, 1)
                # xy_construct[torch.where(y_origin == 1)] = 4
                # y_origin = xy_construct
            else:
                # y_prob = F.softmax(y, dim=1)
                y_origin = torch.argmax(y, dim=1).long() # [H, W]
                t_origin = torch.argmax(t, dim=1).long()
                # visualize_image_using_emoji(x[0], y[0], t[0])

            correct_ratio = (y_origin == t_origin).sum().float() / t_origin.numel()
            n_pixels_wrong = (y_origin != t_origin).sum().int()

            print('Task: {} | {:>5} {} | {:>6.2f}% correct | {} Pixels Wrong'.format(
                key, 
                'train' if j < len_train else 'test', 
                j+1 if j < len_train else j-len_train+1, 
                correct_ratio*100, 
                n_pixels_wrong
            ))
            
            # visualize
            if config.test.params.verbose_single:
                plot_xyt(x_origin, y_origin, t_origin)
            else:
                task_result.append((x_origin, y_origin, t_origin))

        if not config.test.params.verbose_single:
            plot_xyts(task_result, title_prefix=key)


def test(config, model=None):
    hparams_data = OmegaConf.to_container(config.data.params, resolve=True)
    hparams_model = OmegaConf.to_container(config.model.params, resolve=True)
    hparmas_test = OmegaConf.to_container(config.test.params, resolve=True)
    base_path = hparams_data.pop('base_path')
    model_path = hparmas_test.pop('model_path')

    if model is None or isinstance(model, type):
        model_class = get_model_class(config.model.name if model is None else model.__name__)
        model = model_class(**hparams_model, model=model if isinstance(model, type) else None)
        print(OmegaConf.to_yaml(config))

        if model_path.endswith('.pth'):
            state_dict = torch.load(model_path)
        elif model_path.endswith('.ckpt'):
            state_dict = torch.load(model_path)['state_dict']
        model.load_state_dict(state_dict)

    device = torch.device(_AcceleratorConnector._choose_auto_accelerator())
    model.eval()
    model.to(device)

    # Reading files
    challenges = base_path + 'arc-agi_training_challenges.json'
    solutions = base_path + 'arc-agi_training_solutions.json'

    dataset = ARCDataset(challenges, solutions, **hparams_data)
    torch.set_printoptions(sci_mode=False, precision=1)

    with torch.no_grad():
        _test(config, model, dataset, device)


cs = ConfigStore.instance()
cs.store(group="data", name="base_data", node=DataConfig, package="data")
cs.store(group="test", name="base_test", node=TestConfig, package="test")
cs.store(group="model", name="base_FillerKeepInput", node=FillerKeepInputConfig, package="arc_prize")
cs.store(group="model", name="base_FillerKeepInputIgnoreColor", node=FillerKeepInputIgnoreColorConfig, package="arc_prize")


@hydra.main(config_path=os.path.join('..', "configs"), config_name="test", version_base=None)
def main(config: DictConfig) -> None:
    # warnings.filterwarnings('ignore')
    test(config)


if __name__ == '__main__':
    main()
