import sys
import os

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
import datetime
from rich import print
from rich.traceback import install
install()

import arc.model as config
from arc.model import get_model_class
from data import ARCDataModule

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from torchview_custom.torchview import draw_graphs


def visualize_model(config: DictConfig, model=None, filter_funcs=None, **kwargs_data):
    hparams_data = OmegaConf.to_container(config.data.params, resolve=True)
    hparams_model = OmegaConf.to_container(config.model.params, resolve=True)
    hparams_train = OmegaConf.to_container(config.train.params, resolve=True)
    hparams_lightning = OmegaConf.to_container(config.lightning.params, resolve=True)
    batch_size_max, augment_data, lr, save_dir = \
        hparams_train.get('batch_size_max'), \
        hparams_train.get('augment_data'), \
        hparams_train.get('lr'), \
        hparams_train.get('save_dir'), \

    hparams_data['batch_size_max'] = batch_size_max
    hparams_data['augment_data'] = augment_data

    # Hydra Output Directory
    if os.path.exists('outputs'):
        save_dir_parent = os.path.join('outputs', sorted(os.listdir('outputs/'))[-1])
        save_dir = os.path.join(save_dir_parent, sorted(os.listdir(save_dir_parent))[-1])
    elif save_dir is not None:
        now = datetime.datetime.now()
        save_dir = os.path.join(save_dir, now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))
        os.makedirs(save_dir, exist_ok=True)

    if model is None or isinstance(model, type):
        model = model if isinstance(model, type) else None
        model_class = get_model_class(config.model.name if model is None else model.__name__)
        model = model_class(lr=lr, model=model, save_dir=save_dir, **hparams_lightning, **hparams_model)
        print(OmegaConf.to_yaml(config))
        print(model)

    datamodule = ARCDataModule(filter_funcs=filter_funcs, **hparams_data, **kwargs_data)
    dataloader = datamodule.train_dataloader()
    # dataloader = datamodule.test_dataloader()
    # dataloader = datamodule.val_dataloader()

    for (batches_train, batches_test, task_id) in dataloader:
        for (x, t) in batches_train:
            draw_graphs(model, (x,), input_names=['Input'], output_names=['Output'], directory='./model_viz/', hide_module_functions=True, print_code_path=False)
            break
        break


cs = ConfigStore.instance()
cs.store(group="data", name="base_data", node=config.DataConfig, package="data")
cs.store(group="train", name="base_train", node=config.TrainConfig, package="train")
cs.store(group="test", name="base_test", node=config.TestConfig, package="test")

cs.store(group="lightning", name="base_lightning_FillerKeepInput", node=config.FillerKeepInputLightningConfig, package="lightning")
cs.store(group="lightning", name="base_lightning_FillerKeepInputIgnoreColor", node=config.FillerKeepInputIgnoreColorLightningConfig, package="lightning")
cs.store(group="lightning", name="base_lightning_PixelEachSubstitutor", node=config.PixelEachSubstitutorLightningConfig, package="lightning")
cs.store(group="lightning", name="base_lightning_PixelEachSubstitutorRepeat", node=config.PixelEachSubstitutorRepeatLightningConfig, package="lightning")

cs.store(group="model", name="base_model_FillerKeepInput", node=config.FillerKeepInputConfig, package="model")
cs.store(group="model", name="base_model_FillerKeepInputIgnoreColor", node=config.FillerKeepInputIgnoreColorConfig, package="model")
cs.store(group="model", name="base_model_PixelEachSubstitutor", node=config.PixelEachSubstitutorConfig, package="model")
cs.store(group="model", name="base_model_PixelEachSubstitutorRepeat", node=config.PixelEachSubstitutorRepeatConfig, package="model")


@hydra.main(config_path=os.path.join('..', "configs"), config_name="train", version_base=None)
def main(config: DictConfig) -> None:
    visualize_model(config)


if __name__ == '__main__':
    main()
