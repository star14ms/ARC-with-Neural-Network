
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os

import warnings
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig, open_dict
from rich import print
from rich.traceback import install
install()

from arc_prize.model import (
    get_model_class,
    DataConfig,
    TrainConfig,
    TestConfig,
    FillerKeepInputConfig,
    FillerKeepInputIgnoreColorConfig
)
from arc_prize.utils.lightning_custom import RichProgressBarCustom
from data import ARCDataModule
from test import test as test_fn


def train(config: DictConfig, model=None, test=False, return_model=False):
    hparams_data = OmegaConf.to_container(config.data.params, resolve=True)
    hparams_model = OmegaConf.to_container(config.model.params, resolve=True)
    hparams_train = OmegaConf.to_container(config.train.params, resolve=True)
    max_epochs, batch_size_max, lr, save_dir, ckpt_path = \
        hparams_train.get('max_epochs', None), \
        hparams_train.get('batch_size_max', None), \
        hparams_train.get('lr', None), \
        hparams_train.get('save_dir', None), \
        hparams_train.get('ckpt_path', None)

    if model is None or isinstance(model, type):
        model = model if isinstance(model, type) else None
        model_class = get_model_class(config.model.name if model is None else model.__name__)
        model = model_class(lr=lr, model=model, **hparams_model)
        print(OmegaConf.to_yaml(config))
        print(model)

    # Initialize a trainer
    logger = TensorBoardLogger("./src/lightning_logs/", name=model.__class__.__name__)
    logger.log_hyperparams(params={
        'model': hparams_model,
        'train': hparams_train,
        'data': hparams_data,
        'model_details': model.__str__(),
        'seed': torch.seed(),
    })

    trainer = Trainer(
        max_epochs=max_epochs, 
        logger=logger, 
        log_every_n_steps=1, 
        callbacks=[
            RichProgressBarCustom(),
            # ModelCheckpoint(every_n_epochs=50, save_top_k=3, monitor='epoch', mode='max')
        ]
    )
    datamodule = ARCDataModule(local_world_size=trainer.num_devices, batch_size_max=batch_size_max, **hparams_data)

    # Train the model
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Save the model to disk (optional)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '{}.ckpt'.format(model.model.__class__.__name__))
    trainer.save_checkpoint(save_path)
    print('Seed used', torch.seed())
    print('Model saved to:', save_path)
    
    if test:
        with open_dict(config):
            config.data.params.augment_data = False

        test_fn(config, model)

    if return_model:
        return model


cs = ConfigStore.instance()
cs.store(group="data", name="base_data", node=DataConfig, package="data")
cs.store(group="train", name="base_train", node=TrainConfig, package="train")
cs.store(group="test", name="base_test", node=TestConfig, package="test")
cs.store(group="model", name="base_FillerKeepInput", node=FillerKeepInputConfig, package="model")
cs.store(group="model", name="base_FillerKeepInputIgnoreColor", node=FillerKeepInputIgnoreColorConfig, package="model")


@hydra.main(config_path=os.path.join('..', "configs"), config_name="train", version_base=None)
def main(config: DictConfig) -> None:
    # warnings.filterwarnings('ignore')
    train(config, test=False)


if __name__ == '__main__':
    main()
