
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os
import json

import warnings
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
import datetime
from rich import print
from rich.traceback import install
install()

import arc.model as config
from arc.model import get_model_class
from arc.utils.lightning_custom import TrainerCustom, RichProgressBarCustom
from data import ARCDataModule
from test import test as test_fn


def train(config: DictConfig, model=None, filter_funcs=None, test=False, return_model=False, **kwargs_data):
    hparams_data = OmegaConf.to_container(config.data.params, resolve=True)
    hparams_model = OmegaConf.to_container(config.model.params, resolve=True)
    hparams_train = OmegaConf.to_container(config.train.params, resolve=True)
    hparams_lightning = OmegaConf.to_container(config.lightning.params, resolve=True)
    max_epochs, batch_size_max, augment_data, lr, save_dir, ckpt_path = \
        hparams_train.get('max_epochs'), \
        hparams_train.get('batch_size_max'), \
        hparams_train.get('augment_data'), \
        hparams_train.get('lr'), \
        hparams_train.get('save_dir'), \
        hparams_train.get('ckpt_path')

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

    # Initialize a trainer
    logger = TensorBoardLogger("./src/lightning_logs/", name=model.__class__.__name__)
    logger.log_hyperparams(params={
        'model': hparams_model,
        'train': hparams_train,
        'data': hparams_data,
        'model_details': model.__str__(),
        'seed': torch.seed(),
    })

    trainer = TrainerCustom(
        accelerator='cpu',
        max_epochs=max_epochs, 
        logger=logger, 
        log_every_n_steps=1, 
        callbacks=[
            RichProgressBarCustom(),
            # ModelCheckpoint(every_n_epochs=50, save_top_k=3, monitor='epoch', mode='max')
        ]
    )
    datamodule = ARCDataModule(local_world_size=trainer.num_devices, filter_funcs=filter_funcs, **hparams_data, **kwargs_data)

    # Train the model
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    print('Seed used', torch.seed())

    # Save the model to disk (optional)
    # save_path = os.path.join(save_dir, '{}.ckpt'.format(model.model.__class__.__name__))
    # trainer.save_checkpoint(save_path)
    # print('Model saved to:', save_path)

    # Save the test results to disk (optional)
    save_path = os.path.join(save_dir, 'test_results.json')
    with open(save_path, 'w') as f:
        json.dump(model.test_results, f)
    print("Test results saved to: '{}'".format(save_path))

    # Save the submission to disk (optional)
    save_path = os.path.join(save_dir, 'submission.json')
    with open(save_path, 'w') as f: 
        json.dump(model.submission, f)
    print("Submission saved to: '{}'".format(save_path))

    if test:
        test_fn(config, model)

    if return_model:
        return model


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
    # warnings.filterwarnings('ignore')
    train(config, test=False)


if __name__ == '__main__':
    main()
