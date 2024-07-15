
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os

import warnings
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from rich import print
from rich.traceback import install
install()

from arc_prize import (
    get_model_class,
    DataConfig,
    TrainConfig,
    ShapeStableSolverConfig,
)
from data import ARCDataModule
from utils.lightning_custom import RichProgressBarCustom


def train(config: DictConfig):
    hparams_data = OmegaConf.to_container(config.data.params, resolve=True)
    hparams_model = OmegaConf.to_container(config.model.params, resolve=True)
    hparams_train = OmegaConf.to_container(config.train.params, resolve=True)
    max_epochs = hparams_train.pop("epoch", None)

    hparams_shared = {
    }

    datamodule = ARCDataModule(**hparams_data)

    model_class = get_model_class(config.model.name)
    model = model_class(**hparams_model, **hparams_shared, **hparams_train)
    print(OmegaConf.to_yaml(config))
    print(model)

    # Initialize a trainer
    logger = TensorBoardLogger("./src/lightning_logs/", name=model.__class__.__name__)
    logger.log_hyperparams(params={
        'model': hparams_model,
        'train': hparams_train,
        'data': hparams_data,
        'model_details': model.__str__(),
    })
    
    trainer = Trainer(
        max_epochs=max_epochs, 
        logger=logger, 
        log_every_n_steps=1, 
        accelerator='mps' if torch.backends.mps.is_available() else None,
        callbacks=[RichProgressBarCustom()]
    )
    
    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # Save the model to disk (optional)
    os.makedirs('./output', exist_ok=True)
    model_path = './output/model_{}.pth'.format(model.model.__class__.__name__)
    torch.save(model.state_dict(), model_path)
    print('Seed used', torch.seed())
    print('Model saved to:', model_path)


cs = ConfigStore.instance()
cs.store(group="data", name="base_data", node=DataConfig, package="data")
cs.store(group="train", name="base_train", node=TrainConfig, package="train")
cs.store(group="model", name="base_ShapeStableSolver", node=ShapeStableSolverConfig, package="model")


@hydra.main(config_path=os.path.join('..', "configs"), config_name="train", version_base=None)
def main(config: DictConfig) -> None:
    # warnings.filterwarnings('ignore')
    train(config)


if __name__ == '__main__':
    main()
