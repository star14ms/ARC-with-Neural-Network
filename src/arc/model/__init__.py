from dataclasses import dataclass
from typing import List
from typing import List


def get_model_class(model_name: str):
    from arc.model.fill.lightning import (
        FillerKeepInputL,
        FillerKeepInputIgnoreColorL,
    )
    from arc.model.substitute.lightning import (
        PixelEachSubstitutorL,
    )

    model_classes = [
        FillerKeepInputL,
        FillerKeepInputIgnoreColorL,
        PixelEachSubstitutorL,
    ]

    for model_class in model_classes:
        model_class_name = model_class.__name__
        if model_name == model_class_name or model_name + 'L' == model_class_name:
            return model_class
    else:
        raise ValueError(f"Model name {model_name} not found")


@dataclass
class DataConfig:
    name: str = 'default'
    cold_value: int = -1
    ignore_color: bool = False

@dataclass
class TrainConfig:
    max_epochs: int = 1
    batch_size_max: int = 8
    augment_data: bool = True
    lr: float = 0.01
    save_dir: str = "./output/"
    ckpt_path: str | None = None

@dataclass
class TestConfig:
    model_path: str = "./output/model_FillerKeepInput.ckpt"
    verbose_single: bool = False
    augment_data: bool = False # Rotate and Flip

@dataclass
class ModelConfig:
    name: str = 'FillerKeepInput'
    pass

@dataclass
class FillerKeepInputConfig:
    reduced_channels_encoder: List[int] = (512, 32)
    reduced_channels_decoder: List[int] = (128, 32)
    pad_value: int = -1
    d_conv_feature: int = 16
    d_class_feature: int = 32

@dataclass
class FillerKeepInputIgnoreColorConfig:
    reduced_channels_encoder: List[int] = (512, 32)
    reduced_channels_decoder: List[int] = (128, 32)

@dataclass
class PixelEachSubstitutorConfig:
    pad_size: int = 1
    max_width: int = 3
    max_height: int = 3
    dims_reduced: List[int] = (9,)
    dims_decoded: List[int] = (1,)
    pad_class_initial: int = 0
    pad_n_head: int | None = None
    pad_dim_feedforward: int = 1
    pad_num_layers: int = 4
    L_n_head: int | None = None
    L_dim_feedforward: int = 1
    L_num_layers: int = 6
    C_n_head: int | None = None
    C_dim_feedforward: int = 1
    C_num_layers: int = 1
    dropout: float = 0.1
    num_classes: int = 10

    max_trial: int = 1
    hyperparams_for_each_trial: List[dict] = tuple()
    max_epochs_for_each_task: int = 300
    train_loss_threshold_to_stop: float = 0.01
