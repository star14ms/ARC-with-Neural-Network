import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import BatchesProcessedColumn, CustomBarColumn, CustomTimeColumn, ProcessingSpeedColumn, MetricsTextColumn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.trainer.trainer import Trainer, log

from typing_extensions import override
from typing import Any, Optional, Union

from lightning_fabric.utilities.apply_func import convert_tensors_to_scalars
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import (
    _EVALUATE_OUTPUT,
    _PREDICT_OUTPUT,
    EVAL_DATALOADERS,
)
from rich.progress import TaskID, TextColumn
from rich import print


class RichProgressBarCustom(RichProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.theme.progress_bar = 'bar.complete'
        # self.theme.progress_bar_finished = 'bar.finished'
        # self.theme.progress_bar_pulse = 'bar.pulse'

    def _get_train_description(self, current_epoch: int) -> str:
        train_description = f" Task "
        if self._trainer.fit_loop.max_batches is not None:
            train_description += '{:>5}'.format(f"1/{self._trainer.fit_loop.max_batches}")
        if len(self.validation_description) > len(train_description):
            # Padding is required to avoid flickering due of uneven lengths of "Epoch X"
            # and "Validation" Bar description
            train_description = f"{train_description:{len(self.validation_description)}}"
        return train_description
    
    @override
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_start(trainer, pl_module)
        max_epochs = self._trainer.max_epochs
        
        if max_epochs != 1:
            self.train_progress_bar0_id = self.progress.add_task(description=f'Epoch 1/{max_epochs}', total=max_epochs)

    def _update(self, progress_bar_id: Optional["TaskID"], current: float, completed: float | None = None, visible: bool = True, description: str | None = None) -> None:
        if self.progress is not None and self.is_enabled:
            assert progress_bar_id is not None
            task = filter(lambda task: task.id == progress_bar_id, self.progress.tasks).__next__()
            total = task.total
            assert total is not None
            if current is not None and not self._should_update(current, total):
                return

            if description is None:
                description = task.description

            if completed is None:
                leftover = current % self.refresh_rate
                advance = leftover if (current == total and leftover != 0) else self.refresh_rate
                self.progress.update(progress_bar_id, advance=advance, visible=visible, description=description)
            else:
                self.progress.update(progress_bar_id, completed=completed, visible=visible, description=description)

            self.refresh()

    @override
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        train_description = ' Task {:>5}'.format("{}/{}".format(batch_idx + 1, self._trainer.fit_loop.max_batches))
        self._update(self.train_progress_bar_id, batch_idx + 1, description=train_description)
        self._update_metrics(trainer, pl_module)
        self.refresh()

    @override
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._trainer.max_epochs != 1:
            current_epoch = self._trainer.current_epoch
            max_epochs = self._trainer.max_epochs
            new_description = 'Epoch {}/{}'.format(current_epoch+1, max_epochs)
            self.progress.update(self.train_progress_bar0_id, advance=1, description=new_description)
            self._update_metrics(trainer, pl_module)
            self.refresh()

    def configure_columns(self, trainer: "pl.Trainer") -> list:
        return [
            TextColumn("[progress.description]{task.description}"),
            CustomBarColumn(
                complete_style=self.theme.progress_bar,
                finished_style=self.theme.progress_bar_finished,
                pulse_style=self.theme.progress_bar_pulse,
            ),
            BatchesProcessedColumn(style=self.theme.batch_progress),
            CustomTimeColumn(style=self.theme.time),
            ProcessingSpeedColumn(style=self.theme.processing_speed),
        ]

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


class TrainerCustom(Trainer):
    def _test_impl(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        ckpt_path: Optional[_PATH] = None,
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ) -> Optional[Union[_PREDICT_OUTPUT, _EVALUATE_OUTPUT]]:
        # --------------------
        # SETUP HOOK
        # --------------------
        log.debug(f"{self.__class__.__name__}: trainer test stage")

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(dataloaders, LightningDataModule):
            datamodule = dataloaders
            dataloaders = None
        # If you supply a datamodule you can't supply test_dataloaders
        if dataloaders is not None and datamodule:
            raise MisconfigurationException("You cannot pass both `trainer.test(dataloaders=..., datamodule=...)`")

        if model is None:
            model = self.lightning_module
            model_provided = False
        else:
            model_provided = True

        self.test_loop.verbose = verbose

        # links data to the trainer
        self._data_connector.attach_data(model, train_dataloaders=datamodule.test_dataloader()) ### Customized
        self.state.fn = TrainerFn.FITTING
        self.training = True

        assert self.state.fn is not None
        ckpt_path = self._checkpoint_connector._select_ckpt_path(
            self.state.fn, ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
        )
        results = self._run(model, ckpt_path=ckpt_path)
        # remove the tensors from the test results
        results = convert_tensors_to_scalars(results)

        assert self.state.stopped
        self.testing = False

        return results
