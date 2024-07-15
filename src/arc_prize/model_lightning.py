import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from rich import print

from arc_prize.model import ShapeStableSolver
from arc_prize.preprocess import one_hot_encode, one_hot_encode_changes, reconstruct_t_from_one_hot


class LightningModuleBase(pl.LightningModule):
    def __init__(self, lr=0.001, *args, **kwargs):
        self.lr = lr
        super().__init__()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self) -> torch.Any:
        return super().train_dataloader()

    def val_dataloader(self) -> torch.Any:
        return super().val_dataloader()

    def test_dataloader(self) -> torch.Any:
        return super().test_dataloader()


class ShapeStableSolverL(LightningModuleBase):
    def __init__(self, lr=0.001, *args, **kwargs):
        super().__init__(lr, *args, **kwargs)

        self.model = ShapeStableSolver(*args, **kwargs)
        self.loss_fn_source = nn.BCEWithLogitsLoss()
        # self.loss_fn_target = nn.BCEWithLogitsLoss()
        
        device = 'mps' if torch.backends.mps.is_available() else None
        self.model.to(device)
    
    def forward(self, inputs):
        # In Lightning, forward defines the prediction/inference actions
        return self.model(inputs)

    def training_step(self, batches):
        total_loss_source = 0
        total_loss_target = 0
        total = len(batches)
        opt = self.optimizers()

        # Process each batch
        for i, (x, t) in enumerate(batches):
            # forward + backward + optimize
            t_source, t_target = t
            y_source, y_target = self.model(x)

            loss_source = self.loss_fn_source(y_source, t_source)
            # loss_target = self.loss_fn_target(y_target, t_target)
            # loss = loss_source + loss_target
            loss = loss_source
            total_loss_source += loss_source
            # total_loss_target += loss_target

            if i == total - 1:
                break

            opt.zero_grad()
            loss.backward()
            opt.step()

        print("Train loss: {:.6f} (source: {:.6f}, target: {:.6f})".format(total_loss_source+total_loss_target, total_loss_source, total_loss_target))
        self.log('Train loss', total_loss_source+total_loss_target)
        self.log('Train loss source', total_loss_source)
        self.log('Train loss target', total_loss_target)
        return loss
