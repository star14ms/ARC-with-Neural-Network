import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from rich import print

from arc_prize.model import ShapeStableSolver
from arc_prize.model_ignore_color import ShapeStableSolverIgnoreColor
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
    def __init__(self, lr=0.001, ignore_color=False, model=None, *args, **kwargs):
        super().__init__(lr, *args, **kwargs)
       
        if model is not None:
            self.model = model(*args, **kwargs)
        elif not ignore_color:
            self.model = ShapeStableSolver(*args, **kwargs)
        else:
            self.model = ShapeStableSolverIgnoreColor(*args, **kwargs)

        self.loss_fn_source = nn.BCEWithLogitsLoss()
        
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model.to(device)
    
    def forward(self, inputs):
        # In Lightning, forward defines the prediction/inference actions
        return self.model(inputs)

    def training_step(self, batches):
        total_loss = 0
        total = len(batches)
        opt = self.optimizers()

        # Process each batch
        for i, (x, t) in enumerate(batches):
            # forward + backward + optimize
            y = self.model(x)
            loss = self.loss_fn_source(y, t)
            total_loss += loss

            if i == total - 1:
                break

            opt.zero_grad()
            loss.backward()
            opt.step()

        print("Train loss: {:.6f}".format(total_loss))
        self.log('Train loss', total_loss, prog_bar=True)
        return loss
