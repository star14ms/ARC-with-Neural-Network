import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from rich import print

from arc_prize.model import ShapeStableSolver
from arc_prize.model_ignore_color import ShapeStableSolverIgnoreColor
from arc_prize.preprocess import one_hot_encode, one_hot_encode_changes, reconstruct_t_from_one_hot
from utils.visualize import plot_xyt

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
    def __init__(self, lr=0.001, model=None, *args, **kwargs):
        super().__init__(lr, *args, **kwargs)
       
        if model is not None:
            self.model = model(*args, **kwargs)
        else:
            self.model = ShapeStableSolver(*args, **kwargs)

        self.loss_fn_source = nn.BCEWithLogitsLoss()
        
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model.to(device)
    
    def forward(self, inputs):
        # In Lightning, forward defines the prediction/inference actions
        outputs = self.model(inputs)
        return outputs

    def training_step(self, batches):
        total_loss = 0
        total = len(batches)
        opt = self.optimizers()
        n_pixel_wrong_total = 0

        # Process each batch
        for i, (x, t) in enumerate(batches):
            # forward + backward + optimize
            y = self.model(x)
            loss = self.loss_fn_source(y, t)
            total_loss += loss

            with torch.no_grad():
                x_origin = torch.argmax(x[0].detach().cpu(), dim=0).long() # [H, W]
                y_prob = F.sigmoid(y.detach().cpu())
                y_origin = torch.argmax(y_prob[0], dim=0).long() # [H, W]
                t = reconstruct_t_from_one_hot(x_origin, t[0].detach().cpu())
                n_pixel_wrong_total += (y_origin != t).sum().int()
                
            if i == total and (y_origin != t).sum().int() == 0:
                plot_xyt(x_origin.detach().cpu(), y_origin.detach().cpu(), t.detach().cpu())

            if i == total - 1:
                break

            opt.zero_grad()
            loss.backward()
            opt.step()

        print("Train loss: {:.6f}, N Pixels Wrong: {}".format(total_loss, n_pixel_wrong_total))
        self.log('Train loss', total_loss, prog_bar=True)
        return loss


class ShapeStableSolverIgnoreColorL(LightningModuleBase):
    def __init__(self, lr=0.001, model=None, *args, **kwargs):
        super().__init__(lr, *args, **kwargs)
        breakpoint()
        if model is not None:
            self.model = model(*args, **kwargs)
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
        n_pixel_wrong_total = 0

        # Process each batch
        for i, (x, t) in enumerate(batches):
            # forward + backward + optimize
            y = self.model(x)
            loss = self.loss_fn_source(y, t)
            total_loss += loss

            with torch.no_grad():
                x_origin = torch.argmax(x[0].detach().cpu(), dim=0).long() # [H, W]
                y_prob = F.sigmoid(y.detach().cpu())
                y_origin = torch.where(y_prob > 0.5, 1, 0).squeeze(0) # [C, H, W]
                n_pixel_wrong_total += (y_origin != t.detach().cpu()).sum().int()
                
            # if i == total-8 and (y_origin != t.detach().cpu()).sum().int() == 0:
            #     plot_xyt(x_origin.detach().cpu(), y_origin.detach().cpu(), t[0].detach().cpu())

            if i == total - 1:
                break

            opt.zero_grad()
            loss.backward()
            opt.step()

        print("Train loss: {:.6f}, N Pixels Wrong: {}".format(total_loss, n_pixel_wrong_total))
        self.log('Train loss', total_loss, prog_bar=True)
        return loss
