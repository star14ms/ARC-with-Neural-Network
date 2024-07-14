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
        self.criterion = nn.CrossEntropyLoss()
        
        device = 'mps' if torch.backends.mps.is_available() else None
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
            source_one_hot, target_one_hot = t
            y = self.model(x[0])
            y_prob = F.softmax(y)
            loss = self.criterion(y_prob, source_one_hot[0])
            total_loss += loss

            if i == total - 1:
                break

            opt.zero_grad()
            loss.backward()
            opt.step()
            
        loss_avg = total_loss / total
        print("Train loss: {:.6f}".format(loss_avg))
        self.log('Train loss', loss_avg)
        return loss
