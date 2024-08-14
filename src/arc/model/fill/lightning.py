import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from rich import print

from arc.model.fill.keep_input import FillerKeepInput
from arc.model.fill.ignore_color import FillerKeepInputIgnoreColor
from arc.utils.visualize import plot_xytc, visualize_image_using_emoji


class LightningModuleBase(pl.LightningModule):
    def __init__(self, lr=0.001, save_n_perfect_epoch=4, *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.n_pixels_wrong_in_epoch = 0
        self.n_continuous_epoch_no_pixel_wrong = 0
        self.save_n_perfect_epoch = save_n_perfect_epoch

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self) -> torch.Any:
        return super().train_dataloader()

    def val_dataloader(self) -> torch.Any:
        return super().val_dataloader()

    def test_dataloader(self) -> torch.Any:
        return super().test_dataloader()
    
    def on_train_batch_end(self, out, batch, batch_idx):
        self.n_pixels_wrong_in_epoch += out['n_pixels_wrong']
        return out

    def on_train_epoch_end(self):
        if self.n_pixels_wrong_in_epoch == 0:
            self.n_continuous_epoch_no_pixel_wrong += 1
        else:
            self.n_continuous_epoch_no_pixel_wrong = 0

        if self.n_continuous_epoch_no_pixel_wrong == self.save_n_perfect_epoch:
            save_path = f"./output/{self.model.__class__.__name__}_{self.current_epoch+1:02d}ep.ckpt"
            self.trainer.save_checkpoint(save_path)
            print(f"Model saved to: {save_path}")

        # free up the memory
        self.n_pixels_wrong_in_epoch = 0


class FillerKeepInputL(LightningModuleBase):
    def __init__(self, lr=0.001, model=None, *args, **kwargs):
        super().__init__(lr, *args, **kwargs)

        model = model if model is not None else FillerKeepInput
        self.model = model(*args, **kwargs)
        self.loss_fn_source = nn.CrossEntropyLoss()
    
    def forward(self, inputs, *args, **kwargs):
        # In Lightning, forward defines the prediction/inference actions
        return self.model(inputs, *args, **kwargs)

    def training_step(self, batches):
        batches_train, *_ = batches
        total = len(batches_train)
        opt = self.optimizers()
        total_loss = 0
        n_pixels_wrong = 0

        # Process each batch
        for i, (x, t) in enumerate(batches_train):
            # forward + backward + optimize
            y = self.model(x)
            loss = self.loss_fn_source(y, t)
            total_loss += loss

            with torch.no_grad():
                # y_prob = F.softmax(y.detach().cpu(), dim=1)
                y_origin = torch.argmax(y.detach().cpu(), dim=1).long() # [H, W]
                t_origin = torch.argmax(t.detach().cpu(), dim=1).long()
                n_pixels_wrong += (y_origin != t_origin).sum().int()
                # visualize_image_using_emoji(x[0], t[0], y[0], titles=['Input', 'Target', 'Output'])

            # if i == total-1: # and (y_origin != t).sum().int() == 0:
            #     plot_xytc(x_origin, y_origin, t_origin)

            if i == total - 1:
                break

            opt.zero_grad()
            loss.backward()
            opt.step()

        print("Epoch {} | Train loss: {:.6f} | N Pixels Wrong: {}".format(self.current_epoch+1, total_loss, n_pixels_wrong))
        self.log('N Pixels Wrong', n_pixels_wrong.float())
        self.log('Train loss', total_loss, prog_bar=True)
        return {'loss': loss, 'n_pixels_wrong': n_pixels_wrong}


class FillerKeepInputIgnoreColorL(LightningModuleBase):
    def __init__(self, lr=0.001, model=None, *args, **kwargs):
        super().__init__(lr, *args, **kwargs)

        model = model if model is not None else FillerKeepInputIgnoreColor
        self.model = model(*args, **kwargs)
        self.loss_fn_source = nn.BCEWithLogitsLoss()

    def forward(self, inputs, *args, **kwargs):
        # In Lightning, forward defines the prediction/inference actions
        return self.model(inputs, *args, **kwargs)

    def training_step(self, batches):
        batches_train, *_ = batches
        total = len(batches_train)
        opt = self.optimizers()
        total_loss = 0
        n_pixels_wrong = 0

        # Process each batch
        for i, (x, t) in enumerate(batches_train):
            # forward + backward + optimize
            y = self.model(x)
            loss = self.loss_fn_source(y, t)
            total_loss += loss

            with torch.no_grad():
                y_prob = F.sigmoid(y.detach().cpu())
                y_origin = torch.where(y_prob > 0.5, 1, 0).squeeze(0) # [C, H, W]
                n_pixels_wrong += (y_origin != t.detach().cpu()).sum().int()
                # x_origin = torch.argmax(x[0].detach().cpu(), dim=0).long() # [H, W]

            # if i == total-8 and (y_origin != t.detach().cpu()).sum().int() == 0:
            #     plot_xytc(x_origin, y_origin, t[0])

            if i == total - 1:
                break

            opt.zero_grad()
            loss.backward()
            opt.step()

        print("Epoch {} | Train loss: {:.6f} | N Pixels Wrong: {}".format(self.current_epoch+1, total_loss, n_pixels_wrong))
        self.log('Train loss', total_loss, prog_bar=True)
        self.log('N Pixels Wrong', n_pixels_wrong.float())
        return {'loss': loss, 'n_pixels_wrong': n_pixels_wrong}
