import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from rich import print

from arc_prize.model.fill.keep_input import FillerKeepInput
from arc_prize.model.fill.ignore_color import FillerKeepInputIgnoreColor
from arc_prize.preprocess import one_hot_encode, one_hot_encode_changes, reconstruct_t_from_one_hot
from arc_prize.utils.visualize import plot_xyt, visualize_image_using_emoji


class LightningModuleBase(pl.LightningModule):
    def __init__(self, lr=0.001, *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.n_pixel_wrong_total_in_epoch = 0
        self.saved_when_no_pixel_wrong = False

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
        self.n_pixel_wrong_total_in_epoch += out['n_pixel_wrong_total']
        return out

    def on_train_epoch_end(self):
        if self.n_pixel_wrong_total_in_epoch == 0 and not self.saved_when_no_pixel_wrong:
            save_path = f"./output/{self.model.__class__.__name__}_{self.current_epoch+1:02d}ep.ckpt"
            self.trainer.save_checkpoint(save_path)
            print(f"Model saved to: {save_path}")
            
            self.saved_when_no_pixel_wrong = True

        # free up the memory
        self.n_pixel_wrong_total_in_epoch = 0


class FillerKeepInputL(LightningModuleBase):
    def __init__(self, lr=0.001, model=None, *args, **kwargs):
        super().__init__(lr, *args, **kwargs)

        model = model if model is not None else FillerKeepInput
        self.model = model(*args, **kwargs)
        self.loss_fn_source = nn.BCEWithLogitsLoss()

        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model.to(device)
    
    def forward(self, inputs, *args, **kwargs):
        # In Lightning, forward defines the prediction/inference actions
        outputs = self.model(inputs, *args, **kwargs)
        return outputs

    def training_step(self, batches):
        batches_train, _ = batches
        total = len(batches_train)
        opt = self.optimizers()
        total_loss = 0
        n_pixel_wrong_total = 0

        # Process each batch
        for i, (x, t) in enumerate(batches_train):
            # forward + backward + optimize
            y = self.model(x)
            loss = self.loss_fn_source(y, t)
            total_loss += loss

            with torch.no_grad():
                # x_origin = torch.argmax(x.detach().cpu(), dim=1).long() # [H, W]
                y_prob = F.sigmoid(y.detach().cpu())
                y_origin = torch.argmax(y_prob, dim=1).long() # [H, W]
                t_origin = torch.argmax(t.detach().cpu(), dim=1).long()
                n_pixel_wrong_total += (y_origin != t_origin).sum().int()
                # visualize_image_using_emoji(x[0], y[0], t[0])

            # if i == total-1: # and (y_origin != t).sum().int() == 0:
            #     plot_xyt(x_origin.detach().cpu(), y_origin.detach().cpu(), t_origin.detach().cpu())

            if i == total - 1:
                break

            opt.zero_grad()
            loss.backward()
            opt.step()

        print("Epoch {} | Train loss: {:.6f} | N Pixels Wrong: {}".format(self.current_epoch+1, total_loss, n_pixel_wrong_total))
        self.log('N Pixels Wrong', n_pixel_wrong_total.to(torch.float32))
        self.log('Train loss', total_loss, prog_bar=True)
        return {'loss': loss, 'n_pixel_wrong_total': n_pixel_wrong_total}


class FillerKeepInputIgnoreColorL(LightningModuleBase):
    def __init__(self, lr=0.001, model=None, *args, **kwargs):
        super().__init__(lr, *args, **kwargs)

        model = model if model is not None else FillerKeepInputIgnoreColor
        self.model = model(*args, **kwargs)
        self.loss_fn_source = nn.BCEWithLogitsLoss()
        
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model.to(device)

    def forward(self, inputs, *args, **kwargs):
        # In Lightning, forward defines the prediction/inference actions
        return self.model(inputs)

    def training_step(self, batches):
        batches_train, _ = batches
        total = len(batches_train)
        opt = self.optimizers()
        total_loss = 0
        n_pixel_wrong_total = 0

        # Process each batch
        for i, (x, t) in enumerate(batches_train):
            # forward + backward + optimize
            y = self.model(x)
            loss = self.loss_fn_source(y, t)
            total_loss += loss

            with torch.no_grad():
                y_prob = F.sigmoid(y.detach().cpu())
                y_origin = torch.where(y_prob > 0.5, 1, 0).squeeze(0) # [C, H, W]
                n_pixel_wrong_total += (y_origin != t.detach().cpu()).sum().int()
                # x_origin = torch.argmax(x[0].detach().cpu(), dim=0).long() # [H, W]

            # if i == total-8 and (y_origin != t.detach().cpu()).sum().int() == 0:
            #     plot_xyt(x_origin.detach().cpu(), y_origin.detach().cpu(), t[0].detach().cpu())

            if i == total - 1:
                break

            opt.zero_grad()
            loss.backward()
            opt.step()

        print("Epoch {} | Train loss: {:.6f} | N Pixels Wrong: {}".format(self.current_epoch+1, total_loss, n_pixel_wrong_total))
        self.log('Train loss', total_loss, prog_bar=True)
        self.log('N Pixels Wrong', n_pixel_wrong_total.to(torch.float32))
        return {'loss': loss, 'n_pixel_wrong_total': n_pixel_wrong_total}
