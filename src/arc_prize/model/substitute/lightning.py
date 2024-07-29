import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from rich import print

from arc_prize.model.substitute.pixel_each import PixelEachSubstitutor
from arc_prize.utils.visualize import visualize_image_using_emoji


class LightningModuleBase(pl.LightningModule):
    def __init__(self, lr=0.001, save_n_perfect_epoch=4, *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.n_pixels_wrong_in_epoch = 0
        self.n_pixels_total_in_epoch = 0
        self.n_tasks_wrong_in_epoch = 0
        self.n_tasks_total_in_epoch = 0
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
        self.n_pixels_total_in_epoch += out['n_pixels_total']
        self.n_tasks_wrong_in_epoch += out['n_tasks_wrong']
        self.n_tasks_total_in_epoch += out['n_tasks_total']
        return out

    def on_train_epoch_end(self):
        print('Epoch {} | {:>5.1f}% Tasks Correct ({:>3} Tasks Wrong) | {:>5.1f}% Pixels Correct ({} Pixels Wrong)'.format(
            self.current_epoch+1, 
            (self.n_tasks_total_in_epoch - self.n_tasks_wrong_in_epoch) / self.n_tasks_total_in_epoch * 100,
            self.n_tasks_wrong_in_epoch,
            (self.n_pixels_total_in_epoch - self.n_pixels_wrong_in_epoch) / self.n_pixels_total_in_epoch * 100, 
            self.n_pixels_wrong_in_epoch
        ))

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
        self.n_pixels_total_in_epoch = 0
        self.n_tasks_wrong_in_epoch = 0
        self.n_tasks_total_in_epoch = 0


class PixelEachSubstitutorL(LightningModuleBase):
    def __init__(self, lr=0.001, model=None, *args, **kwargs):
        super().__init__(lr, *args, **kwargs)

        model = model if model is not None else PixelEachSubstitutor
        self.model = model(*args, **kwargs)
        self.loss_fn_source = nn.CrossEntropyLoss()
    
    def forward(self, inputs, *args, **kwargs):
        # In Lightning, forward defines the prediction/inference actions
        return self.model(inputs, *args, **kwargs)

    def training_step(self, batches):
        batches_train, batches_test, task_id = batches

        self.model = PixelEachSubstitutor()
        self.model.to(batches_train[0][0].device)
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        max_epochs = 50
        total_loss_train = 0
        progress_id = self.progress._add_task(max_epochs, f'Epoch 1/{max_epochs}')

        # Find Pattern in Train Data
        for i in range(max_epochs):
            self.update_progress(progress_id, i, max_epochs, task_id, total_loss_train)
            total_loss_train = 0

            for (x, t) in batches_train:
                # forward + backward + optimize
                y = self.model(x)
                loss = self.loss_fn_source(y, t)

                opt.zero_grad()
                loss.backward()
                opt.step()
                
                total_loss_train += loss.item()

        self.progress.progress.remove_task(progress_id)
        self.model.eval()

        total_loss = 0
        n_pixels_wrong = 0
        n_pixels_total = 0
        n_tasks_wrong = 0

        # Process Test Data
        for i, (x, t) in enumerate(batches_test):
            y = self.model(x)
            loss = self.loss_fn_source(y, t)
            total_loss += loss

            with torch.no_grad():
                y_prob = F.sigmoid(y.detach().cpu())
                y_origin = torch.argmax(y_prob, dim=1).long() # [H, W]
                t_origin = torch.argmax(t.detach().cpu(), dim=1).long()
                n_pixels_correct = (y_origin == t_origin).sum().int()
                n_pixels = t_origin.numel()
                correct_ratio = n_pixels_correct / n_pixels * 100

                n_pixels_total += n_pixels
                n_tasks_wrong += 1 if n_pixels_correct != n_pixels else 0
                n_pixels_wrong += n_pixels - n_pixels_correct
                # visualize_image_using_emoji(x[0], t[0], y[0], torch.where(y_origin == t_origin, 3, 2))

                self.log('N Pixels Wrong', n_pixels_wrong.float())
                print("Task {} | Train loss: {:.6f} | {:>5.1f}% Correct ({} Pixels Wrong)".format(task_id, loss, correct_ratio, n_pixels - n_pixels_correct))

        return {
            'loss': total_loss, 
            'n_pixels_total': n_pixels_total, 
            'n_pixels_wrong': n_pixels_wrong, 
            'n_tasks_wrong': n_tasks_wrong, 
            'n_tasks_total': len(batches_test)
        }

    def on_train_start(self):
        self.progress = filter(lambda callback: hasattr(callback, 'progress'), self.trainer.callbacks).__next__()

    def update_progress(self, progress_id, i, max_epochs, task_id, total_loss_train):
        self.progress._update(progress_id, i+1, description=f'Epoch {i+1}/{max_epochs}')
        self.trainer.progress_bar_metrics['Task ID'] = task_id
        self.trainer.progress_bar_metrics['Train Loss'] = total_loss_train
        self.progress._update_metrics(self.trainer, self)
        self.progress.refresh()
