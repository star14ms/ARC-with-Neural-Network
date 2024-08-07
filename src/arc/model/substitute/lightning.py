import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from rich import print
from collections import defaultdict

from arc.model.substitute.pixel_each import PixelEachSubstitutor
from arc.utils.visualize import visualize_image_using_emoji, plot_xyt
from arc.utils.print import is_notebook


class LightningModuleBase(pl.LightningModule):
    def __init__(self, lr=0.001, save_n_perfect_epoch=4, *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.n_pixels_correct_in_epoch = 0
        self.n_pixels_total_in_epoch = 0
        self.n_tasks_correct_in_epoch = 0
        self.n_tasks_total_in_epoch = 0
        self.n_continuous_epoch_no_pixel_wrong = 0
        self.save_n_perfect_epoch = save_n_perfect_epoch
        self.automatic_optimization = False

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
        if self.no_label:
            return out
        self.n_pixels_correct_in_epoch += out['n_pixels_correct']
        self.n_pixels_total_in_epoch += out['n_pixels_total']
        self.n_tasks_correct_in_epoch += out['n_tasks_correct']
        self.n_tasks_total_in_epoch += out['n_tasks_total']
        return out

    def on_train_epoch_end(self):
        if self.no_label:
            return
        print('Epoch {} | Accuracy: {:>5.1f}% Tasks ({}/{}), {:>5.1f}% Pixels ({}/{})'.format(
            self.current_epoch+1, 
            self.n_tasks_correct_in_epoch / self.n_tasks_total_in_epoch * 100,
            self.n_tasks_correct_in_epoch,
            self.n_tasks_total_in_epoch,
            self.n_pixels_correct_in_epoch / self.n_pixels_total_in_epoch * 100, 
            self.n_pixels_correct_in_epoch,
            self.n_pixels_total_in_epoch
        ))
        
        current_epoch = self.trainer.current_epoch
        max_epochs = self.trainer.max_epochs

        if current_epoch+1 == max_epochs:
            self.progress.progress.stop()

        if self.n_pixels_correct_in_epoch == 0:
            self.n_continuous_epoch_no_pixel_wrong += 1
        else:
            self.n_continuous_epoch_no_pixel_wrong = 0

        if self.n_continuous_epoch_no_pixel_wrong == self.save_n_perfect_epoch:
            save_path = f"./output/{self.model.__class__.__name__}_{self.current_epoch+1:02d}ep.ckpt"
            self.trainer.save_checkpoint(save_path)
            print(f"Model saved to: {save_path}")

        # free up the memory
        self.n_pixels_correct_in_epoch = 0
        self.n_pixels_total_in_epoch = 0
        self.n_tasks_correct_in_epoch = 0
        self.n_tasks_total_in_epoch = 0


class PixelEachSubstitutorL(LightningModuleBase):
    def __init__(self, lr=0.001, model=None, n_trials=5, hyperparams_for_each_trial=[], max_epochs_for_each_task=300, train_loss_threshold_to_stop=0.01, top_k_submission=2, *args, **kwargs):
        super().__init__(lr, *args, **kwargs)

        model = model if model is not None else PixelEachSubstitutor
        self.model = model(*args, **kwargs)
        self.model_args = args
        self.model_kwargs = kwargs
        self.loss_fn = nn.CrossEntropyLoss()

        self.n_trials = n_trials
        self.params_for_each_trial = hyperparams_for_each_trial if hyperparams_for_each_trial else [{}]*n_trials
        self.max_epochs_for_each_task = max_epochs_for_each_task
        self.train_loss_threshold_to_stop = train_loss_threshold_to_stop

        self.submission = {}
        self.top_k_submission = top_k_submission
        self.test_results = defaultdict(list)
    
    def forward(self, inputs, *args, **kwargs):
        # In Lightning, forward defines the prediction/inference actions
        return self.model(inputs, *args, **kwargs)

    def training_step(self, batches):
        batches_train, batches_test, task_id = batches
        print('Task ID: [bold white]{}[/bold white]'.format(task_id))

        progress_id = self.progress._add_task(self.n_trials, f'Trial 1/{self.n_trials}')
        self.no_label = True if len(batches_test[0][1].shape) == 2 else False

        results = []
        for n in range(self.n_trials):
            info_train = self._training_step_train(batches_train, task_id, n)
            self.print_log('Train', **info_train)

            info, outputs = self._training_step_test(batches_test, task_id, n) if not self.no_label else self._test_step_test(batches_test, task_id, n)
            if not self.no_label:
                self.print_log('Test', **info)

            results.append({
                'accuracy': info_train['n_tasks_correct'] / info_train['n_tasks_total'],
                'loss': info_train['loss'],
                'outputs': outputs,
            })

            print(f"Trial {n+1}/{self.n_trials} | Restarting the training\n")
            self.progress._update(progress_id, n+1, description=f'Trial {n+1}/{self.n_trials}')

        self.progress.progress.remove_task(progress_id)
        self.add_submission(task_id, results)

        return info

    def _training_step_train(self, batches_train, task_id, n):
        max_epochs_for_each_task = self.max_epochs_for_each_task
        n_tasks_total = sum([len(b[0]) for b in batches_train])
        
        total_loss = 0
        progress_id = self.progress._add_task(max_epochs_for_each_task, f'Epoch 1/{max_epochs_for_each_task}')
        
        self.model.__init__(*self.model_args, **{**self.model_kwargs, **self.params_for_each_trial[n]})
        self.model.to(batches_train[0][0].device)
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        for i in range(max_epochs_for_each_task):
            self.update_task_progress(progress_id, i, task_id, total_loss)
            total_loss = 0
            n_pixels_correct = 0
            n_pixels_total = 0
            n_tasks_correct = 0

            for (x, t) in batches_train:
                y = self.model(x)
                loss = self.loss_fn(y, t)

                opt.zero_grad()
                loss.backward()
                opt.step()
                
                total_loss += loss.item()

                with torch.no_grad():
                    y_origin = torch.argmax(y.detach().cpu(), dim=1).long()
                    t_origin = torch.argmax(t.detach().cpu(), dim=1).long()
                    n_tasks_correct += sum(torch.all(y_one == t_one).item() for y_one, t_one in zip(y_origin, t_origin))
                    n_pixels_total += t_origin.numel()
                    n_pixels_correct += (y_origin == t_origin).sum().int()

            if n_tasks_correct == n_tasks_total and total_loss < self.train_loss_threshold_to_stop:
                break

        self.progress.progress.remove_task(progress_id)

        return {
            'loss': total_loss,
            'n_pixels_correct': n_pixels_correct,
            'n_pixels_total': n_pixels_total,
            'n_tasks_correct': n_tasks_correct,
            'n_tasks_total': n_tasks_total,
        }

    def _training_step_test(self, batches_test, task_id, n):
        total_loss = 0
        n_pixels_correct = 0
        n_pixels_total = 0
        n_tasks_correct = 0
        task_result = []
        outputs = []

        for i, (x, t) in enumerate(batches_test):
            y = self.model(x)
            loss = self.loss_fn(y, t)
            total_loss += loss

            with torch.no_grad():
                y_origin = torch.argmax(y.detach().cpu(), dim=1).long()
                t_origin = torch.argmax(t.detach().cpu(), dim=1).long()
                n_correct = (y_origin == t_origin).sum().int()
                n_pixels = t_origin.numel()

                n_pixels_total += n_pixels
                n_tasks_correct += n_correct == n_pixels
                n_pixels_correct += n_correct
                
                outputs.append(y_origin[0])
                
            correct_pixels = torch.where(y_origin == t_origin, 3, 2)
            if is_notebook():
                plot_xyt(x[0], y[0], t[0], correct_pixels, task_id=task_id)
            else:
                visualize_image_using_emoji(x[0], y[0], t[0], correct_pixels)

            task_result.append({
                'input': x[0].tolist(),
                'output': y_origin[0].tolist(),
                'target': t_origin[0].tolist(),
                'correct_pixels': correct_pixels[0].tolist(),
                'hparams': {**self.model_kwargs, **self.params_for_each_trial[n]},
            })

            print("Test {} | Correct: {} | Accuracy: {:>5.1f}% ({}/{})".format(
                i+1, '🟩' if n_correct == n_pixels else '🟥', n_correct/n_pixels*100, n_correct, n_pixels, 
            ))

        self.test_results[task_id].append(task_result)

        return {
            'loss': total_loss, 
            'n_pixels_correct': n_pixels_correct, 
            'n_pixels_total': n_pixels_total, 
            'n_tasks_correct': n_tasks_correct, 
            'n_tasks_total': len(batches_test),
        }, outputs

    def _test_step_test(self, batches_test, task_id, n):
        task_result = []
        outputs = []

        for x, _ in batches_test:
            y = self.model(x)

            with torch.no_grad():
                y_origin = torch.argmax(y.detach().cpu(), dim=1).long()
                outputs.append(y_origin[0])
                
            if is_notebook():
                plot_xyt(x[0], y[0], task_id=task_id)
            else:
                visualize_image_using_emoji(x[0], y[0])

            task_result.append({
                'input': x[0].tolist(),
                'output': y_origin[0].tolist(),
                'hparams': {**self.model_kwargs, **self.params_for_each_trial[n]},
            })

        self.test_results[task_id].append(task_result)

        return {}, outputs

    def on_train_start(self):
        self.progress = filter(lambda callback: hasattr(callback, 'progress'), self.trainer.callbacks).__next__()

    def update_task_progress(self, progress_id, i, task_id, total_loss_train):
        self.progress._update(progress_id, i+1, description=f'Epoch {i+1}/{self.max_epochs_for_each_task}')
        self.trainer.progress_bar_metrics['Task ID'] = task_id
        self.trainer.progress_bar_metrics['Train Loss'] = '{:.4f}'.format(total_loss_train)
        self.progress._update_metrics(self.trainer, self)
        self.progress.refresh()

    @staticmethod
    def print_log(mode, n_tasks_correct, n_tasks_total, n_pixels_correct, n_pixels_total, loss, end='\n'):
        print('{} Accuracy: {:>5.1f}% Tasks ({}/{}), {:>5.1f}% Pixels ({}/{}) | {} loss {:.4f}'.format(
            mode,
            n_tasks_correct / n_tasks_total * 100,
            n_tasks_correct,
            n_tasks_total,
            n_pixels_correct / n_pixels_total * 100, 
            n_pixels_correct,
            n_pixels_total,
            mode,
            loss
        ), end=end)

    def add_submission(self, task_id, results):
        # choose top k outputs from all trials. Accuracy is the first priority, loss is the second.
        results = sorted(results, key=lambda x: (x['accuracy'], -x['loss']), reverse=True)

        results_with_idx = [(i, result) for i, result in enumerate(results)]
        results_with_idx = sorted(results_with_idx, key=lambda x: (x[1]['accuracy'], -x[1]['loss']), reverse=True)
        results = [result for _, result in results_with_idx]
        idxs_priority = [i for i, _ in results_with_idx]

        submission_task = [{} for _ in range(len(results[0]['outputs']))]
        for i, result in enumerate(results[:self.top_k_submission]):
            for j, output in enumerate(result['outputs']):
                submission_task[j][f'attempt_{i+1}'] = output.tolist()

        self.submission[task_id] = submission_task
        
        # change the format of the test results reordering based on the priority
        self.test_results[task_id] = [self.test_results[task_id][idx] for idx in idxs_priority]
        self.test_results[task_id] = list(zip(*self.test_results[task_id]))
