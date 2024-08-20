import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
import os
import copy
import warnings
from rich import print

from arc.model.substitute.base import PixelEachSubstitutorNonColorEncoding
from arc.model.substitute.color_encode import PixelEachSubstitutor
from arc.preprocess import one_hot_encode
from arc.utils.visualize import visualize_image_using_emoji, plot_xytc
from arc.utils.print import is_notebook


class LightningModuleBase(pl.LightningModule):
    def __init__(self, lr=0.001, save_n_perfect_epoch=4, top_k_submission=2, save_dir=None, *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.save_n_perfect_epoch = save_n_perfect_epoch
        self.top_k_submission = top_k_submission

        self.submission = {}
        self.test_results = defaultdict(list)
        self.save_dir = save_dir
        self.log_file = self.save_dir + '/train.log' if save_dir is not None else None

        self.n_tasks_correct_in_epoch = 0
        self.n_tasks_total = 0
        self.n_trials_correct_in_epoch = 0
        self.n_trials_total_in_epoch = 0
        # self.n_continuous_epoch_no_pixel_wrong = 0

        self.automatic_optimization = False
        self.is_notebook = is_notebook()

    def forward(self, inputs, *args, **kwargs):
        # In Lightning, forward defines the prediction/inference actions
        return self.model(inputs, *args, **kwargs)

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
        if self.is_notebook:
            train_description = "Task {}/{}".format(batch_idx + 1, self._trainer.fit_loop.max_batches)
            os.system(f'echo \"{train_description}\"')

        if self.no_label:
            return out

        self.n_tasks_correct_in_epoch += out['is_task_correct']
        self.n_trials_correct_in_epoch += out['n_trials_correct']
        self.n_trials_total_in_epoch += out['n_trials_total']

        return out

    def on_train_epoch_end(self):
        if self.no_label:
            return
        self.print_and_write('Epoch {} | Accuracy: {:>5.1f}% Tasks ({}/{}), {:>5.1f}% Trials ({}/{})'.format(
            self.current_epoch+1, 
            self.n_tasks_correct_in_epoch / self.n_tasks_total * 100,
            self.n_tasks_correct_in_epoch,
            self.n_tasks_total,
            self.n_trials_correct_in_epoch / self.n_trials_total_in_epoch * 100,
            self.n_trials_correct_in_epoch,
            self.n_trials_total_in_epoch
        ))

        # current_epoch = self.trainer.current_epoch
        # max_epochs = self.trainer.max_epochs

        # if current_epoch+1 == max_epochs:
        #     self.progress.progress.stop()

        # if self.n_tasks_correct_in_epoch == self.n_tasks_total:
        #     self.n_continuous_epoch_no_pixel_wrong += 1
        # else:
        #     self.n_continuous_epoch_no_pixel_wrong = 0

        # if self.n_continuous_epoch_no_pixel_wrong == self.save_n_perfect_epoch:
        #     save_path = f"./output/{self.model.__class__.__name__}_{self.current_epoch+1:02d}ep.ckpt"
        #     self.trainer.save_checkpoint(save_path)
        #     self.print_and_write(f"Model saved to: {save_path}")

        # free up the memory
        self.n_tasks_correct_in_epoch = 0
        self.n_tasks_total = 0

    def on_train_start(self):
        self.progress = filter(lambda callback: hasattr(callback, 'progress'), self.trainer.callbacks).__next__()
        self.n_tasks_total = self.trainer.fit_loop.max_batches

    def update_task_progress(self, progress_id, current=None, task_id=None, loss=None, n_queue=None, depth=None, completed=None, description=None):
        self.progress._update(progress_id, current=current, completed=completed, description=description)
        if task_id is not None:
            self.trainer.progress_bar_metrics['Task ID'] = task_id
        if loss is not None:
            self.trainer.progress_bar_metrics['Loss'] = '{:.4f}'.format(loss)
        if n_queue is not None:
            self.trainer.progress_bar_metrics['Queue'] = '{}'.format(n_queue)
        if depth is not None:
            self.trainer.progress_bar_metrics['Depth'] = '{}'.format(depth)
        self.progress._update_metrics(self.trainer, self)
        self.progress.refresh()

    def print_log(self, mode, n_sub_tasks_correct, n_sub_tasks_total, loss):
        self.print_and_write('{} Accuracy: {:>5.1f}% Tasks ({}/{}) | {} loss {:.4f}'.format(
            mode,
            n_sub_tasks_correct / n_sub_tasks_total * 100,
            n_sub_tasks_correct,
            n_sub_tasks_total,
            mode,
            loss
        ))

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

    def print_and_write(self, log, end='\n'):
        print(log, end=end)
        if self.log_file is not None:
            with open(self.log_file, 'a') as f:
                print(log, end=end, file=f)


class PixelEachSubstitutorBase(LightningModuleBase):
    def __init__(self, lr=0.001, n_trials=5, save_dir=None, hyperparams_for_each_cell=[], max_epochs_for_each_task=300, train_loss_threshold_to_stop=0.01, *args, **kwargs):
        super().__init__(lr=lr, save_dir=save_dir, *args, **kwargs)
        
        if self.top_k_submission > n_trials:
            warnings.warn(f'top_k_submission should be less than or equal to n_trials. top_k_submission is set to {n_trials}.')

        self.model = self.model_class(*args, **kwargs)
        self.model_args = args
        self.model_kwargs = kwargs
        self.loss_fn = nn.CrossEntropyLoss()

        self.n_trials = n_trials
        self.params_for_each_cell = [{ 'id': 0 }] + (hyperparams_for_each_cell if hyperparams_for_each_cell else [])
        self.max_epochs_for_each_task = max_epochs_for_each_task
        self.train_loss_threshold_to_stop = train_loss_threshold_to_stop

    def training_step(self, batches):
        batches_train, batches_test, task_id = batches
        self.print_and_write('Task ID: [bold white]{}[/bold white]'.format(task_id))

        id_prog_trial = self.progress._add_task(self.n_trials, f'Trial 0/{self.n_trials}')
        self.no_label = True if len(batches_test[0][1].shape) == 2 else False

        results = []
        for n in range(self.n_trials):
            info_train = self._training_step(batches_train, task_id, n)
            self.print_log('Train', **info_train)

            info, outputs = self._training_step_test(batches_test, task_id, n) if not self.no_label else self._test_step_test(batches_test, task_id, n)
            if not self.no_label:
                self.print_log('Test', **info)

            results.append({
                'accuracy': info_train['n_sub_tasks_correct'] / info_train['n_sub_tasks_total'],
                'loss': info_train['loss'],
                'outputs': outputs,
            })

            self.print_and_write(f"Trial {n+1}/{self.n_trials} | Restarting the training\n")
            self.progress._update(id_prog_trial, n+1, description=f'Trial {n+1}/{self.n_trials}')

        self.progress.progress.remove_task(id_prog_trial)

        if self.current_epoch+1 == self.trainer.max_epochs:
            self.add_submission(task_id, results)

        is_task_correct, n_trials_correct, n_trials_total = self.get_task_result(task_id, len(batches_test))

        return {
            'n_trials_correct': n_trials_correct,
            'n_trials_total': n_trials_total,
            'is_task_correct': is_task_correct,
        }

    def _training_step(self, batches_train, task_id, n):
        max_epochs_for_each_task = self.max_epochs_for_each_task
        n_sub_tasks_total = sum([len(b[0]) for b in batches_train])
        
        total_loss = 0
        loss_prev = 0
        n_times_constant_loss = 0
        progress_id = self.progress._add_task(max_epochs_for_each_task, f'Epoch 0/{max_epochs_for_each_task}')
        
        self.model.__init__(*self.model_args, **{**self.model_kwargs, **self.params_for_each_cell[n]})
        self.model.to(batches_train[0][0].device)
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        for i in range(max_epochs_for_each_task):
            self.update_task_progress(progress_id, i+1, task_id, loss=total_loss, description=f'Epoch {i+1}/{self.max_epochs_for_each_task}')
            total_loss = 0
            self.n_sub_tasks_correct = 0

            for (x, t) in batches_train:
                y = self.model(x)
                loss = self.loss_fn(y, t)
                total_loss += loss.item()

                opt.zero_grad()
                loss.backward()
                opt.step()
                
                self.update_prediction_info(y, t)

            if loss_prev == total_loss:
                n_times_constant_loss += 1
            else:
                n_times_constant_loss = 0
                loss_prev = total_loss

            if self.n_sub_tasks_correct == n_sub_tasks_total and total_loss < self.train_loss_threshold_to_stop or (total_loss > 0.5 and n_times_constant_loss == 10):
                break

        self.progress.progress.remove_task(progress_id)

        return {
            'loss': total_loss,
            'n_sub_tasks_correct': self.n_sub_tasks_correct,
            'n_sub_tasks_total': n_sub_tasks_total,
        }

    def _training_step_test(self, batches_test, task_id, n):
        total_loss = 0
        self.n_sub_tasks_correct = 0
        task_result = []
        outputs = []

        for i, (x, t) in enumerate(batches_test):
            y = self.model(x)
            loss = self.loss_fn(y, t)
            total_loss += loss

            y_decoded, t_decoded, n_correct, n_pixels = self.update_prediction_info(y, t)
            outputs.append(y_decoded[0])

            correct_pixels = torch.where(y_decoded == t_decoded, 3, 2)
            if self.is_notebook:
                plot_xytc(x[0], y[0], t[0], correct_pixels, task_id=task_id)
            else:
                visualize_image_using_emoji(x[0], t[0], y[0], correct_pixels, titles=['Input', 'Target', 'Output', 'Correct'])
                visualize_image_using_emoji(x[0], t[0], y[0], correct_pixels, titles=['Input', 'Target', 'Output', 'Correct'], output_file=self.log_file)

            task_result.append({
                'input': x[0].tolist(),
                'output': y_decoded[0].tolist(),
                'target': t_decoded[0].tolist(),
                'correct_pixels': correct_pixels[0].tolist(),
                'hparams_ids': [self.params_for_each_cell[n].get('id')],
            })

            self.print_and_write("Test {} | Correct: {} | Accuracy: {:>5.1f}% ({}/{})".format(
                i+1, '🟩' if n_correct == n_pixels else '🟥', n_correct/n_pixels*100, n_correct, n_pixels, 
            ))

        self.test_results[task_id].append(task_result)

        return {
            'loss': total_loss, 
            'n_sub_tasks_correct': self.n_sub_tasks_correct, 
            'n_sub_tasks_total': len(batches_test),
        }, outputs

    def _test_step_test(self, batches_test, task_id, n):
        task_result = []
        outputs = []

        for x, _ in batches_test:
            y = self.model(x)

            with torch.no_grad():
                y_decoded = torch.argmax(y.detach().cpu(), dim=1).long()
                outputs.append(y_decoded[0])

            if self.is_notebook:
                plot_xytc(x[0], y[0], task_id=task_id)
            else:
                visualize_image_using_emoji(x[0], y[0])

            task_result.append({
                'input': x[0].tolist(),
                'output': y_decoded[0].tolist(),
                'hparams_ids': [self.params_for_each_cell[n].get('id')],
            })

        self.test_results[task_id].append(task_result)

        return {}, outputs

    @torch.no_grad()
    def update_prediction_info(self, y, t):
        y_decoded = torch.argmax(y.detach().cpu(), dim=1).long()
        t_decoded = torch.argmax(t.detach().cpu(), dim=1).long()
        
        n_correct = (y_decoded == t_decoded).sum().int()
        n_pixels = t_decoded.numel()

        self.n_sub_tasks_correct += sum(torch.all(y_one == t_one).item() for y_one, t_one in zip(y_decoded, t_decoded))

        return y_decoded, t_decoded, n_correct, n_pixels

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if self.trainer.current_epoch+1 == self.trainer.max_epochs:
            self.test_results = {
                'results': self.test_results,
                'hparams': {**self.model_kwargs, 'hyperparams_for_each_cell': self.params_for_each_cell},
            }
   
    def get_task_result(self, task_id, n_sub_tasks):
        is_task_correct = True
        n_trials_correct = 0
        n_trials_total = self.n_trials * n_sub_tasks

        for subtask in self.test_results[task_id]:
            solved = False
            for i in range(min(self.top_k_submission, self.n_trials)):
                correct_pixels = subtask[i]['correct_pixels']
                if all(all(pixel == 3 for pixel in row) for row in correct_pixels):
                    n_trials_correct += 1
                    solved = True
                    
            if not solved:
                is_task_correct = False
                
        return is_task_correct, n_trials_correct, n_trials_total


class PixelEachSubstitutorRepeatBase(PixelEachSubstitutorBase):
    def __init__(self, max_AFS=200, max_queue=20, max_depth=4, verbose=False, n_repeat_max_acc_threshold=30, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_AFS = max_AFS
        self.max_queue = max_queue
        self.max_depth = max_depth
        self.verbose = verbose
        self.n_repeat_max_acc_threshold = n_repeat_max_acc_threshold

    def _training_step(self, batches_train, task_id, n):

        def __build_models():
            nonlocal next_id
            if len(models_prev) == 0 or (idx_cell >= 1 and extend): # models_prev[0] should train continuously
                # Create new model
                varied_kwargs = self.params_for_each_cell[idx_cell] if idx_cell >= 1 else {}
                model = self.model_class(*self.model_args, **{**self.model_kwargs, **varied_kwargs})
                model.to(batches_train[0][0].device)
                model.id = self.params_for_each_cell[idx_cell].get('id')
                model.instance_id = (next_id := next_id + 1)
                models = models_prev + [model]
                opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            elif acc_next == 1 or (idx_cell == 0):
                # Lengthen the last model
                models = models_prev + [models_prev[-1]]
                opt = opt_prev
            else:
                # Continue training
                models = models_prev
                opt = opt_prev

            return models, opt

        n_sub_tasks_total = sum([len(b[0]) for b in batches_train])

        accuracy0, answer_map0 = self.get_avg_accuracy(batches_train, return_corrects_info=True)
        checkpoint0 = {
            'models': [],
            'answer_map': answer_map0,
            'opt': None,
            'acc_prev': accuracy0,
            'acc_max': accuracy0,
            'n_epochs_trained': [],
            'extend': False,
            'is_corrected_pixels_maintained_prev': True,
            'is_corrected_pixels_maintained': True,
        }
        queue = [checkpoint0]

        id_prog_acc = self.progress._add_task(100, '  Acc {}'.format('0%'))
        id_prog_afs = self.progress._add_task(self.max_AFS, '  AFS {}'.format(f'0/{self.max_AFS}'))
        next_id = 0
        completed = False
        n_perfect_extension = 0

        for i in range(self.max_AFS): # AFS: Accuracy First Search
            self.update_task_progress(id_prog_afs, i+1, task_id=task_id, n_queue=len(queue), depth=max(len(queue[0]['models']), 1), description='  AFS {}'.format(f'{i+1}/{self.max_AFS}'))

            queue = sorted(queue, key=lambda x: (x['acc_max']), reverse=True) # x['is_corrected_pixels_maintained'], 
            queue = queue[:self.max_queue]
            checkpoint = queue.pop(0)
            models_prev, opt_prev, acc_prev, acc_max, extend, is_corrected_pixels_maintained = \
                checkpoint.get('models'), \
                checkpoint.get('opt'), \
                checkpoint.get('acc_prev'), \
                checkpoint.get('acc_max'), \
                checkpoint.get('extend'), \
                checkpoint.get('is_corrected_pixels_maintained'), \

            for idx_cell in range(len(self.params_for_each_cell)):
                models, opt = __build_models()
                
                max_epoch = self.max_epochs_for_each_task * (2 if len(models) == 1 else 1)
                id_prog_e = self.progress._add_task(max_epoch, f'Epoch 0/{max_epoch}')
                training_branch_generator = self._training_get_checkpoint_generaotr(models, batches_train, opt, acc_prev, acc_max, accuracy0, max_epoch, id_prog_e)

                result = self._training_step_generate_checkpoints(models, opt, training_branch_generator, checkpoint, id_prog_acc, len(queue), idx_cell)
                checkpoints_new, is_extended_correctly, acc_next, total_loss, n_sub_tasks_correct = result

                if is_extended_correctly:
                    n_perfect_extension += 1

                    if n_perfect_extension == 4:
                        completed = True

                # if len(checkpoints_new) > 0 and all([not x['is_corrected_pixels_maintained'] for x in checkpoints_new]):
                #     checkpoints_new = [checkpoints_new[-1]]
                    # min_acc = min([x['acc_prev'] for x in checkpoints_new])
                    # checkpoints_new = [x for x in checkpoints_new if x['acc_prev'] == min_acc or x['is_corrected_pixels_maintained']]

                print(is_corrected_pixels_maintained, round(acc_prev.item()*100, 2), '->')
                print([(x['is_corrected_pixels_maintained'], round(x['acc_prev'].item()*100, 2)) for x in checkpoints_new])
                print()
                queue.extend(checkpoints_new)

                if completed or (len(queue) > 0 and acc_next >= acc_max):
                    break

            if len(queue) == 0:
                queue.append(checkpoint0)
            if completed:
                break

        self.progress.progress.remove_task(id_prog_afs)
        self.progress.progress.remove_task(id_prog_acc)
        self.models = models
        # print('N Epochs Trained:', n_epochs_trained, 'N Recursion', len(models))

        return {
            'loss': total_loss,
            'n_sub_tasks_correct': n_sub_tasks_correct,
            'n_sub_tasks_total': n_sub_tasks_total,
        }

    def _training_step_generate_checkpoints(self, models: nn.Module, opt: torch.optim.Optimizer, training_branch_generator, checkpoint: dict, id_prog_acc: int, n_queue: int, idx_cell: int):
        answer_map_prev, acc_prev, acc_max, n_epochs_trained, is_corrected_pixels_maintained = \
            checkpoint.get('answer_map'), \
            checkpoint.get('acc_prev'), \
            checkpoint.get('acc_max'), \
            checkpoint.get('n_epochs_trained'), \
            checkpoint.get('is_corrected_pixels_maintained'), \

        checkpoints_new = []
        is_extended_correctly = False
        _acc_next = 0.0
        _total_loss = 0.0
        _n_sub_tasks_correct = 0

        for results, total_loss, acc_next, n_sub_tasks_correct, opt, n_epoch_trained in training_branch_generator:
            _acc_next = acc_next
            _total_loss = total_loss
            _n_sub_tasks_correct = n_sub_tasks_correct
            self.update_task_progress(id_prog_acc, n_queue=n_queue+len(checkpoints_new), completed=acc_max*100, description=f'  Acc {acc_max*100:.1f}%' if acc_max != 1 else f'  Acc 100%')

            answer_map = [result['c_decoded'] for result in results]
            is_corrected_pixels_maintained_next = self.is_corrected_pixels_maintained(answer_map_prev, answer_map) # Prvent extension before finding the input
          
            if is_corrected_pixels_maintained_next:
                for j in range(len(checkpoints_new)-1, -1, -1):
                    if not checkpoints_new[j]['is_corrected_pixels_maintained']:
                        del checkpoints_new[j]
                        
            if acc_next == 1 or (acc_next >= acc_max and ((len(models) > 2 and is_corrected_pixels_maintained) or is_corrected_pixels_maintained_next) and len(models) < self.max_depth):
                acc_max = max(acc_max, acc_next)

                checkpoint_kwargs = {
                    'answer_map': answer_map,
                    'acc_prev': acc_next,
                    'acc_max': acc_max,
                    'n_epochs_trained': n_epochs_trained + [n_epoch_trained],
                    'is_corrected_pixels_maintained_prev': is_corrected_pixels_maintained,
                    'is_corrected_pixels_maintained': is_corrected_pixels_maintained_next,
                    'extend': True,
                }

                if acc_next == 1 and acc_prev == acc_next:
                    is_extended_correctly = True

                    checkpoints_new.append({
                        'models': models,
                        'opt': opt,
                        **checkpoint_kwargs
                    })
                else:
                    # duplicated_performace = False
                    # for checkpoint in checkpoints_new:
                    #     if checkpoint['acc_prev'] == acc_next and self.is_same_performance(checkpoint['answer_map'], answer_map):
                    #         duplicated_performace = True
                    #         break
                    # if duplicated_performace:
                    #     continue

                    models_copied, opt_copied = self.copy_model_and_opt(models, opt, idx_cell)

                    if self.verbose:
                        self._print_one_step_result(results, answer_map_prev, len(models), [_model.instance_id for _model in models], is_corrected_pixels_maintained_next, acc_prev, acc_next, acc_max, n_epoch_trained, end='\n' + '-'*100 + '\n')

                    checkpoints_new.append({
                        'models': models_copied,
                        'opt': opt_copied,
                        **checkpoint_kwargs
                    })
                    
        return checkpoints_new, is_extended_correctly, _acc_next, _total_loss, _n_sub_tasks_correct

    def _training_get_checkpoint_generaotr(self, models: nn.Module, batches_train, opt: torch.optim.Optimizer, acc_prev: float, acc_max: float, accuracy0: float, max_epoch: int, id_prog_e: int):
        t_batch = [batch[1] for batch in batches_train]
        acc_max = 0
        n_repeat_max_acc = 0
        label_input = True if acc_prev == accuracy0 and acc_max < accuracy0 else False
        recognize_input = False

        for e in range(max_epoch):
            results = []
            total_loss = 0
            y_batch = []

            for i, (x, t) in enumerate(batches_train):
                y = x.detach().clone()
                for depth, model in enumerate(models):
                    y = model(y, epoch=e, batch_idx=i, return_prob=False if depth == len(models)-1 else True)

                    if depth != len(models)-1:
                        _, max_indices = torch.max(y, dim=1)
                        y = torch.nn.functional.one_hot(max_indices, num_classes=y.size(1)).to(x.dtype).permute(0, 3, 1, 2)
                loss = self.loss_fn(y, t if not label_input else x)

                opt.zero_grad()
                loss.backward()
                opt.step()

                x_decoded = torch.argmax(x, dim=1).long()
                y_decoded = torch.argmax(y, dim=1).long()
                t_decoded = torch.argmax(t, dim=1).long()

                # if (e+1) % 5 == 0:
                #     c_decoded = torch.where(y_decoded == t_decoded, 3, 2)
                #     for x_one, y_one, t_one, c_one in zip(x_decoded, y_decoded, t_decoded, c_decoded):
                #         visualize_image_using_emoji(x_one, t_one, y_one, c_one, titles=['Input', 'Target', 'Output', 'Correct'])
                #     print('-'*100)

                c_decoded = torch.where(y_decoded == t_decoded, 1, 0)
                results.append({'x_decoded': x_decoded, 'y_decoded': y_decoded, 't_decoded': t_decoded, 'c_decoded': c_decoded})
                total_loss += loss
                y_batch.append(y)

            acc_next, n_sub_tasks_correct = self.get_avg_accuracy(zip(y_batch, t_batch), return_n_sub_tasks_correct=True)
            n_repeat_max_acc = 0 if not acc_next <= acc_max or acc_next == 1 else (n_repeat_max_acc + 1)
            acc_max = max(acc_max, acc_next)

            if not recognize_input and acc_prev == accuracy0 and acc_max >= accuracy0:
                label_input = False
                recognize_input = True

            self.update_task_progress(id_prog_e, e+1, loss=loss, depth=len(models), description=f'Epoch {e+1}/{max_epoch}')

            if len(models) != 1 and n_repeat_max_acc == self.n_repeat_max_acc_threshold and len(models) <= 2:
                break
            if acc_next == 1 or total_loss < 0.0001 or (acc_prev < acc_next and acc_max == acc_next and len(models) < self.max_depth):
                yield results, total_loss, acc_next, n_sub_tasks_correct, opt, e+1

                if acc_next == 1:
                    break

        self.progress.progress.remove_task(id_prog_e)

        return results, total_loss, acc_next, n_sub_tasks_correct, opt, e+1

    def _training_step_test(self, batches_test, task_id, n, max_depth=20):
        total_loss = 0
        self.n_sub_tasks_correct = 0
        task_result = []
        outputs = []
        ys = []
        for model in self.models:
            model.eval()
        for i in range(max_depth-len(self.models)):
            self.models.append(self.models[-1])

        for i, (x, t) in enumerate(batches_test):
            y = x
            for depth, model in enumerate(self.models):
                y = model(y, return_prob=False if depth == len(self.models)-1 else True)
                ys.append((y, 'Depth {}'.format(depth+1)))

                if depth != len(self.models)-1:
                    _, max_indices = torch.max(y, dim=1)
                    y = torch.nn.functional.one_hot(max_indices, num_classes=y.size(1)).to(x.dtype).permute(0, 3, 1, 2)

            loss = self.loss_fn(y, t)
            total_loss += loss

            y_decoded, t_decoded, n_correct, n_pixels = self.update_prediction_info(y, t)
            c_decoded = torch.where(y_decoded == t_decoded, 3, 2)
            outputs.append(y_decoded[0])

            xytc = [(x, 'Input')] + ys + [(t_decoded, 'Target')] + [(c_decoded, 'Correct')]
            xytc_batches = [xytc[i:i+4] for i in range(0, len(xytc), 4)]

            for xytc_batch in xytc_batches:
                titles = [title for _, title in xytc_batch]
                xytcs = [xytc for xytc, _ in xytc_batch]
                visualize_image_using_emoji(*xytcs, titles=titles)
                visualize_image_using_emoji(*xytcs, titles=titles, output_file=self.log_file)

            if self.is_notebook:
                plot_xytc(x[0], y[0], t[0], c_decoded, task_id=task_id)
            else:
                visualize_image_using_emoji(x, t, y, c_decoded, titles=['Input', 'Target', 'Output', 'Correct'])
                visualize_image_using_emoji(x, t, y, c_decoded, titles=['Input', 'Target', 'Output', 'Correct'], output_file=self.log_file)

            task_result.append({
                'input': x[0].tolist(),
                'output': y_decoded[0].tolist(),
                'target': t_decoded[0].tolist(),
                'correct_pixels': c_decoded[0].tolist(),
                'hparams_ids': [model.id for model in self.models],
            })

            self.print_and_write("Test {} | Correct: {} | Accuracy: {:>5.1f}% ({}/{})".format(
                i+1, '🟩' if n_correct == n_pixels else '🟥', n_correct/n_pixels*100, n_correct, n_pixels, 
            ))

        if self.trainer.current_epoch+1 == self.trainer.max_epochs:
            self.test_results[task_id].append(task_result)

        return {
            'loss': total_loss, 
            'n_sub_tasks_correct': self.n_sub_tasks_correct, 
            'n_sub_tasks_total': len(batches_test),
        }, outputs

    def copy_model_and_opt(self, models, opt, idx_cell):
        kwargs = self.params_for_each_cell[idx_cell] if idx_cell < len(self.params_for_each_cell) else {}
        id = kwargs.pop('id', None)
        model = self.model_class(*self.model_args, **{**self.model_kwargs, **kwargs})
        model.load_state_dict(copy.deepcopy(models[-1].state_dict()))
        model.id = id
        model.instance_id = models[-1].instance_id
        models_copied = [model for _ in range(len(models))]

        opt_copied = torch.optim.Adam(model.parameters(), lr=self.lr)
        opt_copied.load_state_dict(copy.deepcopy(opt.state_dict()))
        
        return models_copied, opt_copied

    @staticmethod
    def get_avg_accuracy(batches, return_n_sub_tasks_correct=False, return_corrects_info=False):
        accuracy_total = 0
        data_total = 0
        n_sub_tasks_correct = 0
        c_decoded_batch = []

        for (y, t) in batches:
            y_decoded = y.argmax(dim=1).long()
            t_decoded = t.argmax(dim=1).long()
            c_decoded = torch.where(y_decoded == t_decoded, 1, 0)

            if return_corrects_info:
                c_decoded_batch.append(c_decoded)

            accuarcy_each = torch.sum(c_decoded, dim=(1, 2)) / y.argmax(dim=1)[0].numel() # [N, C, H, W]
            accuracy_total += torch.sum(accuarcy_each)
            data_total += len(y)
            n_sub_tasks_correct += torch.sum(torch.where(accuarcy_each == 1, 1, 0))

        avg_accuracy = accuracy_total / data_total

        results = [avg_accuracy]
        if return_n_sub_tasks_correct:
            results.append(n_sub_tasks_correct)
        if return_corrects_info:
            results.append(c_decoded_batch)

        if len(results) == 1:
            return results[0]
        return results

    @staticmethod
    def is_corrected_pixels_maintained(answer_map_prev, answer_map):
        for correct_previous_batch, correct_current_batch in zip(answer_map_prev, answer_map):
            if torch.any((correct_previous_batch - correct_current_batch) > 0):
                return False
        return True
    
    @staticmethod
    def is_same_performance(answer_map_prev, answer_map):
        for correct_previous_batch, correct_current_batch in zip(answer_map_prev, answer_map):
            if not torch.all(correct_previous_batch == correct_current_batch):
                return False
        return True

    @staticmethod
    def _print_one_step_result(results, answer_map_prev, model_length, model_ids, is_corrected_pixels_maintained_next, acc_prev, acc_next, acc_max, n_epoch_trained, end='\n'):
        for result, c_prev in zip(results, answer_map_prev):
            for x_one, y_one, t_one, c_one, c_prev_one in zip(result['x_decoded'], result['y_decoded'], result['t_decoded'], result['c_decoded'], c_prev):
                c_one = torch.where(c_one == 1, 3, 2)
                c_prev_one = torch.where(c_prev_one == 1, 3, 2)
                visualize_image_using_emoji(x_one, t_one, y_one, c_one, c_prev_one, titles=['Input', 'Target', 'Output', 'Correct', 'Correct Prev'])

        print('Accuracy: {:.1f}% -> {:.1f}% ({:.1f}) | Corrects Kept: {} | Depth: {} {} | {} Epoch'.format(
            acc_prev*100 if model_length != 1 else 0.0, 
            acc_next*100, 
            acc_max*100, 
            '[green]T[/green]' if is_corrected_pixels_maintained_next else '[red]F[/red]',
            model_length, 
            model_ids, # '▶️'*model_length, # 
            n_epoch_trained, 
        ), end=end)
        # print('Queue:', ' '.join(
        #     ['{}'.format((round(acc.item()*100, 1), [_model.instance_id for _model in _models])) \
        #     for _models, _, _, _, acc, _, _ in queue]
        # ))


class PixelEachSubstitutorL(PixelEachSubstitutorBase):
    def __init__(self, model=None, *args, **kwargs):
        self.model_class = model if model is not None else PixelEachSubstitutor

        super().__init__(*args, **kwargs)


class PixelEachSubstitutorRepeatL(PixelEachSubstitutorRepeatBase):
    def __init__(self, model=None, *args, **kwargs):
        self.model_class = model if model is not None else PixelEachSubstitutor

        super().__init__(*args, **kwargs)


class PixelEachSubstitutorNonColorEncodingL(PixelEachSubstitutorBase):
    def __init__(self, model=None, *args, **kwargs):
        self.model_class = model if model is not None else PixelEachSubstitutorNonColorEncoding

        super().__init__(*args, **kwargs)


class PixelEachSubstitutorRepeatNonColorEncodingL(PixelEachSubstitutorRepeatBase):
    def __init__(self, model=None, *args, **kwargs):
        self.model_class = model if model is not None else PixelEachSubstitutorNonColorEncoding

        super().__init__(*args, **kwargs)
