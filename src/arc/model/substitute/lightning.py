import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
import os
import copy
import pickle
from rich import print

from arc.model.substitute.pixel_each import PixelEachSubstitutor
from arc.preprocess import one_hot_encode
from arc.utils.visualize import visualize_image_using_emoji, plot_xyt
from arc.utils.print import is_notebook


class LightningModuleBase(pl.LightningModule):
    def __init__(self, lr=0.001, save_n_perfect_epoch=4, top_k_submission=2, *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.save_n_perfect_epoch = save_n_perfect_epoch
        self.top_k_submission = top_k_submission

        self.submission = {}
        self.test_results = defaultdict(list)

        self.n_tasks_correct_in_epoch = 0
        self.n_tasks_total_in_epoch = 0
        self.n_continuous_epoch_no_pixel_wrong = 0

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
        self.n_tasks_correct_in_epoch += out['n_tasks_correct']
        self.n_tasks_total_in_epoch += out['n_tasks_total']
        return out

    def on_train_epoch_end(self):
        if self.no_label:
            return
        print('Epoch {} | Accuracy: {:>5.1f}% Tasks ({}/{})'.format(
            self.current_epoch+1, 
            self.n_tasks_correct_in_epoch / self.n_tasks_total_in_epoch * 100,
            self.n_tasks_correct_in_epoch,
            self.n_tasks_total_in_epoch,
        ))
        
        current_epoch = self.trainer.current_epoch
        max_epochs = self.trainer.max_epochs

        if current_epoch+1 == max_epochs:
            self.progress.progress.stop()

        if self.n_tasks_correct_in_epoch == self.n_tasks_total_in_epoch:
            self.n_continuous_epoch_no_pixel_wrong += 1
        else:
            self.n_continuous_epoch_no_pixel_wrong = 0

        if self.n_continuous_epoch_no_pixel_wrong == self.save_n_perfect_epoch:
            save_path = f"./output/{self.model.__class__.__name__}_{self.current_epoch+1:02d}ep.ckpt"
            self.trainer.save_checkpoint(save_path)
            print(f"Model saved to: {save_path}")

        # free up the memory
        self.n_tasks_correct_in_epoch = 0
        self.n_tasks_total_in_epoch = 0

    def on_train_start(self):
        self.progress = filter(lambda callback: hasattr(callback, 'progress'), self.trainer.callbacks).__next__()

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

    @staticmethod
    def print_log(mode, n_tasks_correct, n_tasks_total, loss, end='\n'):
        print('{} Accuracy: {:>5.1f}% Tasks ({}/{}) | {} loss {:.4f}'.format(
            mode,
            n_tasks_correct / n_tasks_total * 100,
            n_tasks_correct,
            n_tasks_total,
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


class PixelEachSubstitutorL(LightningModuleBase):
    def __init__(self, lr=0.001, model=None, n_trials=5, hyperparams_for_each_trial=[], max_epochs_for_each_task=300, train_loss_threshold_to_stop=0.01, *args, **kwargs):
        super().__init__(lr, *args, **kwargs)

        self.model_class = model if model is not None else PixelEachSubstitutor
        self.model = self.model_class(*args, **kwargs)
        self.model_args = args
        self.model_kwargs = kwargs
        self.loss_fn = nn.CrossEntropyLoss()

        self.n_trials = n_trials
        self.params_for_each_trial = hyperparams_for_each_trial if hyperparams_for_each_trial else [{}]*n_trials
        self.max_epochs_for_each_task = max_epochs_for_each_task
        self.train_loss_threshold_to_stop = train_loss_threshold_to_stop

    def training_step(self, batches):
        batches_train, batches_test, task_id = batches
        print('Task ID: [bold white]{}[/bold white]'.format(task_id))

        id_prog_trial = self.progress._add_task(self.n_trials, f'Trial 0/{self.n_trials}')
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
            self.progress._update(id_prog_trial, n+1, description=f'Trial {n+1}/{self.n_trials}')

        self.progress.progress.remove_task(id_prog_trial)
        self.add_submission(task_id, results)

        n_tasks_correct = 0
        for trials in self.test_results[task_id]:
            n_tasks_correct += any(all(all(pixel == 3 for pixel in row) for row in trial['correct_pixels']) for trial in trials)

        return {
            'n_tasks_correct': n_tasks_correct,
            'n_tasks_total': len(batches_test),
        }

    def _training_step_train(self, batches_train, task_id, n):
        max_epochs_for_each_task = self.max_epochs_for_each_task
        n_tasks_total = sum([len(b[0]) for b in batches_train])
        
        total_loss = 0
        loss_prev = 0
        n_times_constant_loss = 0
        progress_id = self.progress._add_task(max_epochs_for_each_task, f'Epoch 0/{max_epochs_for_each_task}')
        
        self.model.__init__(*self.model_args, **{**self.model_kwargs, **self.params_for_each_trial[n]})
        self.model.to(batches_train[0][0].device)
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        for i in range(max_epochs_for_each_task):
            self.update_task_progress(progress_id, i+1, task_id, loss=total_loss, description=f'Epoch {i+1}/{self.max_epochs_for_each_task}')
            total_loss = 0
            self.n_tasks_correct = 0

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

            if self.n_tasks_correct == n_tasks_total and total_loss < self.train_loss_threshold_to_stop or (total_loss > 0.5 and n_times_constant_loss == 10):
                break

        self.progress.progress.remove_task(progress_id)

        return {
            'loss': total_loss,
            'n_tasks_correct': self.n_tasks_correct,
            'n_tasks_total': n_tasks_total,
        }

    def _training_step_test(self, batches_test, task_id, n):
        total_loss = 0
        self.n_tasks_correct = 0
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
                plot_xyt(x[0], y[0], t[0], correct_pixels, task_id=task_id)
            else:
                visualize_image_using_emoji(x[0], y[0], t[0], correct_pixels)

            task_result.append({
                'input': x[0].tolist(),
                'output': y_decoded[0].tolist(),
                'target': t_decoded[0].tolist(),
                'correct_pixels': correct_pixels[0].tolist(),
                'hparams': {**self.model_kwargs, **self.params_for_each_trial[n]},
            })

            print("Test {} | Correct: {} | Accuracy: {:>5.1f}% ({}/{})".format(
                i+1, '🟩' if n_correct == n_pixels else '🟥', n_correct/n_pixels*100, n_correct, n_pixels, 
            ))

        self.test_results[task_id].append(task_result)

        return {
            'loss': total_loss, 
            'n_tasks_correct': self.n_tasks_correct, 
            'n_tasks_total': len(batches_test),
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
                plot_xyt(x[0], y[0], task_id=task_id)
            else:
                visualize_image_using_emoji(x[0], y[0])

            task_result.append({
                'input': x[0].tolist(),
                'output': y_decoded[0].tolist(),
                'hparams': {**self.model_kwargs, **self.params_for_each_trial[n]},
            })

        self.test_results[task_id].append(task_result)

        return {}, outputs

    @torch.no_grad()
    def update_prediction_info(self, y, t):
        y_decoded = torch.argmax(y.detach().cpu(), dim=1).long()
        t_decoded = torch.argmax(t.detach().cpu(), dim=1).long()
        
        n_correct = (y_decoded == t_decoded).sum().int()
        n_pixels = t_decoded.numel()

        self.n_tasks_correct += sum(torch.all(y_one == t_one).item() for y_one, t_one in zip(y_decoded, t_decoded))

        return y_decoded, t_decoded, n_correct, n_pixels


class PixelEachSubstitutorRepeatL(PixelEachSubstitutorL):
    def __init__(self, max_dfs=200, max_queue=20, max_depth=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_dfs = max_dfs
        self.max_queue = max_queue
        self.max_depth = max_depth

    def training_step(self, batches):
        batches_train, batches_test, task_id = batches
        print('Task ID: [bold white]{}[/bold white]'.format(task_id))

        self.no_label = True if len(batches_test[0][1].shape) == 2 else False

        results = []
        info_train = self._training_step_train(batches_train, task_id)
        self.print_log('Train', **info_train)

        info, outputs = self._training_step_test(batches_test, task_id) if not self.no_label else self._test_step_test(batches_test, task_id)
        if not self.no_label:
            self.print_log('Test', **info)

        results.append({
            'accuracy': info_train['n_tasks_correct'] / info_train['n_tasks_total'],
            'loss': info_train['loss'],
            'outputs': outputs,
        })

        self.add_submission(task_id, results)
        self.test_results['hparams'] = {**self.model_kwargs, 'hyperparams_for_each_trial': self.params_for_each_trial},

        n_tasks_correct = 0
        for trials in self.test_results[task_id]:
            n_tasks_correct += any(all(all(pixel == 3 for pixel in row) for row in trial['correct_pixels']) for trial in trials)

        return {
            'loss': info_train['loss'],
            'n_tasks_correct': n_tasks_correct,
            'n_tasks_total': len(batches_test),
        }

    def _training_step_train(self, batches_train, task_id):
        def __print_one_step_result():
            for output_info_batch in output_info:
                for x_one, y_one, t_one, c_one in zip(*output_info_batch):
                    c_one = torch.where(c_one == 1, 3, 2)
                    visualize_image_using_emoji(x_one, y_one, t_one, c_one)
            print('Accuracy: {:.1f}% -> {:.1f}% ({:.1f}) | Depth: {} | Model {}'.format(acc_prev*100, acc_next*100, acc_prev_max*100, len(models_temp), 'O'*len(models_temp))) # [_model.id for _model in models_temp]
            # print('Queue:', ' '.join(
            #     ['{}'.format((round(acc.item()*100, 1), [_model.id for _model in _models])) \
            #     for _models, _, _, _, acc, _, _ in queue]
            # ))

        def __build_models():
            nonlocal next_id
            if len(models) == 0 or i >= 1:
                # Create new model
                varied_kwargs = self.params_for_each_trial[i-1] if i >= 1 else {}
                model = self.model_class(*self.model_args, **{**self.model_kwargs, **varied_kwargs})
                model.to(batches_train[0][0].device)
                model.id = (next_id := next_id + 1)
                models_temp = models + [model]
                # models_temp = [model] * 3
                opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            elif i == 0:
                # Continue training
                models_temp = models 
                opt = opt_prev

            return models_temp, opt

        def __check_corrected_pixels(answer_map_prev, answer_map):
            is_wrong_corrected_pixels = False
            for correct_previous_batch, correct_current_batch in zip(answer_map_prev, answer_map):
                if torch.any(correct_previous_batch - correct_current_batch) > 0:
                    is_wrong_corrected_pixels = True
                    break
            return is_wrong_corrected_pixels

        n_tasks_total = sum([len(b[0]) for b in batches_train])

        accuracy0, answer_map0 = self.get_avg_accuracy(batches_train, return_corrects_info=True)
        checkpoint0 = {
            'models': [],
            'answer_map': answer_map0,
            'opt': None,
            'acc_prev': accuracy0,
            'acc_prev_max': accuracy0,
            'checkpoint_path': None,
        }
        queue = [checkpoint0]

        id_prog_acc = self.progress._add_task(100, f'Acc      0%')
        id_prog_dfs = self.progress._add_task(self.max_dfs, 'Search 0/{}'.format(self.max_dfs))
        next_id = 0
        completed = False

        for i in range(self.max_dfs): # DFS
            self.update_task_progress(id_prog_dfs, i+1, task_id=task_id, n_queue=len(queue), depth=len(queue[0]['models']), description=f'Search {i+1}/{self.max_dfs}')
            
            queue = sorted(queue, key=lambda x: x['acc_prev_max'], reverse=True)
            checkpoint = queue.pop(0)
            models, answer_map_prev, opt_prev, acc_prev, acc_prev_max, checkpoint_path = \
                checkpoint.get('models'), \
                checkpoint.get('answer_map'), \
                checkpoint.get('opt'), \
                checkpoint.get('acc_prev'), \
                checkpoint.get('acc_prev_max'), \
                checkpoint.get('checkpoint_path')

            for i in range(self.n_trials + 1):
                models_temp, opt = __build_models()
                output_info, total_loss, acc_next, n_tasks_correct, opt = self._training_step_train_one_step(models_temp, batches_train, opt, acc_prev)

                acc_max = max([_checkpoint['acc_prev_max'] for _checkpoint in queue] + [acc_prev_max])
                self.update_task_progress(id_prog_acc, completed=acc_max*100, description=f'  Acc{acc_max*100:>5.1f}%' if acc_max != 1 else f'  Acc  100%')
                __print_one_step_result()

                answer_map = output_info[-1]
                # is_wrong_corrected_pixels = __check_corrected_pixels(answer_map_prev, answer_map)
                acc_prev_max = max(acc_prev_max, acc_next)

                if acc_next == 1 and acc_prev == acc_next:
                    completed = True
                    break
                elif acc_next >= acc_prev_max and len(models_temp) < self.max_depth:
                    kwargs = self.params_for_each_trial[i] if i < len(self.params_for_each_trial) else {}
                    model = self.model_class(*self.model_args, **{**self.model_kwargs, **kwargs})
                    model.load_state_dict(copy.deepcopy(models_temp[-1].state_dict()))
                    model.id = models_temp[-1].id # (next_id := next_id + 1)
                    models_current = [model for _ in range(len(models_temp))]

                    opt_current = torch.optim.Adam(model.parameters(), lr=self.lr)
                    opt_current.load_state_dict(copy.deepcopy(opt.state_dict()))

                    checkpoint_prev = {
                        'models': models_current,
                        'answer_map': answer_map,
                        'opt': opt_current,
                        'acc_prev': acc_next,
                        'acc_prev_max': acc_prev_max,
                        'checkpoint_path': checkpoint_path
                    }

                    os.makedirs('./output/checkpoint/', exist_ok=True)
                    checkpoint_path = './output/checkpoint/acc_{}_depth_{}_{}.pkl'.format(round(acc_next.item()*100, 1), len(models_temp), i+1)
                    pickle.dump(checkpoint_prev, open(checkpoint_path, 'wb'))

                    queue.append({
                        'models': models_temp + models_temp[-1:],
                        'answer_map': answer_map,
                        'opt': opt,
                        'acc_prev': acc_next,
                        'acc_prev_max': acc_prev_max,
                        'checkpoint_path': checkpoint_path
                    })
                elif acc_next == 1 or acc_next >= acc_prev:
                    queue.append({
                        'models': models_temp,
                        'answer_map': answer_map,
                        'opt': opt,
                        'acc_prev': acc_next,
                        'acc_prev_max': acc_prev_max,
                        'checkpoint_path': checkpoint_path
                    })
                # elif checkpoint_path is not None:
                #     print('Load from the previous checkpoint')
                #     checkpoint_prev = pickle.load(open(checkpoint_path, 'rb'))
                #     queue.append(checkpoint_prev)

            if len(queue) == 0:
                print('-'*100)
                queue.append(checkpoint0)
            if completed:
                break

        self.progress.progress.remove_task(id_prog_dfs)
        self.progress.progress.remove_task(id_prog_acc)
        self.models = models

        return {
            'loss': total_loss,
            'n_tasks_correct': n_tasks_correct,
            'n_tasks_total': n_tasks_total,
        }

    def _training_step_train_one_step(self, models, batches_train, opt, acc_prev):
        max_epochs_for_each_task = self.max_epochs_for_each_task
        id_prog_e = self.progress._add_task(max_epochs_for_each_task, f'Epoch 0/{max_epochs_for_each_task}')
    
        t_batch = [batch[1] for batch in batches_train]

        for e in range(max_epochs_for_each_task):
            output_info_batch = []
            total_loss = 0
            y_batch = []

            for x, t in batches_train:
                y = x.detach().clone()
                for model in models:
                    y = model(y)
                loss = self.loss_fn(y, t)

                opt.zero_grad()
                loss.backward()
                opt.step()

                x_decoded = torch.argmax(x, dim=1).long()
                y_decoded = torch.argmax(y, dim=1).long()
                t_decoded = torch.argmax(t, dim=1).long()

                # if (e+1) % 5 == 0:
                #     c_decoded = torch.where(y_decoded == t_decoded, 3, 2)
                #     for x_one, y_one, t_one, c_one in zip(x_decoded, y_decoded, t_decoded, c_decoded):
                #         visualize_image_using_emoji(x_one, y_one, t_one, c_one)
                #     print('-'*50)

                c_decoded = torch.where(y_decoded == t_decoded, 1, 0)
                output_info_batch.append((x_decoded, y_decoded, t_decoded, c_decoded))
                total_loss += loss
                y_batch.append(y)

            acc_next = self.get_avg_accuracy(zip(y_batch, t_batch))

            if (acc_prev < acc_next and len(models) < self.max_depth) or acc_next == 1:
                break
            
            self.update_task_progress(id_prog_e, e+1, loss=loss, description=f'Epoch {e+1}/{self.max_epochs_for_each_task}')

        self.progress.progress.remove_task(id_prog_e)
        acc_next, n_tasks_correct = self.get_avg_accuracy(zip(y_batch, t_batch), return_n_tasks_correct=True)

        return output_info_batch, total_loss, acc_next, n_tasks_correct, opt

    def _training_step_test(self, batches_test, task_id):
        total_loss = 0
        self.n_tasks_correct = 0
        task_result = []
        outputs = []

        for i, (x, t) in enumerate(batches_test):
            y = x
            for model in self.models:
                y = model(y)
            loss = self.loss_fn(y, t)
            total_loss += loss

            y_decoded = torch.argmax(y, dim=1).long()
            t_decoded = torch.argmax(t, dim=1).long()

            y_decoded, t_decoded, n_correct, n_pixels = self.update_prediction_info(y, t)
            outputs.append(y_decoded[0])

            correct_pixels = torch.where(y_decoded == t_decoded, 3, 2)
            if self.is_notebook:
                plot_xyt(x[0], y[0], t[0], correct_pixels, task_id=task_id)
            else:
                visualize_image_using_emoji(x[0], y[0], t[0], correct_pixels)

            task_result.append({
                'input': x[0].tolist(),
                'output': y_decoded[0].tolist(),
                'target': t_decoded[0].tolist(),
                'correct_pixels': correct_pixels[0].tolist(),
            })

            print("Test {} | Correct: {} | Accuracy: {:>5.1f}% ({}/{})".format(
                i+1, '🟩' if n_correct == n_pixels else '🟥', n_correct/n_pixels*100, n_correct, n_pixels, 
            ))

        self.test_results[task_id].append(task_result)

        return {
            'loss': total_loss, 
            'n_tasks_correct': self.n_tasks_correct, 
            'n_tasks_total': len(batches_test),
        }, outputs

    @staticmethod
    def get_avg_accuracy(batches, return_n_tasks_correct=False, return_corrects_info=False):
        accuracy_total = 0
        data_total = 0
        n_tasks_correct = 0
        c_decoded_batch = []

        for (y, t) in batches:
            y_decoded = y.argmax(dim=1).long()
            t_decoded = t.argmax(dim=1).long()
            c_decoded = torch.where(y_decoded == t_decoded, 1, 0)

            if return_corrects_info:
                c_decoded_batch.append(c_decoded)

            accuarcy_each = torch.sum(c_decoded) / y.argmax(dim=1)[0].numel() # [N, C, H, W]
            accuracy_total += torch.sum(accuarcy_each)
            data_total += len(y)
            n_tasks_correct += torch.sum(torch.where(accuarcy_each == 1, 1, 0))

        avg_accuracy = accuracy_total / data_total

        results = [avg_accuracy]
        if return_n_tasks_correct:
            results.append(n_tasks_correct)
        if return_corrects_info:
            results.append(c_decoded_batch)

        if len(results) == 1:
            return results[0]
        return results
