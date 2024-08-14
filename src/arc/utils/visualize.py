import matplotlib.pyplot as plt
from matplotlib import colors
import os
import torch
import json
from rich import print

from arc.utils.print import is_notebook
from arc.constants import COLORS

import warnings


def plot_task(dataset, idx, data_category, fdir_to_save=None):
    """Plots the train and test pairs of a specified task, using the ARC color scheme."""

    def plot_one(input_matrix, ax, train_or_test, input_or_output, cmap, norm):
        ax.imshow(input_matrix, cmap=cmap, norm=norm)
        ax.grid(True, which='both', color='lightgrey', linewidth = 0.5)
        
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])     
        ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
        
        if train_or_test == 'test' and input_or_output == 'output':
            ax.set_title('TEST OUTPUT', color='green', fontweight='bold')
        else:
            ax.set_title(train_or_test + ' ' + input_or_output, fontweight='bold')

    train_inputs, train_outputs, test_inputs, test_outputs, task_id = dataset[idx]  # Load the first task
    
    num_train = len(train_inputs)
    num_test = len(test_inputs)
    num_total = num_train + num_test
    
    fig, axs = plt.subplots(2, num_total, figsize=(3*num_total, 3*2))
    plt.suptitle(f'{data_category.capitalize()} Set #{idx+1}, {task_id}:', fontsize=20, fontweight='bold', y=0.96)
    
    cmap = colors.ListedColormap(COLORS)
    norm = colors.Normalize(vmin=0, vmax=len(COLORS)-1)

    for j in range(num_train):
        plot_one(train_inputs[j], axs[0, j], 'train', 'input', cmap, norm)
        plot_one(train_outputs[j], axs[1, j], 'train', 'output', cmap, norm)

    for j in range(num_test):
        plot_one(test_inputs[j], axs[0, j + num_train], 'test', 'input', cmap, norm)
        if test_outputs != []:
            plot_one(test_outputs[j], axs[1, j + num_train], 'test', 'output', cmap, norm)
        else:
            plot_one([[5]], axs[1, j + num_train], 'test', 'output', cmap, norm)

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')  # substitute 'k' for black
    fig.patch.set_facecolor('#dddddd')

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    
    if fdir_to_save is not None:
        fname = os.path.join(fdir_to_save, '{}_{}.png'.format(idx+1, task_id))
        plt.savefig(fname)
        plt.close()
        print('{} saved'.format(fname))
    else:
        plt.show()


def plot_single_image(matrix, ax, title, cmap, norm):
    ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth = 0.5)
    
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(matrix[0]))])     
    ax.set_yticks([x-0.5 for x in range(1 + len(matrix))])
    
    ax.set_title(title, fontweight='bold', color='black' if not title.lower().startswith('test') else 'green')


def plot_xytc(*images, task_id=None, titles=['Input', 'Predicted', 'Answer', 'Correct'], change_order_021=False, savefile=None):
    """
    Plots the input, predicted, and answer pairs of a specified task, using the ARC color scheme.

    Parameters:
        `images`: `list` # [x, y, ...]
        `task_id`: `int`
        `titles`: `list`
    """
    num_img = len(images)
    
    fig, axs = plt.subplots(1, num_img, figsize=(len(images)*3, 3))
    if task_id is not None:
        plt.suptitle(f'Task {task_id}', fontsize=20, fontweight='bold', y=0.96)

    # If there's only one image, axs may not be an array
    if num_img == 1:
        axs = [axs]  # Make it array

    cmap = colors.ListedColormap(COLORS)
    norm = colors.Normalize(vmin=0, vmax=9)
    
    images = [image.detach().cpu().squeeze(0).long() if isinstance(image, torch.Tensor) else torch.tensor(image) for image in images if image is not None]
    images = [torch.argmax(image, dim=0) if len(image.shape) > 2 else image for image in images]

    for i in range(num_img):
        if num_img > 2 and num_img < 5 and change_order_021:
            i = 2 if i == 1 else 1 if i == 2 else i
        plot_single_image(images[i], axs[i], titles[i], cmap, norm)
    
    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor('#dddddd')
    fig.tight_layout()
    
    if savefile:
        plt.savefig(f'{savefile}.png', bbox_inches='tight')
    plt.show()


def plot_xytc_list(task_result, task_id=None, subtitles=['Inputs', 'Outputs', 'Answers', 'Corrects'], test_idx=None, change_order_021=False, savefile=None):
    """
    Plots rows of input, predicted, and answer triples for a set of tasks, using the ARC color scheme.

    Parameters:
        `task_result`: `list` # [task1, task2, ...] # task1 = [(x1, y1, ...), (x2, y2, ...), ...]
        `task_id`: `int`
        `subtitles`: `list`
        `savefile`: `str`
        `test_idx`: `int`
    """
    num_pairs = len(task_result)
    num_columns = len(task_result[0])

    fig, axs = plt.subplots(num_columns, num_pairs, figsize=(num_columns*6, 10))
    if task_id is not None:
        plt.suptitle(f'Task {task_id}', fontsize=20, fontweight='bold', y=0.96)

    task_result = [tuple(image.detach().cpu().squeeze(0) if isinstance(image, torch.Tensor) else torch.tensor(image) for image in images) for images in task_result]
    task_result = [tuple(torch.argmax(image, dim=0) if len(image.shape) > 2 else image for image in images) for images in task_result]

    # If there's only one task, axs may not be a 2D array
    if num_pairs == 1 and num_columns == 1:
        axs = [[axs]]
    elif num_pairs == 1:
        axs = [[axs_row] for axs_row in axs]

    cmap = colors.ListedColormap(COLORS)
    norm = colors.Normalize(vmin=0, vmax=len(COLORS)-1)

    for i in range(num_pairs):
        for j in range(num_columns):
            if num_columns > 2 and num_columns < 5 and change_order_021:
                j = 2 if j == 1 else 1 if j == 2 else j
            plot_single_image(task_result[i][j], axs[j][i], subtitles[j] if i == 0 else 'Test' if i >= test_idx and j == 0 else '', cmap, norm)

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor('#dddddd')
    plt.tight_layout()

    if savefile:
        plt.savefig(f'{savefile}.png', bbox_inches='tight')
    plt.show()


def plot_xytc_from_json(file_path='./output/test_results.json', titles=['Input', 'Output', 'Answer', 'Correct'], keys_json=['input', 'output', 'target', 'correct_pixels'], plot_only_correct=False, top_k=2, total=400, change_order_021=True, verbose=False):
    results = json.load(open(file_path, 'r'))
    print('Hyperparameters:', results['hparams'])
    results = results['results']

    exist_label = list(results.values())[0][0][0].get('target') is not None
    assert not plot_only_correct or exist_label, 'The results do not contain answer.'
    if len(list(results.values())[0][0]) < top_k:
        warnings.warn(f'Top-k is set to {top_k} but the number of trials is {len(list(results.values())[0][0])}, less than {top_k}.')

    if exist_label:
        if change_order_021:
            titles[1], titles[2] = titles[2], titles[1]
            keys_json[1], keys_json[2] = keys_json[2], keys_json[1]

        task_ids_correct = [key for key, task_result in results.items() if \
            all(any(
                all(all(pixel == 3 for pixel in row) for row in trial['correct_pixels'])
                    for trial in trials[:top_k if len(trials) >= top_k else len(trials)]) for trials in task_result)
        ]
        print('N Submittable: {} | N Total: {} | N Correct: {} | Accuracy: {:.2f}%'. format(len(results), total, len(task_ids_correct), len(task_ids_correct)/total*100))
    else:
        print('N Submittable: {} | N Total: {}'.format(len(results), total))

    for task_id, task_result in results.items():
        if plot_only_correct and task_id not in task_ids_correct:
            continue

        for i, trials in enumerate(task_result):
            for j, trial in enumerate(trials):
                if j == top_k:
                    break

                if verbose:
                    print(f'Task {task_id} | Test {i+1}')

                images = [trial[key] for key in keys_json if key in trial]
                plot_xytc(*images, task_id=task_id, titles=titles)


def plot_kernels_and_outputs(y, kernels):
    # Plotting the output tensor and kernels
    y = y.detach().numpy()

    # Total number of kernels
    num_kernels = y.shape[1]
    kernels_per_page = 256

    # Calculate the number of pages needed
    num_pages = (num_kernels + kernels_per_page - 1) // kernels_per_page

    for page in range(num_pages):
        fig, axes = plt.subplots(16, 32, figsize=(32, 16))
        fig.suptitle(f'Kernel and Output Pairs - Page {page+1} (Shape: {y.shape})', fontsize=16, fontweight='bold')

        for i in range(kernels_per_page):
            kernel_idx = page * kernels_per_page + i
            if kernel_idx >= num_kernels:
                break
            
            row = i // 16
            col_kernel = (i % 16) * 2
            col_output = col_kernel + 1
            
            # Plot the kernel
            ax_kernel = axes[row, col_kernel]
            kernel_img = kernels[kernel_idx, 0, :, :]
            ax_kernel.imshow(kernel_img, cmap='gray')
            ax_kernel.set_title(f'Kernel {kernel_idx}', fontsize=6, color='red')
            ax_kernel.axis('off')

            # Plot the output
            ax_output = axes[row, col_output]
            output_img = y[0, kernel_idx, :, :]
            ax_output.imshow(output_img, cmap='gray')
            ax_output.set_title(f'Output {kernel_idx}', fontsize=6, color='black')
            ax_output.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()


def print_image_with_probs(*images):
    for h in range(max(images, key=lambda x: x.shape[0]).shape[0]):
        for image in images:
            if h >= image.shape[0]:
                print('  ', end='')
                continue
            for w in range(image.shape[1]):
                pixel_prob = image[h, w].item()
                if pixel_prob < 0.01:
                    print('⬛️', end='')
                elif pixel_prob < 0.05:
                    print('🟫', end='')
                elif pixel_prob < 0.1:
                    print('🟥', end='')
                elif pixel_prob < 0.3:
                    print('🟧', end='')
                elif pixel_prob < 0.5:
                    print('🟨', end='')
                elif pixel_prob < 0.7:
                    print('🟩', end='')
                elif pixel_prob < 0.95:
                    print('🟦', end='')
                elif pixel_prob < 0.99:
                    print('🟪', end='')
                else:
                    print('⬜️', end='')
            print('  ', end='')
        print()
    print()


def visualize_image_using_emoji(*images, titles=['Input', 'Output', 'Answer', 'Correct'], output_file=None, emoji_map=['⬛️', '🟦', '🟥', '🟩', '🟨', '⬜️', '🟪', '🟧', '⏹️ ', '🟫', '◽️'], padding_idx=-1):
    '''
    Parameters:
        `emoji_map`: `list`
        Defulat: ⬛️ = 0, 🟦 = 1, 🟥 = 2, 🟩 = 3, 🟨 = 4, ⬜️ = 5, 🟪 = 6, 🟧 = 7, ⏹️ = 8, 🟫 = 9, '◽️' = 10
    '''

    def add_space(emoji):
        if emoji == '◽' and output_file:
            return '⬛️'
        elif emoji == '⏹️' and (not output_file or is_ipython):
            return emoji + ' '
        else:
            return emoji

    images = [image.squeeze(0).detach().cpu() for image in images]
    images = [torch.argmax(image, dim=0).long() if len(image.shape) > 2 else image for image in images]
    is_ipython = is_notebook()

    n_lines = max(images, key=lambda x: x.shape[0]).shape[0]

    line = ''
    if titles:
        for title, image_width in zip(titles, [image.shape[1] for image in images]):
            line += title.ljust(image_width * 2) + '  '
        line += '\n'

    for h in range(n_lines):
        for image in images:
            if h >= image.shape[0]:
                line += image.shape[1] * '  ' + '  '
                continue
            for w in range(image.shape[1]):
                pixel_key = image[h, w].item()
                if pixel_key < len(emoji_map):
                    line += add_space(emoji_map[pixel_key])
                else:
                    line += add_space(emoji_map[padding_idx])
            line += '  '
        line += '\n' if h != n_lines - 1 else ''

    if output_file:
        with open(output_file, 'a') as f:
            f.write(line + '\n')
    else:
        print(line)
