import matplotlib.pyplot as plt
from matplotlib import colors
import os
import torch

from arc_prize.constants import COLORS


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
    norm = colors.Normalize(vmin=0, vmax=9)

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
    
    ax.set_title(title, fontweight='bold')


def plot_xyt(input_tensor, predicted_tensor, answer_tensor=None, idx=0):
    """Plots the input, predicted, and answer pairs of a specified task, using the ARC color scheme."""
    num_img = 3
    fig, axs = plt.subplots(1, num_img, figsize=(9, num_img))
    plt.suptitle(f'Task {idx}', fontsize=20, fontweight='bold', y=0.96)
    
    cmap = colors.ListedColormap(COLORS)
    norm = colors.Normalize(vmin=0, vmax=9)
    
    input_tensor = input_tensor.detach().cpu().squeeze(0)
    predicted_tensor = predicted_tensor.detach().cpu().squeeze(0)
    answer_tensor = answer_tensor.detach().cpu().squeeze(0) if answer_tensor is not None else None
    
    if len(input_tensor.shape) == 3:
        input_tensor = torch.argmax(input_tensor, dim=0).long()
    if len(predicted_tensor.shape) == 3:
        predicted_tensor = torch.argmax(predicted_tensor, dim=0).long()
    if answer_tensor is not None and len(answer_tensor.shape) == 3:
        answer_tensor = torch.argmax(answer_tensor, dim=0).long()

    plot_single_image(input_tensor, axs[0], 'Input', cmap, norm)
    plot_single_image(predicted_tensor, axs[1], 'Predicted', cmap, norm)
    if answer_tensor is not None:
        plot_single_image(answer_tensor, axs[2], 'Answer', cmap, norm)
    
    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor('#dddddd')

    fig.tight_layout()
    plt.show()


def plot_xyts(task_result, title_prefix="Task"):
    """Plots rows of input, predicted, and answer triples for a set of tasks, using the ARC color scheme."""
    num_pairs = len(task_result)
    num_columns = 3
    fig, axs = plt.subplots(num_columns, num_pairs, figsize=(num_columns*6, 10))
    
    task_result = [(
        x.detach().cpu().squeeze(0).long(),
        y.detach().cpu().squeeze(0).long(),
        t.detach().cpu().squeeze(0).long(),
    ) for x, y, t in task_result]

    task_result = [(
        torch.argmax(x, dim=0).long() if len(x.shape) > 2 else x, 
        torch.argmax(y, dim=0).long() if len(y.shape) > 2 else y,
        torch.argmax(t, dim=0).long() if len(t.shape) > 2 else t,
    ) for x, y, t in task_result]

    # If there's only one task, axs may not be a 2D array
    if num_pairs == 1:
        axs = [axs]  # Make it 2D array

    cmap = colors.ListedColormap(COLORS)
    norm = colors.Normalize(vmin=0, vmax=9)
    
    for i in range(num_pairs):
        plot_single_image(task_result[i][0], axs[0][i], f'{title_prefix} Inputs' if i == 0 else '', cmap, norm)
        plot_single_image(task_result[i][1], axs[1][i], f'{title_prefix} Predictions' if i == 0 else '', cmap, norm)
        if isinstance(task_result[i][2], type(task_result[i][0])):
            plot_single_image(task_result[i][2], axs[2][i], f'{title_prefix} Answers' if i == 0 else '', cmap, norm)
    
    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor('#dddddd')
    
    plt.tight_layout()
    plt.show()


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


def visualize_image_using_emoji(*images):
    '''
    ⬛️ = 0, 🟦 = 1, 🟥 = 2, 🟩 = 3, 🟨 = 4, ⬜️ = 5, 🟪 = 6, 🟧 = 7, ⏹️ = 8, 🟫 = 9
    '''
    images = [image.squeeze(0).detach().cpu() for image in images]
    images = [torch.argmax(image, dim=0).long() if len(image.shape) > 2 else image for image in images]
    
    print()
    for h in range(max(images, key=lambda x: x.shape[0]).shape[0]):
        line = ''
        for image in images:
            if h >= image.shape[0]:
                line += image.shape[1] * '  ' + '  '
                continue
            for w in range(image.shape[1]):
                pixel_key = image[h, w].item()
                if pixel_key == 0:
                    line += '⬛️'
                elif pixel_key == 1:
                    line += '🟦'
                elif pixel_key == 2:
                    line += '🟥'
                elif pixel_key == 3:
                    line += '🟩'
                elif pixel_key == 4:
                    line += '🟨'
                elif pixel_key == 5:
                    line += '⬜️'
                elif pixel_key == 6:
                    line += '🟪'
                elif pixel_key == 7:
                    line += '🟧'
                elif pixel_key == 8:
                    line += '⏹️ '
                elif pixel_key == 9:
                    line += '🟫'
                else:
                    line += '❌'
            line += '  '
        print(line)
