import matplotlib.pyplot as plt
from matplotlib import colors

from constants import COLORS


def plot_task(dataset_train, dataset_test, idx):
    """Plots the train and test pairs of a specified task, using the ARC color scheme."""
    task_key = dataset_train.task_key(idx)  # Get the task ID
    train_inputs, train_outputs = dataset_train[idx]  # Load the first task
    test_inputs, test_outputs = dataset_test[idx]  # Load the first task
    
    num_train = len(train_inputs)
    num_test = len(test_inputs)
    num_total = num_train + num_test
    
    fig, axs = plt.subplots(2, num_total, figsize=(3*num_total, 3*2))
    plt.suptitle(f'Set #{idx}, {task_key}:', fontsize=20, fontweight='bold', y=0.96)
    
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
    plt.show()


def plot_input_predicted_answer(input_tensor, predicted_tensor, answer_tensor, idx=0):
    """Plots the input, predicted, and answer pairs of a specified task, using the ARC color scheme."""
    
    def _plot_one(matrix, ax, title, cmap, norm):
        ax.imshow(matrix, cmap=cmap, norm=norm)
        ax.grid(True, which='both', color='lightgrey', linewidth = 0.5)
        
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        ax.set_xticks([x-0.5 for x in range(1 + len(matrix[0]))])     
        ax.set_yticks([x-0.5 for x in range(1 + len(matrix))])
        
        ax.set_title(title, fontweight='bold')
    
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    plt.suptitle(f'Task {idx}', fontsize=20, fontweight='bold', y=0.96)
    
    cmap = colors.ListedColormap(COLORS)
    norm = colors.Normalize(vmin=0, vmax=9)
    
    _plot_one(input_tensor, axs[0], 'Input', cmap, norm)
    _plot_one(predicted_tensor, axs[1], 'Predicted', cmap, norm)
    _plot_one(answer_tensor, axs[2], 'Answer', cmap, norm)
    
    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor('#dddddd')

    fig.tight_layout()
    fig.subplots_adjust(top=0.76)
    plt.show()


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
