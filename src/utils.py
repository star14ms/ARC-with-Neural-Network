import matplotlib.pyplot as plt
from matplotlib import colors

  
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
    
    cmap = colors.ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                                  '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    for j in range(num_train):
        plot_one(train_inputs[j], axs[0, j], 'train', 'input', cmap, norm)
        plot_one(train_outputs[j], axs[1, j], 'train', 'output', cmap, norm)

    for j in range(num_test):
        plot_one(test_inputs[j], axs[0, j + num_train], 'test', 'input', cmap, norm)
        plot_one(test_outputs[j], axs[1, j + num_train], 'test', 'output', cmap, norm)

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')  # substitute 'k' for black
    fig.patch.set_facecolor('#dddddd')

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
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
