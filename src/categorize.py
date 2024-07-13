from data import ARCDataset
from utils import plot_task

from constants import COLORS


# Check if all inputs and outputs have the same shape
def is_same_shape(xs, ys):
    return all(x.shape == y.shape for x, y in zip(xs, ys))

# Check if all colors appeared in the input present in the output
def is_same_colors(xs, ys):
    for x, y in zip(xs, ys):
        colors_input = set(x.unique().tolist())
        colors_output = set(y.unique().tolist())
        
        if colors_input != colors_output:
            return False
    return True

# Check if the number of colors is the same in the input and output
def is_same_number_of_colors(xs, ys):
    for x, y in zip(xs, ys):
        colors_input = set(x.unique().tolist())
        colors_output = set(y.unique().tolist())
        
        if len(colors_input) != len(colors_output):
            return False
    return True

# Check if the number of background colors is the same in the input and output
def is_same_number_of_background_colors(xs, ys):
    for x, y in zip(xs, ys):
        bincount_input = x.flatten().to(int).bincount(minlength=len(COLORS)).tolist()
        bincount_output = y.flatten().to(int).bincount(minlength=len(COLORS)).tolist()

        n_pixels_background_color_input = max(bincount_input)
        n_pixels_background_color_output = max(bincount_output)

        if n_pixels_background_color_input != n_pixels_background_color_output:
            return False
    return True


def get_class_vector(data_item, verbose=False):
    '''Classify the data based on different criteria.

    Returns:
        class_vector (list): A list of 4 elements, each element is a boolean indicating if the task satisfies the criteria.
            meaning of each element:
    - 1: Same shape
    - 2: Same colors
    - 3: Same number of colors
    - 4: Same number of background colors
    '''
    xs, ys = data_item
    n_dim = 4
    class_vector = [0] * n_dim

    if is_same_shape(xs, ys):
        class_vector[0] = 1
    if is_same_colors(xs, ys):
        class_vector[1] = 1
    if is_same_number_of_colors(xs, ys):
        class_vector[2] = 1
    if is_same_number_of_background_colors(xs, ys):
        class_vector[3] = 1
    
    if verbose:
        print(f'Task {index}: {class_vector}')
        print('Same shape:', bool(class_vector[0]))
        print('Same colors:', bool(class_vector[1]))
        print('Same number of colors:', bool(class_vector[2]))
        print('Same number of background colors:', bool(class_vector[3]))
    
    return class_vector


if __name__ == '__main__':
    from rich import print

    verbose = True
    base_path = './data/arc-prize-2024/'

    # Reading files
    challenges = base_path + 'arc-agi_training_challenges.json'
    solutions = base_path + 'arc-agi_training_solutions.json'
    # challenges = base_path + 'arc-agi_evaluation_challenges.json'
    # solutions = base_path + 'arc-agi_evaluation_solutions.json'
    # challenges = base_path + 'arc-agi_test_challenges.json'
    # solutions = None

    # Example usage
    dataset_train = ARCDataset(challenges, solutions, train=True)
    dataset_test = ARCDataset(challenges, solutions, train=False)
    print(f'Data size: {len(dataset_train)}')
    
    dict_classes_dataset = {}
    
    # Visualize a task
    for index in range(len(dataset_train)):
        dict_classes_dataset[index] = get_class_vector(dataset_train[index], verbose=verbose)
        
        if verbose:
            plot_task(dataset_train, dataset_test, index)
            
    # for key, value in dict_classes_dataset.items():
    #     print(f'{key}: {value}')
