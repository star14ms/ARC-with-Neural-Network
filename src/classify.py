from utils.visualize import plot_task
from constants import COLORS
from functools import partial
from torch import tensor


class ARCDataClassifier:

    # Check if all inputs and outputs have the same shape
    def is_same_shape(xs, ys):
        return all(x.shape == y.shape for x, y in zip(xs, ys))

    # Check if all inputs and outputs have the same shape
    def is_same_shape_f(bool=True):
        if bool is True:
            return lambda xs, ys: ARCDataClassifier.is_same_shape(xs, ys)
        else:
            return lambda xs, ys: not ARCDataClassifier.is_same_shape(xs, ys)

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

        if ARCDataClassifier.is_same_shape(xs, ys):
            class_vector[0] = 1
        if ARCDataClassifier.is_same_colors(xs, ys):
            class_vector[1] = 1
        if ARCDataClassifier.is_same_number_of_colors(xs, ys):
            class_vector[2] = 1
        if ARCDataClassifier.is_same_number_of_background_colors(xs, ys):
            class_vector[3] = 1
        
        if verbose:
            print(f'Task {index}: {class_vector}')
            print('Same shape:', bool(class_vector[0]))
            print('Same colors:', bool(class_vector[1]))
            print('Same number of colors:', bool(class_vector[2]))
            print('Same number of background colors:', bool(class_vector[3]))
        
        return class_vector


    def is_n_m_colored_in_out_f(n, m):

        def is_n_colored_input_m_colored_output(xs, ys, n, m):
            for x, y in zip(xs, ys):
                colors_input = set(x.unique().tolist())
                colors_output = set(y.unique().tolist())

                if len(colors_input) != n or len(colors_output) != m:
                    return False
            return True

        return partial(is_n_colored_input_m_colored_output, n=n, m=m)


    def are_input_output_similar_f(threshold=0.9):

        def are_input_output_similar(xs, ys, threshold):
            for x, y in zip(xs, ys):
                if (x == y).sum() / x.numel() < threshold:
                    return False
            return True
        return partial(are_input_output_similar, threshold=threshold)


    def is_dominent_color_stable(xs, ys):
        for x, y in zip(xs, ys):
            x_number_of_colors = tensor([(x == i).sum() for i in range(10)])
            y_number_of_colors = tensor([(y == i).sum() for i in range(10)])
            x_dominant_color = x_number_of_colors.argmax()
            y_dominant_color = y_number_of_colors.argmax()

            if not (x_dominant_color == y_dominant_color and x_number_of_colors[x_dominant_color] == y_number_of_colors[y_dominant_color]):
                return False
        return True
    

    def is_dominent_color_stable_f(bool=True):
        if bool is True:
            return lambda xs, ys: ARCDataClassifier.is_dominent_color_stable(xs, ys)
        else:
            return lambda xs, ys: not ARCDataClassifier.is_dominent_color_stable(xs, ys)


def get_filter_funcs():
    filter_funcs = (
        ARCDataClassifier.is_same_shape_f(True),
        ARCDataClassifier.is_n_m_colored_in_out_f(2, 3),
        # ARCDataClassifier.is_dominent_color_stable_f(True),
    )
    return filter_funcs


if __name__ == '__main__':
    from rich import print

    from data import ARCDataset
    from constants import get_challenges_solutions_filepath

    verbose = True
    data_category = 'train'

    challenges, solutions = get_challenges_solutions_filepath(data_category)

    # # Example usage
    # dataset_train = ARCDataset(challenges, solutions, train=True, one_hot=False)
    # dataset_test = ARCDataset(challenges, solutions, train=False, one_hot=False)
    # print(f'Data size: {len(dataset_train)}')
    
    # dict_classes_dataset = {}
    
    # # Visualize a task
    # for index in range(len(dataset_train)):
    #     dict_classes_dataset[index] = get_class_vector(dataset_train[index], verbose=verbose)
        
    #     if verbose:
    #         plot_task(dataset_train, dataset_test, index)
            
    # for key, value in dict_classes_dataset.items():
    #     print(f'{key}: {value}')

    # Example usage
    filter_funcs = (
        ARCDataClassifier.is_same_shape_f(True),
        ARCDataClassifier.is_n_m_colored_in_out_f(2, 3),
        # ARCDataClassifier.is_dominent_color_stable_f(True),
    )

    dataset_train = ARCDataset(challenges, solutions, train=True, one_hot=False, filter_funcs=filter_funcs)
    dataset_test = ARCDataset(challenges, solutions, train=False, one_hot=False, filter_funcs=filter_funcs)
    print(f'Data size: {len(dataset_train)}')
    
    # Visualize a task
    for index in range(len(dataset_train)):
        plot_task(dataset_train, dataset_test, index, data_category)
