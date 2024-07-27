from arc_prize.utils.visualize import plot_task
from arc_prize.constants import COLORS
from functools import partial
from torch import tensor
import torch


class ARCDataClassifier:

    # Check if all inputs and outputs have the same shape
    def is_same_shape(xs, ys, *args, **kwargs):
        return all(x.shape == y.shape for x, y in zip(xs, ys))

    def is_same_shape_f(bool=True):
        if bool is True:
            return lambda *args, **kwargs: ARCDataClassifier.is_same_shape(*args, **kwargs)
        else:
            return lambda *args, **kwargs: not ARCDataClassifier.is_same_shape(*args, **kwargs)

    # Check if all colors appeared in the input present in the output
    def is_same_colors(xs, ys, *args, **kwargs):
        for x, y in zip(xs, ys):
            colors_input = set(x.unique().tolist())
            colors_output = set(y.unique().tolist())
            
            if colors_input != colors_output:
                return False
        return True
    
    def is_same_colors_f(bool=True):
        if bool is True:
            return lambda *args, **kwargs: ARCDataClassifier.is_same_colors(*args, **kwargs)
        else:
            return lambda *args, **kwargs: not ARCDataClassifier.is_same_colors(*args, **kwargs)

    # Check if the number of colors is the same in the input and output
    def is_same_number_of_colors(xs, ys, *args, **kwargs):
        for x, y in zip(xs, ys):
            colors_input = set(x.unique().tolist())
            colors_output = set(y.unique().tolist())
            
            if len(colors_input) != len(colors_output):
                return False
        return True

    def is_same_number_of_colors_f(bool=True):
        if bool is True:
            return lambda *args, **kwargs: ARCDataClassifier.is_same_number_of_colors(*args, **kwargs)
        else:
            return lambda *args, **kwargs: not ARCDataClassifier.is_same_number_of_colors(*args, **kwargs)

    # Check if the number of background colors is the same in the input and output
    def is_same_number_of_background_colors(xs, ys, *args, **kwargs):
        for x, y in zip(xs, ys):
            bincount_input = x.flatten().int().bincount(minlength=len(COLORS)).tolist()
            bincount_output = y.flatten().int().bincount(minlength=len(COLORS)).tolist()

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


    def is_n_m_colored_in_out_f(n, m, *args, **kwargs):
        def is_n_colored_input_m_colored_output(xs, ys, *args, **kwargs):
            for x, y in zip(xs, ys):
                colors_input = set(x.unique().tolist())
                colors_output = set(y.unique().tolist())

                if len(colors_input) != n or len(colors_output) != m:
                    return False
            return True

        return partial(is_n_colored_input_m_colored_output, n=n, m=m, *args, **kwargs)


    def are_input_output_similar_f(threshold=0.9, *args, **kwargs):

        def are_input_output_similar(xs, ys, *args, **kwargs):
            for x, y in zip(xs, ys):
                if (x == y).sum() / x.numel() < threshold:
                    return False
            return True
        return partial(are_input_output_similar, *args, **kwargs)


    def is_dominent_color_stable(xs, ys, *args, **kwargs):
        for x, y in zip(xs, ys):
            x_number_of_colors = tensor([x.eq(i).sum() for i in range(10)])
            y_number_of_colors = tensor([y.eq(i).sum() for i in range(10)])
            x_dominant_color = x_number_of_colors.argmax()
            y_dominant_color = y_number_of_colors.argmax()

            if not (x_dominant_color == y_dominant_color and x_number_of_colors[x_dominant_color] == y_number_of_colors[y_dominant_color]):
                return False
        return True

    def is_dominent_color_stable_f(bool=True):
        if bool is True:
            return lambda *args, **kwargs: ARCDataClassifier.is_dominent_color_stable(*args, **kwargs)
        else:
            return lambda *args, **kwargs: not ARCDataClassifier.is_dominent_color_stable(*args, **kwargs)

    
    def in_data_codes_f(codes, *args, **kwargs):
        def in_data_codes(xs, ys, key, *args, **kwargs):
            return True if key in codes else False
        
        return partial(in_data_codes, *args, **kwargs)


    def is_same_number_of_pixels_of_one_color(xs, ys, *args, **kwargs):
        for x, y in zip(xs, ys):
            x_number_of_colors = tensor([x.eq(i).sum() for i in range(10)])
            y_number_of_colors = tensor([y.eq(i).sum() for i in range(10)])
            if not (x_number_of_colors == y_number_of_colors).any():
                return False
        return True
    
    def is_same_number_of_pixels_of_one_color_f(bool=True, *args, **kwargs):
        if bool is True:
            return lambda *args, **kwargs: ARCDataClassifier.is_same_number_of_pixels_of_one_color(*args, **kwargs)
        else:
            return lambda *args, **kwargs: not ARCDataClassifier.is_same_number_of_pixels_of_one_color(*args, **kwargs)


    def is_n_color_stable(xs, ys, *args, **kwargs):
        for x, y in zip(xs, ys):
            x_indexes_colors = [torch.where(x == i) for i in range(10)]
            y_indexes_colors = [torch.where(y == i) for i in range(10)]

            n_color_stable = [
                len(x_indexes_colors[i][0]) != 0 and \
                len(x_indexes_colors[i][0]) == len(y_indexes_colors[i][0]) and \
                all(x_indexes_colors[i][0] == y_indexes_colors[i][0]) and \
                all(x_indexes_colors[i][1] == y_indexes_colors[i][1]) \
                    for i in range(10)
            ]
            n_color_stable = sum(n_color_stable)

            if n_color_stable < kwargs['n']:
                return False
        return True
    
    def is_n_color_stable_f(n, *args, **kwargs):
        return partial(ARCDataClassifier.is_n_color_stable, n=n, *args, **kwargs)


def get_filter_funcs():
    filter_funcs = (
        ARCDataClassifier.in_data_codes_f(['00d62c1b']),
        # ARCDataClassifier.is_same_shape_f(True),
        # ARCDataClassifier.is_n_color_stable_f(1),
        # ARCDataClassifier.is_same_colors_f(False),
        # ARCDataClassifier.is_dominent_color_stable_f(True)
    )
    return filter_funcs


if __name__ == '__main__':
    from rich import print

    from data import ARCDataset
    from arc_prize.constants import get_challenges_solutions_filepath

    verbose = True
    data_category = 'train'

    challenges, solutions = get_challenges_solutions_filepath(data_category)

    # Example usage
    # filter_funcs = get_filter_funcs()
    filter_funcs = (
        # ARCDataClassifier.in_data_codes_f(['00d62c1b']),
        ARCDataClassifier.is_same_shape_f(True),
        ARCDataClassifier.is_n_color_stable_f(1),
        ARCDataClassifier.is_same_colors_f(False),
        ARCDataClassifier.is_dominent_color_stable_f(True)
    )

    dataset = ARCDataset(challenges, solutions, one_hot=False, filter_funcs=filter_funcs)
    print(f'Data size: {len(dataset)}')

    # Visualize a task
    for index in range(len(dataset)):
        plot_task(dataset, index, data_category)
