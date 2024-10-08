from arc.utils.visualize import plot_task
from arc.constants import COLORS
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

    
    def in_data_codes_f(codes, reorder=False, *args, **kwargs):
        def in_data_codes(xs, ys, key, *args, **kwargs):
            return True if key in codes else False
        
        return partial(in_data_codes, codes=codes, reorder=reorder, *args, **kwargs)


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


    def is_shape_size_in(xs, ys, *args, **kwargs):
        start, stop = kwargs['start'], kwargs['stop']
        range_size = range(start, stop)

        for x, y in zip(xs, ys):
            if not (x.shape[0] in range_size and 
                    x.shape[1] in range_size and 
                    y.shape[0] in range_size and 
                    y.shape[1] in range_size
                ):
                return False if kwargs['bool'] else True
        return True if kwargs['bool'] else False
    
    def is_shape_size_in_f(start, stop, bool=True, *args, **kwargs):
        return partial(ARCDataClassifier.is_shape_size_in, start=start, stop=stop, bool=bool, *args, **kwargs)


# *NNNNNNNN: Deactivate the task
tasks_fill_1 = '''\
4258a5f9 ce22a75a b60334d2 b1948b0a c8f0f002 f76d97a5 d90796e8 25d8a9c8 0d3d703e 3aa6fb7a a699fb00 0ca9ddb6 d364b489 95990924'''

tasks_fill_2 = '''\
60b61512 22233c11 3e980e27 72322fa7 88a10436 11852cab 0962bcdd 36d67576 913fb3ed 56ff96f3 e9614598 af902bf9 928ad970'''

tasks_solvable_with_3x3_kernel = '''\
25ff71a9 3618c87e 42a50994 4347f46a 50cb2852 67385a82 67a423a3 *6e02f1e3 6f8cd79b a79310a0 a9f96cdd aedd82e4 b6afb2da bb43febb d511f180 54d9e175 *6c434453 913fb3ed'''

tasks_fulid = '''\
444801d8 *f1cefba8 d4f3cd78 aba27056'''

tasks_sequential_simple_line = '''\
d9f24cd1 3bd67248 5c0a986e *7ddcd7ec'''
# 99fa7670 *d06dbe63 *f151fac 508bd3b6'''
# *d07ae81c *e21d9049 *855e0971 *bd4472b8 *264363fd *ec883f72 *25d487eb *82819916 *6d58a25d *6e19193c *d43fd935 *1f0c79e5 *b8cdaf2b *8d510a79 *41e4d17e *623ea044 *a78176bb *ea786f4a *e40b9e2f

tasks_reasoning_abs_pixels = '''\
aabf363d'''

def filter_data_codes(data_codes: list[str]):
    return tuple(filter(lambda x: len(x) == 8, data_codes.split()))


def get_filter_funcs():
    filter_funcs = (
        ARCDataClassifier.in_data_codes_f([
            # '22168020',
            # *filter_data_codes(tasks_fill_1),
            # *filter_data_codes(tasks_fill_2),
            # *filter_data_codes(tasks_solvable_with_3x3_kernel),
            *filter_data_codes(tasks_sequential_simple_line),
            # *filter_data_codes(tasks_reasoning_abs_pixels),
            # *filter_data_codes(tasks_fulid),
        ], reorder=True),
        # ARCDataClassifier.is_same_shape_f(True),
        # ARCDataClassifier.is_shape_size_in_f(start=1, stop=21),
    )
    return filter_funcs


if __name__ == '__main__':
    from rich import print

    from data import ARCDataset
    from arc.constants import get_challenges_solutions_filepath

    verbose = True
    data_category = 'train'

    challenges, solutions = get_challenges_solutions_filepath(data_category)

    # Example usage
    filter_funcs = get_filter_funcs()
    # filter_funcs = (
    #     ARCDataClassifier.is_same_shape_f(True),
    # )

    dataset = ARCDataset(challenges, solutions, one_hot=False, augment_data=False, augment_test_data=False, filter_funcs=filter_funcs)
    print(f'Data size: {len(dataset)}')

    # Visualize a task
    for index in range(len(dataset)):
        plot_task(dataset, index, data_category)
