
def min_and_max_width_and_height(dataset):
    max_width = 0
    max_height = 0
    min_width = float('inf')
    min_height = float('inf')
    for task in dataset:
        for x, y in zip(task[0], task[1]):
            max_width = max(max_width, x.shape[0], y.shape[1])
            max_height = max(max_height, x.shape[0], y.shape[1])
            min_width = min(min_width, x.shape[0], y.shape[1])
            min_height = min(min_height, x.shape[0], y.shape[1])
    return max_width, max_height, min_width, min_height
