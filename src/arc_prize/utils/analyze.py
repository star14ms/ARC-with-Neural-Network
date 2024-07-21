
def max_width_and_height(dataset):
    max_width = 0
    max_height = 0
    for i in range(len(dataset)):
        for x, y in zip(dataset[i][0], dataset[i][1]):
            max_width = max(max_width, x.shape[0], y.shape[1])
            max_height = max(max_height, x.shape[0], y.shape[1])
    return max_width, max_height
