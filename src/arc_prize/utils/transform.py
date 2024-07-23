from collections import defaultdict
import torch


def collate_fn_same_shape(task, batch_size_max=4, shuffle=True):
    """
    Collate function for the DataLoader.

    Args:
        batch: List of tuples of the form (image, target).

    Returns:
        Tuple of the form (images, targets).
    """
    xs, ts = task[0]
    
    if not all(x.shape == t.shape for x, t in zip(xs, ts)) and not all(x.shape[1:] == t.shape for x, t in zip(xs, ts)):
        return xs, ts
      
    if shuffle:
        indices = torch.randperm(len(xs)).tolist()
        xs = [xs[i] for i in indices]
        ts = [ts[i] for i in indices]

    shape_dict = defaultdict(list)
    for i, (x, t) in enumerate(zip(xs, ts)):
        shape_dict[x.shape].append(i)
    
    xs_new = []
    ts_new = []
    for shape, indices in shape_dict.items():
        n_piece = 1
        batch_size = len(indices)
        
        while batch_size > batch_size_max:
            n_piece += 1
            batch_size = len(indices) // n_piece

        for i in range(0, len(indices), batch_size):
            images = [xs[j] for j in indices[i:i+batch_size]]
            targets = [ts[j] for j in indices[i:i+batch_size]]
            xs_new.append(torch.stack(images))
            ts_new.append(torch.stack(targets))

        if len(indices) % batch_size == 0:
            continue

        images = [xs[j] for j in indices[i+batch_size:]]
        targets = [ts[j] for j in indices[i+batch_size:]]
        xs_new.append(torch.stack(images))
        ts_new.append(torch.stack(targets))
        
    return xs_new, ts_new
