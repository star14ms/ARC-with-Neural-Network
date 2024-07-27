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
    xs, ts, xs_test, ts_test, task_id = task[0]

    if batch_size_max == 1 or (
            not all(x.shape == t.shape for x, t in zip(xs, ts)) and \
            not all(x.shape[1:] == t.shape for x, t in zip(xs, ts))
        ):
        return \
            [x.unsqueeze(0) for x in xs], \
            [t.unsqueeze(0) for t in ts], \
            [x.unsqueeze(0) for x in xs_test], \
            [t.unsqueeze(0) for t in ts_test], \
            task_id

    if shuffle:
        indices = torch.randperm(len(xs)).tolist()
        xs = [xs[i] for i in indices]
        ts = [ts[i] for i in indices]

    shape_dict = defaultdict(list)
    for i, (x, t) in enumerate(zip(xs, ts)):
        shape_dict[x.shape].append(i)
    
    xs_new = []
    ts_new = []
    for _, indices in shape_dict.items():
        n_piece = 1
        batch_size = len(indices)
        
        while batch_size > batch_size_max:
            n_piece += 1
            batch_size = len(indices) // n_piece

        xs_same_shape = [torch.stack([xs[j] for j in indices[i:i+batch_size]]) for i in range(0, len(indices), batch_size)]
        ts_same_shape = [torch.stack([ts[j] for j in indices[i:i+batch_size]]) for i in range(0, len(indices), batch_size)]

        if len(indices) % batch_size != 0:
            xs_same_shape.append(torch.stack([xs[j] for j in indices[-len(indices) % batch_size:]]))
            ts_same_shape.append(torch.stack([ts[j] for j in indices[-len(indices) % batch_size:]]))

        xs_new.extend(xs_same_shape)
        ts_new.extend(ts_same_shape)
        
    return xs_new, ts_new, [x.unsqueeze(0) for x in xs_test], [t.unsqueeze(0) for t in ts_test], task_id
