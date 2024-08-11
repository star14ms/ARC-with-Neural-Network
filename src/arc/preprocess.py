import torch
from arc.constants import COLORS


def one_hot_encode(matrix, n_class=len(COLORS), cold_value=0, last_dim_ones=False, device=None):
    # Ensure the input is a tensor
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix)
    
    # Get the dimensions of the input matrix
    H, W = matrix.shape
    
    # Create a one-hot encoded tensor with shape [n_class, H, W]
    if cold_value == 0:
        one_hot_matrix = torch.zeros(n_class, H, W, device=device)
    else:
        one_hot_matrix = torch.full([n_class, H, W], cold_value, dtype=torch.float32, device=device)
    
    # Use scatter_ to fill in the one-hot encoded tensor
    one_hot_matrix.scatter_(0, matrix.unsqueeze(0), 1)

    # Last dim filled with ones
    if last_dim_ones:
        one_hot_matrix = torch.cat([one_hot_matrix, torch.ones(1, H, W)], dim=0)

    return one_hot_matrix


def one_hot_encode_changes(x, t, n_class=len(COLORS)):
    # Ensure the inputs are tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)

    # Get the dimensions of the input matrices
    N, M = x.shape

    # Create one-hot encoded tensors with shape [n_class, N, M]
    source = torch.zeros((N, M))
    target_one_hot = torch.zeros((n_class, N, M))

    # Find the indices where x and t differ
    source = change_indices = (x != t)

    # Get the values at the differing positions
    target_values = t[change_indices].long()

    # Fill the one-hot encoded tensors
    target_one_hot[target_values, change_indices] = 1  # Mark target based on values in t

    return source.float(), target_one_hot


def reconstruct_t_from_one_hot(x, target_one_hot):
    # Ensure the inputs are tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(target_one_hot, torch.Tensor):
        target_one_hot = torch.tensor(target_one_hot)

    x = x.squeeze(0)
    target_one_hot = target_one_hot.squeeze(0)

    # Initialize t as a copy of x
    t_reconstructed = x.clone()

    # Find the indices where target_one_hot is 1
    target_indices = torch.argmax(target_one_hot, dim=0)
    
    # Update t with the target values
    t_reconstructed[target_one_hot.sum(dim=0).bool()] = target_indices[target_one_hot.sum(dim=0).bool()]

    return t_reconstructed



def one_hot_encode_shape(matrix, h_max=30, w_max=30):
    shape = torch.tensor(matrix.shape)
    H = shape[:1]
    W = shape[1:]
    
    one_hot_H = torch.zeros(h_max).scatter_(0, H, 1)
    one_hot_W = torch.zeros(w_max).scatter_(0, W, 1)

    return torch.cat([one_hot_H, one_hot_W], dim=0)
