import torch
from constants import COLORS


def one_hot_encode(matrix, num_classes=len(COLORS)):
    # Ensure the input is a tensor
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix)
    
    # Get the dimensions of the input matrix
    N, M = matrix.shape
    
    # Create a one-hot encoded tensor with shape [num_classes, N, M]
    one_hot_matrix = torch.zeros(num_classes, N, M)
    
    # Use scatter_ to fill in the one-hot encoded tensor
    one_hot_matrix.scatter_(0, matrix.unsqueeze(0), 1)
    
    return one_hot_matrix


def one_hot_encode_changes(x, t, num_classes=len(COLORS)):
    # Ensure the inputs are tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)

    # Get the dimensions of the input matrices
    N, M = x.shape

    # Create one-hot encoded tensors with shape [num_classes, N, M]
    source = torch.zeros((N, M))
    target_one_hot = torch.zeros((num_classes, N, M))

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

    # Get the dimensions of the input matrices
    N, M = x.shape

    # Initialize t as a copy of x
    t_reconstructed = x.clone()

    # Find the indices where target_one_hot is 1
    target_indices = torch.argmax(target_one_hot, dim=0)
    
    # Update t with the target values
    t_reconstructed[target_one_hot.sum(dim=0).bool()] = target_indices[target_one_hot.sum(dim=0).bool()]

    return t_reconstructed
