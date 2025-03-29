import torch


def convert_targets(x, y, t):
    """
    Args:
        x: torch.Tensor of shape [N, V] - input features
        y: torch.Tensor of shape [N, 1] - predicted outputs
        t: torch.Tensor of shape [N, 1] - target values
    
    Returns:
        t_modified: torch.Tensor of shape [N, 1] - modified target values
    """
    # Create a copy of t to modify
    t_modified = t.clone()
    
    # Compute pairwise equality of x vectors
    x_i = x.unsqueeze(1)  # [N, 1, V]
    x_j = x.unsqueeze(0)  # [1, N, V]
    x_equal = torch.all(x_i == x_j, dim=-1)  # [N, N]
    
    # Compare y with t to find correct predictions
    correct_preds = (y == t).float()  # [N, 1]
    
    # For each sample i
    for i in range(x.shape[0]):
        # Find indices j where x[j] == x[i]
        same_x_indices = x_equal[i]  # [N]
        
        # Get corresponding y values and correctness for these indices
        y_at_same_x = y[same_x_indices]  # [M, 1] where M is number of matches
        correct_at_same_x = correct_preds[same_x_indices]  # [M, 1]
        
        # If there exists any correct prediction among same x values
        if torch.any(correct_at_same_x):
            # Get one of the correct y values (first one found)
            # correct_ys = y_at_same_x[correct_at_same_x.bool().squeeze()]
            t_at_same_x = t[same_x_indices]
            is_t_zero_most = torch.sum(t_at_same_x == 0) > 0.9 * t_at_same_x.size(0)

            # If prediction is wrong (y[i] != t[i])
            if y[i] == t[i]:
                if y[i] == 0 and torch.any(t_at_same_x != 0) and not is_t_zero_most:
                    # Update t[i] to match the correct prediction
                    t_modified[i] = t_at_same_x[t_at_same_x != 0][0] # exclude 0 values
            else:
                # check if more than 90% of the t values are zero:
                if is_t_zero_most:
                    t_modified[i] = 0
                else:
                    t_modified[i] = t_at_same_x[t_at_same_x != 0][0]
    
    return t_modified


def create_mask(x_obs: torch.Tensor, x_flatten: torch.Tensor, y_flatten: torch.Tensor, t_flatten: torch.Tensor):
    """
    Args:
        x_obs: torch.Tensor of shape [N, V] - input features
        y: torch.Tensor of shape [N, 1] - predicted outputs
        t: torch.Tensor of shape [N, 1] - target values

    Returns:
        mask: torch.Tensor of shape [N, 1] - binary mask
    """

    # Initialize mask with ones
    mask = torch.ones_like(y_flatten, dtype=torch.float32, requires_grad=True)

    # Compare y_flatten with t_flatten
    x_y_diff = (x_flatten != y_flatten).float()  # [N, 1]
    y_t_diff = (y_flatten != t_flatten).float()  # [N, 1]

    # Compute pairwise equality of x_obs vectors
    # Expand x_obs to [N, 1, V] and [1, N, V] for broadcasting
    x_i = x_obs.unsqueeze(1)  # [N, 1, V]
    x_j = x_obs.unsqueeze(0)  # [1, N, V]
    x_equal = torch.all(x_i == x_j, dim=-1)  # [N, N]
    
    # Compare y_flatten values pairwise
    t_i = t_flatten.unsqueeze(1)  # [N, 1, 1]
    t_j = t_flatten.unsqueeze(0)  # [1, N, 1]
    t_diff = (t_i != t_j).float() # [N, N, 1]

    # For each i, check if there exists any j where:
    # 1. x_obs[i] == x_obs[j]
    # 2. t_flatten[i] != t_flatten[j]
    conflicting_pairs = (x_equal.unsqueeze(-1) * t_diff)  # [N, N, 1]
    has_conflict = torch.any(conflicting_pairs, dim=1)  # [N, 1]

    # Final mask: 0 where y[i] != t_flatten[i] AND there exists conflicting x[j], y[j]
    mask = mask * (1 - (x_y_diff * has_conflict))  # [N, 1]
    # mask = mask * (1 - (x_y_diff * y_t_diff * has_conflict))  # [N, 1]

    return mask
