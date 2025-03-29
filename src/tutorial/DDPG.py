import torch
from torch import nn
from tqdm import tqdm
from copy import deepcopy
import sys
sys.path.append('./src/')

from arc.constants import get_challenges_solutions_filepath
from classify import ARCDataClassifier
from data import ARCDataset
from arc.model.substitute.v2_CL_encode import PixelEachSubstitutor
from arc.utils.visualize import visualize_image_using_emoji
from arc.preprocess import one_hot_encode


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.layers(x)

    
class ActorCritics():
    def __init__(self, W_kernel_max=3, H_kernel_max=3, q_hidden_dim=32, n_colors=10):
        self.pi = PixelEachSubstitutor(
            n_range_search=1, W_kernel_max=W_kernel_max, H_kernel_max=H_kernel_max, 
            emerge_color=True, vec_abs=False,
            C_dims_encoded=[10, 3], L_dims_encoded=[9, 4], L_dims_decoded=[9, 1],
            # pad_class_initial=-1
        )
        self.q = QNetwork(W_kernel_max*H_kernel_max, action_dim=n_colors, hidden_dim=q_hidden_dim)


# class MSBELoss(nn.Module):
#     def __init__(self):
#         super(MSBELoss, self).__init__()
#         self.mse_loss = nn.MSELoss()

#     def forward(self, q_values, rewards, next_q_values, dones, gamma=0.99):
#         """
#         Computes the Mean-Squared Bellman Error (MSBE) loss.

#         Args:
#             q_values (torch.Tensor): Predicted Q-values for the current state-action pairs.
#             rewards (torch.Tensor): Observed rewards for taking the actions.
#             next_q_values (torch.Tensor): Predicted Q-values for the next states.
#             dones (torch.Tensor): Boolean tensor (1 if the episode terminated, 0 otherwise).
#             gamma (float): Discount factor.

#         Returns:
#             torch.Tensor: Computed MSBE loss.
#         """
#         target_q_values = rewards + gamma * next_q_values * (1 - dones)
#         loss = self.mse_loss(q_values, target_q_values.detach())  # Detach to prevent backprop through target
#         return loss



def forward_pi(x, t, max_recursion, acc_max_list, i):
    y_prev = x.clone()
    acc_inital = (x.argmax(dim=1) == t.argmax(dim=1)).sum().item() / (x_size:= x.size(2) * x.size(3))
    acc_max = acc_inital

    for depth in range(max_recursion):
        if depth != 0:
            y_prev = y
            max_indices = torch.argmax(y_prev, dim=1)
            y_prev = torch.scatter(torch.zeros_like(y_prev), 1, max_indices.unsqueeze(1), 1)
        
        y = pi(y_prev) # [N, C, H, W]

        n_pixel_correct = (y.argmax(dim=1) == t.argmax(dim=1)).sum().item()
        acc = n_pixel_correct / x_size
        # is_new_correct_pixel = torch.any(correct.int() - correct_inital.int() == 1)

        if acc > acc_max_list[i]:
            acc_max_list[i] = acc
            y_changed = torch.where(y.argmax(dim=1) != y_prev.argmax(dim=1), 1, 0) # (N, H, W)
            y_changed_green = torch.where(y_changed == 1, 3, 0)
            visualize_image_using_emoji(x[0], t[0], y[0], one_hot_encode(y_changed_green[0]), titles=['Input', 'Target', 'Output', 'Changed'])
        
        if acc < acc_max or n_pixel_correct == x_size or torch.all(y_prev.argmax(dim=1) == y.argmax(dim=1)):
            break

        if acc > acc_max:
            acc_max = acc
            
    return y, n_pixel_correct, x_size, depth


def get_state_action_pair(pi, x, t, max_recursion):
    y_prev = x.clone()
    acc_inital = (x.argmax(dim=1) == t.argmax(dim=1)).sum().item() / (x_size:= x.size(2) * x.size(3))
    acc_max = acc_inital

    for depth in range(max_recursion):
        if depth != 0:
            y_prev = y
            max_indices = torch.argmax(y_prev, dim=1)
            y_prev = torch.scatter(torch.zeros_like(y_prev), 1, max_indices.unsqueeze(1), 1)
        
        observations, y = pi(y_prev, return_observation=True) # [N, C, H, W]

        n_pixel_correct = (y.argmax(dim=1) == t.argmax(dim=1)).sum().item()
        acc = n_pixel_correct / x_size
        # is_new_correct_pixel = torch.any(correct.int() - correct_inital.int() == 1)
        
        if acc < acc_max or n_pixel_correct == x_size or torch.all(y_prev.argmax(dim=1) == y.argmax(dim=1)):
            break

        if acc > acc_max:
            acc_max = acc
            
    return observations, y


def compute_loss_q(ac: ActorCritics, x, t, max_recursion, acc_max_list, i, r, o2, d):
    o, a = get_state_action_pair(ac.pi, x, t, max_recursion)
    Q_value = ac.q(o, a)

    # Bellman backup for Q function
    with torch.no_grad():
        q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
        backup = r + gamma * (1 - d) * q_pi_targ

    # MSE loss against Bellman backup
    loss_q = ((Q_value - backup)**2).mean()

    return loss_q


def train(xs_train, ts_train, xs_test, ts_test, ac: ActorCritics, alpha, gamma, polyak=0.995, n_iteration=50, max_recursion=30):
    optimizer_pi = torch.optim.Adam(ac.pi.parameters(), lr=0.01)
    optimizer_q = torch.optim.Adam(ac.q.parameters(), lr=0.01)

    loss_fn = nn.CrossEntropyLoss()
    acc_max_list = [(x.argmax(dim=1) == t.argmax(dim=1)).sum().item() / (x.size(1) * x.size(2)) for x, t in zip(xs_train, ts_train)]
    n_iters_perfect = 0

    for e in (pbar:= tqdm(range(n_iteration))):
        n_task_correct = 0

        o2 = torch.tensor([])

        for i, (x, t) in enumerate(zip(xs_train, ts_train)):
            x = x.unsqueeze(0)
            t = t.unsqueeze(0)

            # First run one gradient descent step for Q.
            optimizer_q.zero_grad()
            loss_q, loss_info = compute_loss_q(ac, x, t, max_recursion, acc_max_list, i, r, o2, d)
            loss_q.backward()
            optimizer_q.step()
            
            # Freeze Q-network so you don't waste computational effort 
            # computing gradients for it during the policy learning step.
            for p in ac.q.parameters():
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            optimizer_pi.zero_grad()
            y, n_pixel_correct, x_size, depth = forward_pi(x, t, max_recursion, acc_max_list, i)
            loss = loss_fn(y, t)
            loss.backward()
            optimizer_pi.step()
            
            # Unfreeze Q-network so you can optimize it at next DDPG step.
            for p in ac.q.parameters():
                p.requires_grad = True
                
            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

            if n_pixel_correct == x_size:
                n_task_correct += 1

        pbar.set_description(f"Loss: {loss.item():.4f}, Recursion: {depth+1}, Correct Task: {n_task_correct}/{len(xs_train)}")

        if n_task_correct == len(xs_train):
            break

    return pi


def test(pi, xs_test, ts_test, max_recursion=30, show_depth=False):
    n_correct_tasks = 0
    for x, t in zip(xs_test, ts_test):
        x = x.unsqueeze(0)
        t = t.unsqueeze(0)
        y_prev = x.clone()
        
        ys = []

        for depth in range(max_recursion):
            if depth != 0:
                y_prev = y
                max_indices = torch.argmax(y_prev, dim=1)
                y_prev = torch.scatter(torch.zeros_like(y_prev), 1, max_indices.unsqueeze(1), 1)
            
            y = pi(y_prev)
            ys.append((y, 'Depth {}'.format(depth+1)))
            c_decoded = torch.where(y.argmax(dim=1) == t.argmax(dim=1), 3, 2)

        str_visualization = ''
        if len(ys) > 1 and show_depth:
            xytc = [(x, 'Input')] + ys + [(t, 'Target')] + [(c_decoded, 'Correct')]
            xytc_batches = [xytc[i:i+4] for i in range(0, len(xytc), 4)]
            for xytc_batch in xytc_batches:
                titles = [title for _, title in xytc_batch]
                xytcs = [xytc for xytc, _ in xytc_batch]
                str_visualization += visualize_image_using_emoji(*xytcs, titles=titles, return_str=True)
        else:
            str_visualization = visualize_image_using_emoji(x, y, t, c_decoded, return_str=True)
        print(str_visualization)

        if torch.all(y.argmax(dim=1) == t.argmax(dim=1)):
            n_correct_tasks += 1
            
    print(f"Correct Tasks: {n_correct_tasks}/{len(xs_test)}")


if __name__ == '__main__':
    alpha = 0.1 # Learning Rate
    gamma = 0.9 # Discount Factor
    polyak = 0.995
    n_iteration = 300
    n_recursion = 30

    data_category = 'train'
    task_id = '3bd67248'
    challenges, solutions = get_challenges_solutions_filepath(data_category)

    filter_funcs = (ARCDataClassifier.in_data_codes_f([task_id], reorder=True),)
    dataset = ARCDataset(challenges, solutions, one_hot=True, augment_data=False, augment_test_data=False, filter_funcs=filter_funcs)

    torch.set_printoptions(precision=2, sci_mode=False)
    xs_train, ts_train, xs_test, ts_test, task_id = dataset[0]

    ac = ActorCritics(3, 3)
    ac_targ = deepcopy(ac)

    pi = train(xs_train, ts_train, xs_test, ts_test, ac, alpha, gamma, polyak, n_iteration)
    test(pi, xs_test, ts_test, n_recursion)
