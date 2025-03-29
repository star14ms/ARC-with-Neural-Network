import torch
from torch import nn
from tqdm import tqdm
import sys
sys.path.append('./src/')

from arc.constants import get_challenges_solutions_filepath
from classify import ARCDataClassifier
from data import ARCDataset
from arc.model.components.pixel_vector_extractor import PixelVectorExtractor
from arc.model.substitute.v2_CL_encode import Encoder, Reasoner, Decoder, PixelEachSubstitutor
from arc.utils.visualize import visualize_image_using_emoji


class BehaviorPolicyLocation(nn.Module):
    def __init__(
            self, 
            n_range_search=-1, 
            memory_channel=False, 
            W_kernel_max=61, 
            H_kernel_max=61, 
            C_dims_encoded=[10, 2], 
            L_dims_encoded=[9, 4], 
            L_num_layers=1, L_n_head=None, L_dim_feedforward=1, 
            C_num_layers=1, C_n_head=None, C_dim_feedforward=1, 
            dropout=0.1, 
            n_class=10
        ):
        super().__init__()
        assert n_range_search != -1 and W_kernel_max >= 1 + 2*n_range_search and H_kernel_max >= 1 + 2*n_range_search

        self.encoder = Encoder(
            C_dims_encoded=C_dims_encoded,
            L_dims_encoded=L_dims_encoded,
            L_dim_feedforward=L_dim_feedforward,
            C_dim_feedforward=C_dim_feedforward,
            memory_channel=memory_channel,
            n_class=n_class,
            dropout=dropout,
            bias=False,
        )

        self.reasoner = Reasoner(
            VC_dim=C_dims_encoded[-1],
            VL_dim=L_dims_encoded[-1],
            L_num_layers=L_num_layers,
            L_n_head=L_n_head,
            L_dim_feedforward=L_dim_feedforward,
            C_num_layers=C_num_layers,
            C_n_head=C_n_head,
            C_dim_feedforward=C_dim_feedforward,
            dropout=dropout,
            bias=False,
        )

        # self.decoder = LocationDecoder(
        #     VC_dim=C_dims_encoded[-1], 
        #     L_dim_feedforward=1, 
        #     dropout=0.1, 
        #     bias=False,
        # )

        self.ff_c = nn.Linear(C_dims_encoded[-1], 1)
        self.ff_l = nn.Linear(L_dims_encoded[-1], 1)

    def forward(self, x, xs=None, **kwargs):
        N, H, W, C, VHVW = x.shape
        x = x.view(N*H*W, C, VHVW)

        x_VC_VL, x_VC_L, x_VC, x_C = self.encoder(x, xs)
        mem = self.reasoner(x_VC_VL)
        # y = self.decoder(x_VC_L, x_VC_VL, mem)
        y = self.ff_c(mem.transpose(1, 2)).transpose(1, 2)
        y = self.ff_l(y)

        return y.view(1, N*H*W)


class BehaviorPolicyColor(nn.Module):
    def __init__(
            self, 
            n_range_search=-1, 
            emerge_color=False, 
            memory_channel=False, 
            W_kernel_max=61, 
            H_kernel_max=61, 
            C_dims_encoded=[10, 2], 
            L_dims_encoded=[9, 4], 
            L_dims_decoded=[9, 1], 
            L_num_layers=1, L_n_head=None, L_dim_feedforward=1, 
            C_num_layers=1, C_n_head=None, C_dim_feedforward=1, 
            dropout=0.1, 
            n_class=10
        ):
        super().__init__()
        assert n_range_search != -1 and W_kernel_max >= 1 + 2*n_range_search and H_kernel_max >= 1 + 2*n_range_search

        self.encoder = Encoder(
            C_dims_encoded=C_dims_encoded,
            L_dims_encoded=L_dims_encoded,
            L_dim_feedforward=L_dim_feedforward,
            C_dim_feedforward=C_dim_feedforward,
            memory_channel=memory_channel,
            n_class=n_class,
            dropout=dropout,
            bias=False,
        )

        self.reasoner = Reasoner(
            VC_dim=C_dims_encoded[-1],
            VL_dim=L_dims_encoded[-1],
            L_num_layers=L_num_layers,
            L_n_head=L_n_head,
            L_dim_feedforward=L_dim_feedforward,
            C_num_layers=C_num_layers,
            C_n_head=C_n_head,
            C_dim_feedforward=C_dim_feedforward,
            dropout=dropout,
            bias=False,
        )

        self.decoder = Decoder(
            VL_dim=L_dims_encoded[-1],
            VC_dim=C_dims_encoded[-1],
            L_dim=L_dims_encoded[0],
            C_dim=C_dims_encoded[0],
            L_dims_decoded=L_dims_decoded,
            emerge_color=emerge_color,
            L_dim_feedforward=L_dim_feedforward,
            C_dim_feedforward=C_dim_feedforward,
            dropout=dropout,
            bias=False,
        )

    def forward(self, x, xs=None, **kwargs):
        NHW, C, VHVW = x.shape
        
        x_VC_VL, x_VC_L, x_VC, x_C = self.encoder(x, xs)
        mem = self.reasoner(x_VC_VL)
        y = self.decoder(x, mem, x_VC_VL, x_VC_L, x_VC, x_C)
        y = y.softmax(dim=1).view(-1, C)

        return y


def train(state_initial, state_final, xs_train, vec_extractor, n_action_space_location, n_action_space_color, behavior_policy_l, behavior_policy_c, alpha, gamma, n_iteration=50, n_recursion=30):
    optimizer_c = torch.optim.Adam(behavior_policy_c.parameters(), lr=0.01)
    optimizer_l = torch.optim.Adam(behavior_policy_l.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn_l = nn.BCEWithLogitsLoss()

    for _ in tqdm(range(n_iteration)):
        state = state_initial.clone()
        N, C, H, W = state.size()
        observation = vec_extractor(state)
        NHW, C, VHVW = observation.size()
        observation = observation.reshape(N, H, W, C, VHVW)

        for _ in range(n_recursion):
            # Choose next action
            probs_l = behavior_policy_l(observation)
            action_l = torch.distributions.Categorical(probs_l.softmax(dim=0)).sample().item()
            action_l = (action_l // W, action_l % W)
            # action_l = behavior_policy_location(state, n_action_space_location)

            observation_C = observation[0, action_l[0], action_l[1]].unsqueeze(0)
            probs_c = behavior_policy_c(observation_C)
            
            action_c = torch.distributions.Categorical(probs_c).sample().item()
            action = (action_l, action_c)

            reward_l = reward_func_location(state, state_final, action, sn=None)
            label_l = (state.argmax(dim=1) != state_final.argmax(dim=1)).view(-1, H*W).float()
            # label_l = torch.zeros(N, H*W)
            # label_l[0, action_l[0]*W + action_l[1]] = 1 if reward_l == 1 else 0
            loss_l = loss_fn_l(probs_l, label_l)
            optimizer_l.zero_grad()
            loss_l.backward()
            optimizer_l.step()
            if reward_l == -1:
                continue

            loss_c = loss_fn(probs_c, state_final[:, :, action_l[0], action_l[1]])
            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()

            next_state = state.clone()
            next_state[0, :, action_l[0], action_l[1]] = 0
            next_state[0, action_c, action_l[0], action_l[1]] = 1

            # # Get reward
            reward_c = reward_func_color(state, state_final, action, sn=None)

            # # Update Q-table
            # Q_value = reward_l + gamma * torch.max(Q_table[next_state[0], next_state[1]])
            # TD_error = Q_value - Q_table[state[0], state[1], action]
            # Q_table[state[0], state[1], action] += alpha * TD_error

            # Update state
            state = next_state

            visualize_image_using_emoji(state.argmax(dim=1))
            print('Location:', True if reward_l == 1 else False, loss_l.item())
            print('Color', True if reward_c == 1 else False, loss_c.item())

    return behavior_policy_l, behavior_policy_c


def test(Q_table, action_map, max_iteration=30):
    # Test
    state = [0, 0]
    for _ in range(max_iteration):
        action = torch.argmax(Q_table[state[0], state[1]]).item()
        print(state, action_map[action])
        if action == 0:
            state[1] -= 1
        elif action == 1:
            state[1] += 1
        elif action == 2:
            state[0] -= 1
        elif action == 3:
            state[0] += 1
            
        if state == [2, 2]:
            print('Success')
            break
    else:
        print('Fail')


def reward_func_location(s, sf, a, sn=None):
    action_l = a[0]
    if s[0, :, action_l[0], action_l[1]].argmax() == sf[0, :, action_l[0], action_l[1]].argmax():
        return -1
    else:
        return 1


def reward_func_color(s, sf, a, sn=None):
    action_l = a[0]
    action_c = a[1]
    if action_c == sf[0, :, action_l[0], action_l[1]].argmax().item():
        return 1
    else:
        return -1


def behavior_policy_location(s, n_action_space_location):
    action_probabilities = torch.ones(n_action_space_location) / (s.shape[2] * s.shape[3])
    action = torch.distributions.Categorical(action_probabilities).sample().item()
    
    # convert 1d index to 2d index
    action = [action // s.size(3), action % s.size(3)]

    return action


def behavior_policy_color(s, n_action_space_color):
    action_probabilities = torch.ones(n_action_space_color)
    action = torch.distributions.Categorical(action_probabilities).sample().item()

    return action


if __name__ == '__main__':
    alpha = 0.1 # Learning Rate
    gamma = 0.9 # Discount Factor
    n_iteration = 100
    n_recursion = 30
    n_range_search = 1

    data_category = 'train'
    task_id = 'a2fd1cf0'
    challenges, solutions = get_challenges_solutions_filepath(data_category)

    filter_funcs = (ARCDataClassifier.in_data_codes_f([task_id], reorder=True),)
    dataset = ARCDataset(challenges, solutions, one_hot=True, augment_data=False, augment_test_data=False, filter_funcs=filter_funcs)

    torch.set_printoptions(precision=2, sci_mode=False)
    xs_train, ts_train, xs_test, ts_test, task_id = dataset[0]
    state_initial = xs_train[0].unsqueeze(0)
    state_final = ts_train[0].unsqueeze(0)
    
    N, C, H, W = state_initial.size()

    n_action_space_location = H * W
    n_action_space_color = 10

    vec_extractor = PixelVectorExtractor(n_range_search=n_range_search, W_kernel_max=3, H_kernel_max=3, vec_abs=False)
    behavior_policy_l = BehaviorPolicyLocation(n_range_search=n_range_search)
    behavior_policy_c = BehaviorPolicyColor(n_range_search=n_range_search, emerge_color=True)

    behavior_policy_l, behavior_policy_c = train(
        state_initial, state_final, xs_train, 
        vec_extractor, n_action_space_location, n_action_space_color, 
        behavior_policy_l, behavior_policy_c, alpha, gamma, n_iteration, 
    )
