import torch
from tqdm import tqdm
import sys
sys.path.append('./src/')

from arc.constants import get_challenges_solutions_filepath
from arc.utils.visualize import visualize_image_using_emoji
from classify import ARCDataClassifier
from data import ARCDataset


alpha = 0.1 # Learning Rate
gamma = 0.9 # Discount Factor
n_iteration = 300
max_iteration = 30

action_map = ['Left', 'Right', 'Up', 'Down']


def train(x_train, Q_table, action_mask, reward_map, initial_state, final_state, alpha, gamma, n_iteration=40):
    for _ in tqdm(range(n_iteration)):
        y_train = x_train.argmax(dim=0).clone()
        state = initial_state.copy()
        action_mask_temp = action_mask.clone()

        while state != final_state:
            # Choose next action
            next_state = state.copy()
            action = behavior_policy(state, action_mask_temp)

            if action == 0:
                next_state[1] -= 1
            elif action == 1:
                next_state[1] += 1
            elif action == 2:
                next_state[0] -= 1
            elif action == 3:
                next_state[0] += 1
                
            if next_state != final_state:
                y_train[*next_state] = 8
            action_mask_temp[*next_state, action+1 if action in [0, 2] else action-1] = 0

            # Get reward
            reward = reward_func(next_state, reward_map)

            # Update Q-table
            Q_value = reward + gamma * torch.max(Q_table[*next_state])
            TD_error = Q_value - Q_table[*state, action]
            Q_table[*state, action] += alpha * TD_error

            # Update state
            state = next_state

        # visualize_image_using_emoji(x_train, y_train, titles=['Input', 'Output'])

    return Q_table


def test(x_test, t_test, Q_table, action_map, inital_state, final_state, action_mask, max_iteration=30):
    # Test
    state = inital_state
    y_test = x_test.argmax(dim=0).clone()

    for _ in range(max_iteration):
        Q_table[action_mask == 0] = -float('inf')
        action = torch.argmax(Q_table[*state])
        print(state, action_map[action])

        if action == 0:
            state[1] -= 1
        elif action == 1:
            state[1] += 1
        elif action == 2:
            state[0] -= 1
        elif action == 3:
            state[0] += 1

        if state == final_state:
            print('Success')
            break

        y_test[*state] = 8
    else:
        print('Fail')
        
    visualize_image_using_emoji(x_test, y_test, t_test, titles=['Input', 'Output', 'Target'])

     
def reward_func(s, reward_map):
    return reward_map[s[0], s[1]]


def behavior_policy(s, action_mask):
    action_probability = action_mask / action_mask.sum(dim=-1, keepdim=True)
    action_probabilities = action_probability[s[0], s[1]]
    action = torch.distributions.Categorical(action_probabilities).sample().item()

    return action


def get_q_learning_infos_from_data(xs_train, ts_train, inital_class=2, terminal_class=3):
    reward_maps = []
    inital_states = []
    final_states = []
    action_masks = []
    action_probabiliies = []
    for x, t in zip(xs_train, ts_train):
        C, H, W = x.shape

        x = x.argmax(dim=0)
        inital_state = [int(coord) for coord in torch.where(x == inital_class)]
        final_state = [int(coord) for coord in torch.where(x == terminal_class)]

        reward_map = torch.full_like(x, -1)
        reward_map[*final_state] = 100

        # define action masks to restrict the agent to move only in the valid directions
        action_mask = torch.ones([H, W, 4])
        action_mask[0, :, 2] = 0
        action_mask[-1, :, 3] = 0
        action_mask[:, 0, 0] = 0
        action_mask[:, -1, 1] = 0
        action_probability = action_mask / action_mask.sum(dim=-1, keepdim=True)

        reward_maps.append(reward_map)
        inital_states.append(inital_state)
        final_states.append(final_state)
        action_masks.append(action_mask)
        action_probabiliies.append(action_probability)

    return reward_maps, inital_states, final_states, action_masks, action_probabiliies


if __name__ == '__main__':
    data_category = 'train'
    task_id = 'a2fd1cf0'
    challenges, solutions = get_challenges_solutions_filepath(data_category)

    filter_funcs = (ARCDataClassifier.in_data_codes_f([task_id], reorder=True),)
    dataset = ARCDataset(challenges, solutions, one_hot=True, augment_data=False, augment_test_data=False, filter_funcs=filter_funcs)

    torch.set_printoptions(precision=2, sci_mode=False)
    xs_train, ts_train, xs_test, ts_test, task_id = dataset[0]

    reward_maps, inital_states, final_states, action_masks, action_probabiliies = get_q_learning_infos_from_data(xs_test, ts_test)
    reward_map, inital_state, final_state, action_mask, action_probability = reward_maps[0], inital_states[0], final_states[0], action_masks[0], action_probabiliies[0]

    Q_table = torch.randn([*reward_map.shape, len(action_map)])
    Q_table_inital = Q_table.clone().reshape(-1, 4)
    # print('Q table before training', Q_table.reshape(-1, 4), sep='\n', end='\n\n')

    Q_table = train(xs_test[0], Q_table, action_mask, reward_map, inital_state, final_state, alpha, gamma, n_iteration)

    # print('\nQ table after training', Q_table.reshape(-1, 4), sep='\n', end='\n\n')
    # print('Q table change', Q_table.reshape(-1, 4) - Q_table_inital, sep='\n', end='\n\n')

    test(xs_test[0], ts_test[0], Q_table, action_map, inital_state, final_state, action_mask, max_iteration)
