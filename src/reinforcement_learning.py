import json
import torch


def train(state_initial, state_final, n_action_space_location, n_action_space_color, Q_table_l, Q_table_c, alpha, gamma, n_iteration=40):
    for _ in range(n_iteration):
        state = state_initial.clone()

        while state != [2, 2]:
            # Choose next action
            action_l = behavior_policy_location(state, n_action_space_location)
            action_c = behavior_policy_color(state, n_action_space_color)
            action = (action_l, action_c)

            next_state = state.clone()
            next_state[action_l[0], action_l[1]] = action_c

            # Get reward
            reward_l = reward_func_location(state, state_final, action, sn=None)
            reward_c = reward_func_color(state, state_final, action, sn=None)

            # # Update Q-table
            # Q_value = reward_l + gamma * torch.max(Q_table[next_state[0], next_state[1]])
            # TD_error = Q_value - Q_table[state[0], state[1], action]
            # Q_table[state[0], state[1], action] += alpha * TD_error

            # Update state
            state = next_state
            print(state, reward_l, reward_c)
            input()

    return Q_table_l, Q_table_c


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
    if s[action_l[0], action_l[1]] == sf[action_l[0], action_l[1]]:
        return -1
    else:
        return 1


def reward_func_color(s, sf, a, sn=None):
    action_l = a[0]
    action_c = a[1]
    if action_c == sf[action_l[0], action_l[1]]:
        return 1
    else:
        return -1


def behavior_policy_location(s, n_action_space_location):
    action_probabilities = torch.ones(n_action_space_location) / len(s.flatten())
    action = torch.distributions.Categorical(action_probabilities).sample().item()
    
    # convert 1d index to 2d index
    action = [action // s.size(1), action % s.size(1)]

    return action


def behavior_policy_color(s, n_action_space_color):
    action_probabilities = torch.ones(n_action_space_color)
    action = torch.distributions.Categorical(action_probabilities).sample().item()

    return action


if __name__ == '__main__':
    alpha = 0.1 # Learning Rate
    gamma = 0.9 # Discount Factor
    n_iteration = 40
    data_path = './data/arc-prize-2024/arc-agi_training_challenges.json'
    task_id = 'a2fd1cf0'

    torch.set_printoptions(precision=2, sci_mode=False)
    data = json.load(open(data_path))
    task = data[task_id]['train'][0]

    state_initial = torch.tensor(task['input'])
    state_final = torch.tensor(task['output'])
    
    n_action_space_location = state_initial.size().numel()
    n_action_space_color = 10

    Q_table_l = torch.randn([16, n_action_space_location])
    Q_table_c = torch.randn([16, n_action_space_color])
    # Q_table_l_inital = Q_table_l.reshape(-1, 4).clone()
    # Q_table_c_inital = Q_table_c.reshape(-1, 4).clone()
    # print('Q table before training', Q_table_l.reshape(-1, 4), sep='\n', end='\n\n')

    Q_table_l = train(state_initial, state_final, n_action_space_location, n_action_space_color, Q_table_l, Q_table_c, alpha, gamma, n_iteration=40)
    Q_table_l = Q_table_l.reshape(-1, 4)

    # print('\nQ table after training', Q_table_l, sep='\n', end='\n\n')
    # print('Q table change', Q_table_l - Q_table_l_inital, sep='\n', end='\n\n')

    # test(Q_table_l, action_space_location)
