import torch


alpha = 0.1 # Learning Rate
gamma = 0.9 # Discount Factor
n_iteration = 40
max_iteration = 10

reward_map = torch.tensor([
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -10, 10]
])
action_mask = torch.tensor([
    [[0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 0, 1]],
    [[0, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1]],
    [[0, 1, 1, 0], [1, 1, 1, 0], [1, 0, 1, 0]],
], dtype=torch.float32)
actions_probabilities = action_mask / action_mask.sum(dim=-1, keepdim=True)

action_map = ['Left', 'Right', 'Up', 'Down']


def train(Q_table, actions_probabilities, reward_map, alpha, gamma, n_iteration=40):
    for _ in range(n_iteration):
        state = [0, 0]

        while state != [2, 2]:
            # Choose next action
            next_state = state.copy()
            action = behavior_policy(state, actions_probabilities)

            if action == 0:
                next_state[1] -= 1
            elif action == 1:
                next_state[1] += 1
            elif action == 2:
                next_state[0] -= 1
            elif action == 3:
                next_state[0] += 1

            # Get reward
            reward = reward_func(next_state, reward_map)

            # Update Q-table
            Q_value = reward + gamma * torch.max(Q_table[*next_state])
            TD_error = Q_value - Q_table[*state, action]
            Q_table[*state, action] += alpha * TD_error

            # Update state
            state = next_state
            
    return Q_table


def test(Q_table, action_map, action_mask, max_iteration=30):
    # Test
    state = [0, 0]
    for _ in range(max_iteration):
        Q_table *= action_mask
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
            
        if state == [2, 2]:
            print('Success')
            break
    else:
        print('Fail')


def reward_func(s, reward_map):
    return reward_map[s[0], s[1]]


def behavior_policy(s, actions_probabilities):
    action_probabilities = actions_probabilities[s[0], s[1]]
    action = torch.distributions.Categorical(action_probabilities).sample().item()

    return action


if __name__ == '__main__':
    torch.set_printoptions(precision=2, sci_mode=False)

    Q_table = torch.randn([*reward_map.shape, len(action_map)])
    Q_table_inital = Q_table.clone().reshape(-1, 4)
    print('Q table before training', Q_table.reshape(-1, 4), sep='\n', end='\n\n')

    Q_table = train(Q_table, actions_probabilities, reward_map, alpha, gamma, n_iteration)

    print('\nQ table after training', Q_table.reshape(-1, 4), sep='\n', end='\n\n')
    print('Q table change', Q_table.reshape(-1, 4) - Q_table_inital, sep='\n', end='\n\n')

    test(Q_table, action_map, action_mask, max_iteration)
