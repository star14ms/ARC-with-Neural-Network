
COLORS = [
  '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
  '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]


def get_challenges_solutions_filepath(data_category, base_path='./data/arc-prize-2024/'):
    '''
    Parameters:
    - data_category: str: `train`, `val`, or `test`
    '''
    if data_category == 'train':
        challenges = base_path + 'arc-agi_training_challenges.json'
        solutions = base_path + 'arc-agi_training_solutions.json'
    elif data_category == 'val':
        challenges = base_path + 'arc-agi_evaluation_challenges.json'
        solutions = base_path + 'arc-agi_evaluation_solutions.json'
    elif data_category == 'test':
        challenges = base_path + 'arc-agi_test_challenges.json'
        solutions = None
    else:
        raise ValueError(f'Invalid data category: {data_category}')
    
    return challenges, solutions