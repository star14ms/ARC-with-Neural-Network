import torch
from torch.utils.data import Dataset
import json

from utils import plot_task


class ARCDataset(Dataset):
    def __init__(self, challenge_json, solution_json=None, train=True):
        # Load challenge and solution data
        with open(challenge_json, 'r') as file:
            self.challenges = json.load(file)
        if solution_json is not None:
            with open(solution_json, 'r') as file:
                self.solutions = json.load(file)
        else:
            self.solutions = None
        
        self.train = train  # toggle to switch between train and test data

    def __len__(self):
        return len(self.challenges)

    def __getitem__(self, idx):
        task_id = list(self.challenges.keys())[idx]
        challenge = self.challenges[task_id]
        solution = self.solutions[task_id] if self.solutions is not None else None

        if self.train:
            inputs = [torch.tensor(pair['input'], dtype=torch.float32) for pair in challenge['train']]
            outputs = [torch.tensor(pair['output'], dtype=torch.float32) for pair in challenge['train']]
        else:
            inputs = [torch.tensor(pair['input'], dtype=torch.float32) for pair in challenge['test']]
            outputs = [torch.tensor(pair, dtype=torch.float32) for pair in solution] if self.solutions is not None else []

        return inputs, outputs

    def task_key(self, idx):
        return list(self.challenges.keys())[idx]


if __name__ == '__main__':
    from rich import print

    base_path = './data/arc-prize-2024/'

    # Reading files
    challenges = base_path + 'arc-agi_training_challenges.json'
    solutions = base_path + 'arc-agi_training_solutions.json'
    # challenges = base_path + 'arc-agi_evaluation_challenges.json'
    # solutions = base_path + 'arc-agi_evaluation_solutions.json'

    # Example usage
    dataset_train = ARCDataset(challenges, solutions, train=True)
    dataset_test = ARCDataset(challenges, solutions, train=False)
    print(f'Data size: {len(dataset_train)}')
    
    # Visualize a task
    for index in range(len(dataset_train)):
        plot_task(dataset_train, dataset_test, index)
    