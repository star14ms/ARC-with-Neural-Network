import torch
from torch.utils.data import Dataset, DataLoader
import json
from pytorch_lightning import LightningDataModule



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


class ARCDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __iter__(self):
        for batch in super().__iter__():
            xs, ts = batch
            xs = [x.squeeze(0) for x in xs]
            ts = [t.squeeze(0) for t in ts]
            yield list(zip(xs, ts))
            
    def __len__(self):
        return super().__len__()


class ARCDataModule(LightningDataModule):
    def __init__(self, base_path='./data/arc-prize-2024/', batch_size=1):
        super().__init__()
        self.base_path = base_path
        self.challenges_train = self.base_path + 'arc-agi_training_challenges.json'
        self.solutions_train = self.base_path + 'arc-agi_training_solutions.json'
        self.challenges_val = self.base_path + 'arc-agi_evaluation_challenges.json'
        self.solutions_val = self.base_path + 'arc-agi_evaluation_solutions.json'
        self.challenges_test = base_path + 'arc-agi_test_challenges.json'
        self.solutions_test = None
        self.batch_size = batch_size
        self.prepare_data()

    def prepare_data(self):
        self.setup()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = ARCDataset(self.challenges_train, self.solutions_train, train=True)
            self.val_dataset = ARCDataset(self.challenges_val, self.solutions_val, train=False)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = ARCDataset(self.challenges_test, self.solutions_test, train=False)

    def train_dataloader(self):
        return ARCDataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return ARCDataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return ARCDataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__ == '__main__':
    from rich import print
    from utils.visualize import plot_task

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
        
    # datamodule = ARCDataModule(base_path=base_path, batch_size=1)
    
    # for xs, ts in datamodule.train_dataloader():
    #     print(len(xs), len(ts))
    #     print(xs[0].shape, ts[0].shape)
    #     break
    