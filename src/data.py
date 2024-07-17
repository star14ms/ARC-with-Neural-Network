import torch
from torch.utils.data import Dataset, DataLoader
import json
from pytorch_lightning import LightningDataModule

from classify import get_filter_funcs
from arc_prize.preprocess import one_hot_encode, one_hot_encode_changes


class ARCDataset(Dataset):
    def __init__(self, challenge_json, solution_json=None, train=True, filter_funcs=get_filter_funcs(), one_hot=True, cold_value=-1, augment_data=False):
        self.one_hot = one_hot
        self.cold_value = cold_value

        # Load challenge and solution data
        with open(challenge_json, 'r') as file:
            self.challenges = json.load(file)
        if solution_json is not None:
            with open(solution_json, 'r') as file:
                self.solutions = json.load(file)
        else:
            self.solutions = None
            
        # Convert to torch tensors
        self.challenges = {key: {
            'train': {
                'input': [torch.tensor(task_item['input']) for task_item in task['train']], 
                'output': [torch.tensor(task_item['output']) for task_item in task['train']]
            },
            'test': {
                'input': [torch.tensor(task_item['input']) for task_item in task['test']]
            }
        } for key, task in self.challenges.items()}
        
        self.challenges = {key: task for key, task in list(self.challenges.items()) \
            if all(
                filter_func(
                    [task_item for task_item in task['train']['input']], \
                    [task_item for task_item in task['train']['output']]
                ) for filter_func in filter_funcs)
        }
        self.challenges = {key: task for key, task in list(self.challenges.items()) if key in self.challenges}
        
        if self.solutions is not None:
            self.solutions = {key: [torch.tensor(task_item) for task_item in task] for key, task in self.solutions.items()}
            self.solutions = {key: task for key, task in self.solutions.items() if key in self.challenges}
            
        self.train = train  # toggle to switch between train and test data
        
        if augment_data:
            self.augment_data()

    def __len__(self):
        return len(self.challenges)

    def __getitem__(self, idx):
        task_id = list(self.challenges.keys())[idx]
        challenge = self.challenges[task_id]
        solution = self.solutions[task_id] if self.solutions is not None else None

        # if self.train:
        #     inputs = [pair for pair in challenge['train']['input']]
        #     outputs = [pair for pair in challenge['train']['output']]
        # else:
        #     inputs = [pair for pair in challenge['test']['input']]
        #     outputs = [pair for pair in solution] if self.solutions is not None else []

        if self.train:
            inputs = [one_hot_encode(task_item, cold_value=self.cold_value) if self.one_hot else task_item for task_item in challenge['train']['input']]
            outputs = [tuple(one_hot_encode_changes(task_item_in, task_item_out)) if self.one_hot else task_item_out for task_item_in, task_item_out in zip(challenge['train']['input'], challenge['train']['output'])]
        else:
            inputs = [one_hot_encode(task_item, cold_value=self.cold_value) if self.one_hot else task_item for task_item in challenge['test']['input']]
            outputs = [tuple(one_hot_encode_changes(task_item_in, task_item_out)) if self.one_hot else task_item_out for task_item_in, task_item_out in zip(challenge['test']['input'], solution)] if self.solutions is not None else []

        return inputs, outputs

    def task_key(self, idx):
        return list(self.challenges.keys())[idx]

    def augment_data(self):
        '''Augment data by adding all possible rotated and flipped versions without duplicates'''

        def unique_augmentations(tensor):
            # Create a set to store unique transformations
            unique_transforms = set()
            # List to store the augmented tensors
            augmented_tensors = []

            # Generate all possible transformations
            transformations = [
                tensor,  # original
                torch.rot90(tensor, 1, (0, 1)),  # 90 degrees
                torch.rot90(tensor, 2, (0, 1)),  # 180 degrees
                torch.rot90(tensor, 3, (0, 1)),  # 270 degrees
                torch.flip(tensor, [1]),  # horizontal flip
                torch.flip(tensor, [0]),  # vertical flip
                torch.rot90(torch.flip(tensor, [1]), 1, (0, 1)),  # 90 degrees + horizontal flip
                torch.rot90(torch.flip(tensor, [0]), 1, (0, 1)),  # 90 degrees + vertical flip
                torch.rot90(torch.flip(tensor, [1]), 3, (0, 1)),  # 270 degrees + horizontal flip
                torch.rot90(torch.flip(tensor, [0]), 3, (0, 1)),  # 270 degrees + vertical flip
            ]

            # Add unique transformations to the list
            for t in transformations:
                t_tuple = tuple(t.numpy().ravel())
                if t_tuple not in unique_transforms:
                    unique_transforms.add(t_tuple)
                    augmented_tensors.append(t)
            
            return augmented_tensors

        # Augment challenges data
        augmented_challenges = {}
        for key, task in self.challenges.items():
            augmented_task = {}
            augmented_task['train'] = {
                'input': [],
                'output': []
            }
            augmented_task['test'] = {
                'input': []
            }
            
            # Apply augmentations to train data
            for input_tensor, output_tensor in zip(task['train']['input'], task['train']['output']):
                augmented_task['train']['input'].extend(unique_augmentations(input_tensor))
                augmented_task['train']['output'].extend(unique_augmentations(output_tensor))
            
            # Apply augmentations to test data (input only)
            for input_tensor in task['test']['input']:
                augmented_task['test']['input'].extend(unique_augmentations(input_tensor))
            
            augmented_challenges[key] = augmented_task
        self.challenges = augmented_challenges
        
        if self.solutions is None:
            return

        # Augment solutions data
        augmented_solutions = {}
        for key, task in self.solutions.items():
            augmented_task = []
            augmented_task.extend(task)
            
            # Apply augmentations to each task item
            for task_item in task:
                augmented_task.extend(unique_augmentations(task_item))
            
            augmented_solutions[key] = augmented_task
        self.solutions = augmented_solutions


class ARCDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __iter__(self):
        for batch in super().__iter__():
            xs, ts = batch
            yield list(zip(xs, ts))
            
    def __len__(self):
        return super().__len__()


class ARCDataModule(LightningDataModule):
    def __init__(self, base_path='./data/arc-prize-2024/', batch_size=1, augment_data=False, cold_value=-1):
        super().__init__()
        self.base_path = base_path
        self.challenges_train = self.base_path + 'arc-agi_training_challenges.json'
        self.solutions_train = self.base_path + 'arc-agi_training_solutions.json'
        self.challenges_val = self.base_path + 'arc-agi_evaluation_challenges.json'
        self.solutions_val = self.base_path + 'arc-agi_evaluation_solutions.json'
        self.challenges_test = base_path + 'arc-agi_test_challenges.json'
        self.solutions_test = None

        self.batch_size = batch_size
        self.augment_data = augment_data
        self.cold_value = cold_value
        self.prepare_data()

    def prepare_data(self):
        self.setup()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = ARCDataset(self.challenges_train, self.solutions_train, train=True, augment_data=self.augment_data, cold_value=self.cold_value)
            self.val_dataset = ARCDataset(self.challenges_val, self.solutions_val, train=False, augment_data=self.augment_data, cold_value=self.cold_value)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = ARCDataset(self.challenges_test, self.solutions_test, train=False, augment_data=self.augment_data, cold_value=self.cold_value)

    def train_dataloader(self):
        return ARCDataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return ARCDataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return ARCDataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__ == '__main__':
    from rich import print
    from utils.visualize import plot_task
    import os

    base_path = './data/arc-prize-2024/'
    data_category = 'val'
    fdir_to_save = f'output/task_visualization/{data_category}/'

    # Reading files
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

    # Example usage
    filter_funcs = ()
    dataset_train = ARCDataset(challenges, solutions, train=True, one_hot=False, filter_funcs=filter_funcs)
    dataset_test = ARCDataset(challenges, solutions, train=False, one_hot=False, filter_funcs=filter_funcs)
    print(f'Data size: {len(dataset_train)}')

    # save figure images
    if fdir_to_save is not None:
        fdir_to_save = os.path.join(os.getcwd(), fdir_to_save)
        os.makedirs(fdir_to_save, exist_ok=True)

    # Visualize a task
    for index in range(len(dataset_train)):
        plot_task(dataset_train, dataset_test, index, data_category, fdir_to_save=fdir_to_save)


    # # Example usage
    # datamodule = ARCDataModule(base_path=base_path, batch_size=1)

    # for task in datamodule.train_dataloader():
    #     print(len(task))
    #     for x, t in task:
    #         print(x[0].shape, t[0].shape)
    #         break
