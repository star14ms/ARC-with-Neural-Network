import torch
from torch.utils.data import Dataset, DataLoader
import json
from pytorch_lightning import LightningDataModule

from classify import get_filter_funcs
from arc_prize.preprocess import one_hot_encode, one_hot_encode_changes
from arc_prize.utils.transform import collate_fn_same_shape
from lightning_fabric.utilities.data import suggested_max_num_workers
from functools import partial
from classify import ARCDataClassifier


class ARCDataset(Dataset):
    def __init__(self, challenge_json, solution_json=None, train=True, filter_funcs=get_filter_funcs(), one_hot=True, cold_value=-1, augment_data=False, ignore_color=False):
        self.one_hot = one_hot
        self.cold_value = cold_value
        self.ignore_color = ignore_color

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
                    [task_item for task_item in task['train']['output']], \
                    key
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
        for x, t in zip(challenge['train']['input'], challenge['train']['output']):
            assert x.shape == t.shape, f"Input and output shapes do not match: {x.shape} != {t.shape}"
        if self.train:
            inputs = [one_hot_encode(task_item, cold_value=self.cold_value) if self.one_hot else task_item for task_item in challenge['train']['input']]
            if self.ignore_color:
                outputs = [one_hot_encode_changes(task_item_in, task_item_out)[0] if self.one_hot else task_item_out for task_item_in, task_item_out in zip(challenge['train']['input'], challenge['train']['output'])]
            else:
                outputs = [one_hot_encode(task_item_out, cold_value=0) if self.one_hot else task_item_out for task_item_out in challenge['train']['output']]
        else:
            inputs = [one_hot_encode(task_item, cold_value=self.cold_value) if self.one_hot else task_item for task_item in challenge['test']['input']]
            if self.ignore_color:
                outputs = [one_hot_encode_changes(task_item_in, task_item_out)[0] if self.one_hot else task_item_out for task_item_in, task_item_out in zip(challenge['test']['input'], solution)] if self.solutions is not None else []
            else:
                outputs = [one_hot_encode(task_item_out, cold_value=0) if self.one_hot else task_item_out for task_item_out in solution] if self.solutions is not None else []

        return inputs, outputs

    def task_key(self, idx):
        return list(self.challenges.keys())[idx]

    def augment_data(self):
        '''Augment data by adding all possible rotated and flipped versions without duplicates'''

        def unique_augmentations(tensor, indices_to_remove=None):
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
            
            if indices_to_remove is not None:
                transformations = [t for i, t in enumerate(transformations) if i not in indices_to_remove]
                return transformations

            # Add unique transformations to the list
            indices_to_remove = set()
            for i, t in enumerate(transformations):
                t_tuple = tuple(t.numpy().ravel())
                if t_tuple not in unique_transforms:
                    unique_transforms.add(t_tuple)
                    augmented_tensors.append(t)
                else:
                    indices_to_remove.add(i)
            
            return augmented_tensors, indices_to_remove

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
                unique_inputs, indices_to_remove = unique_augmentations(input_tensor)
                unique_outputs = unique_augmentations(output_tensor, indices_to_remove)
                augmented_task['train']['input'].extend(unique_inputs)
                augmented_task['train']['output'].extend(unique_outputs)

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
    def __init__(self, base_path='./data/arc-prize-2024/', batch_size_max=1, shuffle=True, augment_data=False, cold_value=-1, ignore_color=False, num_workers=None, local_world_size=1):
        super().__init__()
        self.base_path = base_path
        self.challenges_train = self.base_path + 'arc-agi_training_challenges.json'
        self.solutions_train = self.base_path + 'arc-agi_training_solutions.json'
        self.challenges_val = self.base_path + 'arc-agi_evaluation_challenges.json'
        self.solutions_val = self.base_path + 'arc-agi_evaluation_solutions.json'
        self.challenges_test = base_path + 'arc-agi_test_challenges.json'
        self.solutions_test = None

        self.batch_size_max = batch_size_max
        self.shuffle = shuffle
        self.augment_data = augment_data
        self.cold_value = cold_value
        self.ignore_color = ignore_color

        self.num_workers = num_workers if num_workers else suggested_max_num_workers(local_world_size=local_world_size or 1)
        self.prepare_data()

    def prepare_data(self):
        self.setup()

    def setup(self, stage=None):
        kwargs = {
            'cold_value': self.cold_value,
            'augment_data': self.augment_data,
            'ignore_color': self.ignore_color
        }

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = ARCDataset(self.challenges_train, self.solutions_train, train=True, **kwargs)
            self.val_dataset = ARCDataset(self.challenges_val, self.solutions_val, train=True, **kwargs)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = ARCDataset(self.challenges_test, self.solutions_test, train=False, **kwargs)

    def train_dataloader(self):
        collate_fn = partial(collate_fn_same_shape, batch_size_max=self.batch_size_max, shuffle=self.shuffle)
        return ARCDataLoader(self.train_dataset, batch_size=1, num_workers=self.num_workers, persistent_workers=True, collate_fn=collate_fn)

    def val_dataloader(self):
        collate_fn = partial(collate_fn_same_shape, batch_size_max=self.batch_size_max, shuffle=False)
        return ARCDataLoader(self.val_dataset, batch_size=1, num_workers=self.num_workers, persistent_workers=True, collate_fn=collate_fn)

    def test_dataloader(self):
        collate_fn = partial(collate_fn_same_shape, batch_size_max=1, shuffle=False)
        return ARCDataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, persistent_workers=True, collate_fn=collate_fn)


if __name__ == '__main__':
    from rich import print
    from arc_prize.utils.visualize import plot_task
    import os
    
    from arc_prize.constants import get_challenges_solutions_filepath

    data_category = 'train'
    fdir_to_save = None
    # fdir_to_save = f'output/task_visualization/{data_category}/'

    # Example usage
    challenges, solutions = get_challenges_solutions_filepath(data_category)
    dataset_train = ARCDataset(challenges, solutions, train=True, one_hot=False, filter_funcs=())
    dataset_test = ARCDataset(challenges, solutions, train=False, one_hot=False, filter_funcs=())
    print(f'Data size: {len(dataset_train)}')

    # save figure images
    if fdir_to_save is not None:
        fdir_to_save = os.path.join(os.getcwd(), fdir_to_save)
        os.makedirs(fdir_to_save, exist_ok=True)

    # Visualize a task
    for index in range(len(dataset_train)):
        plot_task(dataset_train, dataset_test, index, data_category, fdir_to_save=fdir_to_save)


    # # Show Each Size of Batch
    # datamodule = ARCDataModule(batch_size_max=4, augment_data=True)

    # for task in datamodule.val_dataloader():
    #     print('{} Data -> {} Batches'.format(sum([len(t) for x, t in task]), len(task)))
    #     for x, t in task:
    #         print(x.shape)
