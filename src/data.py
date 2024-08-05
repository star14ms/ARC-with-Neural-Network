import json
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from collections import OrderedDict

from classify import get_filter_funcs
from arc.preprocess import one_hot_encode, one_hot_encode_changes
from arc.utils.transform import collate_fn_same_shape
from lightning_fabric.utilities.data import suggested_max_num_workers
from functools import partial


class ARCDataset(Dataset):
    def __init__(self, challenge_json, solution_json=None, filter_funcs=get_filter_funcs(), one_hot=True, cold_value=-1, augment_data=False, ignore_color=False):
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
            
        if augment_data:
            self.augment_data()
        
        # reordering challenges based on the argument of the filter function, in_data_codes()
        if len(filter_funcs) > 0 and type(filter_funcs[0]).__name__ == 'partial' and filter_funcs[0].func.__name__ == 'in_data_codes' and filter_funcs[0].keywords.get('reorder'):
            self.challenges = OrderedDict(sorted(self.challenges.items(), key=lambda x: filter_funcs[0].keywords.get('codes').index(x[0])))
            self.solutions = OrderedDict(sorted(self.solutions.items(), key=lambda x: filter_funcs[0].keywords.get('codes').index(x[0]))) if self.solutions is not None else None

    def __len__(self):
        return len(self.challenges)

    def __getitem__(self, idx):
        task_id = list(self.challenges.keys())[idx]
        challenge = self.challenges[task_id]
        solution = self.solutions[task_id] if self.solutions is not None else None
        
        xs_train = [one_hot_encode(task_item, cold_value=self.cold_value) if self.one_hot else task_item for task_item in challenge['train']['input']]
        if self.ignore_color:
            ts_train = [one_hot_encode_changes(task_item_in, task_item_out)[0] if self.one_hot else task_item_out for task_item_in, task_item_out in zip(challenge['train']['input'], challenge['train']['output'])]
        else:
            ts_train = [one_hot_encode(task_item_out) if self.one_hot else task_item_out for task_item_out in challenge['train']['output']]

        xs_test = [one_hot_encode(task_item, cold_value=self.cold_value) if self.one_hot else task_item for task_item in challenge['test']['input']]

        if self.solutions is None: 
            ts_test = []
        elif self.ignore_color:
            ts_test = [one_hot_encode_changes(task_item_in, task_item_out)[0] if self.one_hot else task_item_out for task_item_in, task_item_out in zip(challenge['test']['input'], solution)]
        else:
            ts_test = [one_hot_encode(task_item_out) if self.one_hot else task_item_out for task_item_out in solution]

        return xs_train, ts_train, xs_test, ts_test, task_id

    def task_id(self, idx):
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
        indices_to_remove_test = {}
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
                unique_inputs, indices_to_remove = unique_augmentations(input_tensor)
                augmented_task['test']['input'].extend(unique_inputs)
                indices_to_remove_test[key] = indices_to_remove
            
            augmented_challenges[key] = augmented_task
        self.challenges = augmented_challenges
        
        if self.solutions is None:
            return

        # # Augment solutions data
        # augmented_solutions = {}
        # for key, task in self.solutions.items():
        #     augmented_task = []
            
        #     # Apply augmentations to each task item
        #     for task_item in task:
        #         augmented_task.extend(unique_augmentations(task_item, indices_to_remove_test[key]))
            
        #     augmented_solutions[key] = augmented_task
        # self.solutions = augmented_solutions


class ARCDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __iter__(self):
        for batch in super().__iter__():
            xs_train, ts_train, xs_test, ts_test, task_id = batch
            yield list(zip(xs_train, ts_train)), list(zip(xs_test, ts_test)), task_id

    def __len__(self):
        return super().__len__()


class ARCDataModule(LightningDataModule):
    def __init__(self, base_path='./data/arc-prize-2024/', batch_size_max=1, shuffle=True, augment_data=False, filter_funcs=None, cold_value=-1, ignore_color=False, num_workers=None, local_world_size=1, debug=False):
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
        self.filter_funcs = filter_funcs if filter_funcs is not None else get_filter_funcs()
        self.cold_value = cold_value
        self.ignore_color = ignore_color

        self.num_workers = num_workers if num_workers else suggested_max_num_workers(local_world_size=local_world_size or 1)
        self.kwargs_dataloader = {} if debug else {'num_workers': self.num_workers, 'persistent_workers': True}
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
            self.train_dataset = ARCDataset(self.challenges_train, self.solutions_train, filter_funcs=self.filter_funcs, **kwargs)
            self.val_dataset = ARCDataset(self.challenges_val, self.solutions_val, filter_funcs=self.filter_funcs, **kwargs)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = ARCDataset(self.challenges_test, self.solutions_test, **kwargs)

    def train_dataloader(self):
        collate_fn = partial(collate_fn_same_shape, batch_size_max=self.batch_size_max, shuffle=self.shuffle)
        return ARCDataLoader(self.train_dataset, batch_size=1, collate_fn=collate_fn, **self.kwargs_dataloader)

    def val_dataloader(self):
        collate_fn = partial(collate_fn_same_shape, batch_size_max=self.batch_size_max, shuffle=False)
        return ARCDataLoader(self.val_dataset, batch_size=1, collate_fn=collate_fn, **self.kwargs_dataloader)

    def test_dataloader(self):
        collate_fn = partial(collate_fn_same_shape, batch_size_max=1, shuffle=False)
        return ARCDataLoader(self.test_dataset, batch_size=1, collate_fn=collate_fn, **self.kwargs_dataloader)


if __name__ == '__main__':
    from rich import print
    from arc.utils.visualize import plot_task
    import os
    
    from arc.constants import get_challenges_solutions_filepath

    data_category = 'train'
    fdir_to_save = None
    # fdir_to_save = f'output/task_visualization/{data_category}/'

    # Example usage
    challenges, solutions = get_challenges_solutions_filepath(data_category)
    dataset = ARCDataset(challenges, solutions, one_hot=False, filter_funcs=())
    print(f'Data size: {len(dataset)}')

    # save figure images
    if fdir_to_save is not None:
        fdir_to_save = os.path.join(os.getcwd(), fdir_to_save)
        os.makedirs(fdir_to_save, exist_ok=True)

    # Visualize a task
    for index in range(len(dataset)):
        plot_task(dataset, index, data_category, fdir_to_save=fdir_to_save)


    # # Show Each Size of Batch
    # datamodule = ARCDataModule(batch_size_max=8, augment_data=False)

    # for task in datamodule.train_dataloader():
    #     print('{} Data -> {} Batches'.format(sum([len(xs) for xs, *_ in task[0]]), len(task[0])))
    #     for xs_train, ts_train in task[0]:
    #         print(xs_train.shape)
    #     break
    
    #     # print(task[2], len(task[1]))