import os

import numpy as np

import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader

import os
from typing import List, Tuple
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
import torchvision.transforms as T

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_train_frames,
                 num_val_frames, random_seed, target_size, train_fraction,
                 num_burn_in_frames=None):
        super().__init__()

        self.random_state = np.random.RandomState(seed=random_seed)

        self.batch_size = batch_size
        self.num_burn_in_frames = num_burn_in_frames
        self.num_train_frames = num_train_frames
        self.num_val_frames = num_val_frames
        self.random_seed = random_seed
        self.target_size = target_size
        self.train_fraction = train_fraction
        assert(self.train_fraction > 0 and self.train_fraction <= 1.0)

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage = None):
        self.train_set = self.create_dataset('train')
        self.val_set = self.create_dataset('val')
        self.test_set = self.create_dataset('test')

    def create_dataset(self, split: str):
        raise NotImplementedError()

    def train_dataloader(self):
        assert(self.train_set)
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=min(20, os.cpu_count()),
            shuffle=True, drop_last=True, persistent_workers=True,
            pin_memory=True)

    def val_dataloader(self):
        assert(self.val_set)
        return DataLoader(
            self.val_set, batch_size=self.batch_size,
            num_workers=min(20, os.cpu_count()), shuffle=False,
            drop_last=False, persistent_workers=True,
            pin_memory=True)

    def test_dataloader(self):
        assert(self.test_set)
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=min(20, os.cpu_count()), shuffle=False,
            drop_last=False, persistent_workers=True,
            pin_memory=True)

class ViProDataset(Dataset):
    """Dataset for loading orbits videos."""

    def __init__(
             self,
             samples,
             num_frames: int,
             max_instances: int,
             random_state,
             load_keys: List[str],
             target_size: Tuple[int, int],
    ):
        super().__init__()

        self.samples = samples

        self.num_frames = num_frames
        self.max_instances = max_instances
        self.random_state = random_state
        self.load_keys = load_keys
        self.target_size = target_size # H, W

        self.video_transforms = T.Compose([T.ToTensor(), T.Resize(
            self.target_size, interpolation=T.InterpolationMode.BILINEAR, antialias=False)])

    def __getitem__(self, idx):
        sample = self.samples[idx]

        result = {}

        # Load additional data from metadata
        with open(os.path.join(sample, 'metadata.pkl'), 'rb') as f:
            tmp = pickle.load(f)
        metadata = tmp['metadata']

        num_instances = min(metadata['num_instances'], self.max_instances)
        num_frames = metadata['num_frames']
        resolution = tuple(metadata['resolution'])
        instances = tmp['instances'][:self.max_instances] if tmp['instances'] else None
        camera = tmp['camera']

        assert(resolution[0] >= self.target_size[0] and resolution[1] >= self.target_size[1])
        assert(num_frames >= self.num_frames)

        start_idx = 0
        end_idx = start_idx + self.num_frames

        # Video
        if 'video' in self.load_keys:
            video = np.load(os.path.join(sample, 'video.npz'))['video'][start_idx:end_idx, ..., :3]
            video = torch.stack([self.video_transforms(v) for v in video], dim=0)

            result['video'] = video

        camera_pos = None
        if 'physics' in self.load_keys or 'camera' in self.load_keys:
            camera_pos = torch.as_tensor(camera['positions'], dtype=torch.float32)[start_idx:end_idx]

            result['camera_positions'] = camera_pos

            camera_quat = torch.as_tensor(camera['quaternions'])[start_idx:end_idx]
            result['camera_quaternions'] = camera_quat

            if 'thetas' in camera:
                camera_thetas = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in camera['thetas']], dim=0)[1+start_idx:end_idx+1]
                camera_dthetas = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in camera['dthetas']], dim=0)[1+start_idx:end_idx+1]

                result['camera_thetas'] = camera_thetas
                result['camera_dthetas'] = camera_dthetas

        if 'physics' in self.load_keys:
            assert(instances is not None)

            # Mass
            mass = torch.zeros((self.num_frames, self.max_instances, 1), dtype=torch.float32)
            mass[:, :num_instances] = torch.stack([torch.tile(torch.as_tensor(v['mass']), (self.num_frames, 1)) for v in instances], dim=1)
            result['mass'] = mass

            # Velocities
            velocities = torch.zeros((self.num_frames, self.max_instances, 3), dtype=torch.float32)
            velocities[:, :num_instances] = torch.stack([
                 torch.as_tensor(v['velocities'][start_idx:end_idx]) for v in instances], dim=1)
            result['velocities'] = velocities

            # World Positions
            world_positions = torch.zeros((self.num_frames, self.max_instances, 3), dtype=torch.float32)
            world_positions[:, :num_instances] = torch.stack([
                torch.as_tensor(v['positions'][start_idx:end_idx]) for v in instances], dim=1)
            result['positions'] = world_positions

            # If applicable: theta and dtheta
            if 'theta' in instances[0]:
                assert(num_instances == 2)
                theta = torch.zeros((self.num_frames, self.max_instances, 1), dtype=torch.float32)
                theta[:, :num_instances] = torch.stack(
                    [torch.unsqueeze(torch.as_tensor(v['theta'][start_idx:end_idx]), dim=-1) for v in instances], dim=1)
                result['theta'] = theta

                dtheta = torch.zeros((self.num_frames, self.max_instances, 1), dtype=torch.float32)
                dtheta[:, :num_instances] = torch.stack(
                    [torch.unsqueeze(torch.as_tensor(v['dtheta'][start_idx:end_idx]), dim=-1) for v in instances], dim=1)
                result['dtheta'] = dtheta

        result['num_instances'] = torch.tile(torch.as_tensor(num_instances).unsqueeze(0), (self.num_frames, 1))

        return result

    def __len__(self):
        return len(self.samples)

class ViProVPDataset(ViProDataset):
    def __init__(self, num_burn_in_frames, **kwargs):
        super().__init__(**kwargs)

        self.num_burn_in_frames = num_burn_in_frames

    def __getitem__(self, idx):
        data_dict = super().__getitem__(idx)

        # Split all entries into burn in and unroll parts
        burn_in_dict = {}
        unroll_dict = {}

        for k, v in data_dict.items():
            if len(v.shape) < 1 or v.shape[0] != self.num_frames:
                burn_in_dict[k] = v
                unroll_dict[k] = v
            else:
                burn_in_dict[k] = torch.as_tensor(v[:self.num_burn_in_frames]) if v is not None else v
                unroll_dict[k] = torch.as_tensor(v[self.num_burn_in_frames:]) if v is not None else v

        return {'burn_in': burn_in_dict, 'unroll': unroll_dict}


class ViProDataModule(BaseDataModule):
    def __init__(self, data_dir, name,
                 load_keys,
                 max_instances=None, **kwargs):
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.base_dir = os.path.join(data_dir, name)
        self.load_keys = load_keys

        with open(os.path.join(self.base_dir, 'metainfo.pkl'), 'rb') as f:
            self.data_metainfo = pickle.load(f)

        self.max_instances = vars(self.data_metainfo['args']).get('max_num_objects')
        if max_instances is not None:
            self.max_instances = max_instances

        self.train_data = [os.path.join(self.base_dir, str(i)) for i in self.data_metainfo['train_indices']]
        if self.train_fraction != 1.0:
            # Only take a fraction of training samples
            self.train_data = self.train_data[:int(len(self.train_data) * self.train_fraction)]
        self.val_data = [os.path.join(self.base_dir, str(i)) for i in self.data_metainfo['val_indices']]

    def create_dataset(self, split: str):
        assert(self.train_data is not None)
        assert(self.val_data is not None)

        if split == 'train':
            return ViProVPDataset(
                samples=self.train_data,
                num_frames=self.num_burn_in_frames + self.num_train_frames,
                max_instances=self.max_instances,
                random_state=np.random.RandomState(seed=self.random_seed),
                load_keys=self.load_keys,
                target_size=self.target_size,
                num_burn_in_frames=self.num_burn_in_frames,
                )
        elif split == 'val':
            return ViProVPDataset(
                samples=self.val_data,
                num_frames=self.num_burn_in_frames + self.num_val_frames,
                max_instances=self.max_instances,
                random_state=None,
                load_keys=self.load_keys,
                target_size=self.target_size,
                num_burn_in_frames=self.num_burn_in_frames,
            )

        elif split == 'test':
            return ViProVPDataset(
                samples=self.val_data,
                num_frames=self.num_burn_in_frames + self.num_val_frames,
                max_instances=self.max_instances,
                random_state=None,
                load_keys=self.load_keys,
                target_size=self.target_size,
                num_burn_in_frames=self.num_burn_in_frames,
            )
        else:
            raise ValueError("Invalid split name!")

    def val_dataloader(self):
        return super().val_dataloader()
