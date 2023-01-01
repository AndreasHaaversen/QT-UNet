from typing import Tuple, Union
from nuscenes.nuscenes import NuScenes
from torch.utils.data import Dataset
from nuscenes.utils import splits
import numpy as np

import os


class nuScenesLidarDataset(Dataset):

    def __init__(self, nusc, transforms=None, target_transforms=None,
                 split: str = "train"):

        self.nusc = nusc

        self.transforms = transforms
        self.target_transforms = target_transforms
        self.split = split

        if nusc.version == 'v1.0-trainval':
            train_scenes = splits.train
            val_scenes = splits.val
        elif nusc.version == 'v1.0-test':
            train_scenes = splits.test
            val_scenes = []
        elif nusc.version == 'v1.0-mini':
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
        else:
            raise NotImplementedError

        available_scenes = self.get_available_scenes()
        available_scene_names = [s['name'] for s in available_scenes]
        train_scenes = list(
            filter(lambda x: x in available_scene_names, train_scenes))
        val_scenes = list(
            filter(lambda x: x in available_scene_names, val_scenes))
        train_scenes = set(
            [available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
        val_scenes = set(
            [available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

        self.train_token_list, self.val_token_list = self.get_path_infos(
            train_scenes, val_scenes)

        print('%s: train scene(%d), val scene(%d)' %
              (nusc.version, len(train_scenes), len(val_scenes)))

    def __len__(self) -> int:
        if self.split == 'train':
            return len(self.train_token_list)
        elif self.split == 'val':
            return len(self.val_token_list)
        elif self.split == 'test':
            return len(self.val_token_list)

    def __getitem__(self, index:  int) -> Tuple[any, any]:
        if self.split == 'train':
            sample_token = self.train_token_list[index]
        elif self.split == 'val':
            sample_token = self.val_token_list[index]
        elif self.split == 'test':
            sample_token = self.val_token_list[index]

        lidar_path = os.path.join(self.nusc.dataroot, self.nusc.get(
            'sample_data', sample_token)['filename'])

        # Get only XYZ
        raw_data = np.fromfile(
            lidar_path, dtype=np.float32).reshape((-1, 5))[:, :3]

        lidarseg_path = os.path.join(self.nusc.dataroot, self.nusc.get(
            'lidarseg', sample_token)['filename'])
        annotated_data = np.fromfile(
            lidarseg_path, dtype=np.uint8).reshape((-1, 1))

        # Not all datapairs have the same lengths as other datapairs. We pad out the rest with zeros so that they're all the same length
        max_len = 34880
        npad = ((0, max_len - raw_data.shape[0]), (0, 0))
        npad_annotated = ((0, max_len - annotated_data.shape[0]), (0, 0))

        if raw_data.shape[0] < max_len:
            raw_data = np.pad(raw_data, npad, mode="constant")
            annotated_data = np.pad(
                annotated_data, npad_annotated, mode="constant")

        if self.transforms != None:
            raw_data = self.transforms(raw_data)

        if self.target_transforms != None:
            annotated_data = self.target_transforms(annotated_data)

        return raw_data, annotated_data

    def get_available_scenes(self):
        available_scenes = []
        print('total scene num:', len(self.nusc.scene))
        for scene in self.nusc.scene:
            scene_token = scene['token']
            scene_rec = self.nusc.get('scene', scene_token)
            sample_rec = self.nusc.get(
                'sample', scene_rec['first_sample_token'])
            sd_rec = self.nusc.get(
                'sample_data', sample_rec['data']['LIDAR_TOP'])
            has_more_frames = True
            scene_not_exist = False
            while has_more_frames:
                lidar_path, boxes, _ = self.nusc.get_sample_data(
                    sd_rec['token'])
                if not os.path.exists(lidar_path):
                    scene_not_exist = True
                    break
                else:
                    break

            if scene_not_exist:
                continue
            available_scenes.append(scene)
        print('exist scene num:', len(available_scenes))
        return available_scenes

    def get_path_infos(self, train_scenes, val_scenes):
        train_token_list = []
        val_token_list = []
        for sample in self.nusc.sample:
            scene_token = sample['scene_token']
            data_token = sample['data']['LIDAR_TOP']
            if scene_token in train_scenes:
                train_token_list.append(data_token)
            else:
                val_token_list.append(data_token)
        return train_token_list, val_token_list


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, '../nuScenes')
    nusc = NuScenes(dataroot=data_dir)
    dataset = nuScenesLidarDataset(nusc)
    data = dataset[0]
    print(data)
    print(data[0].shape, data[1].shape)
