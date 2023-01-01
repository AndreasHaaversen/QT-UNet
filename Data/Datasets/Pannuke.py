import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

split_to_fold = {
    "train": "fold1",
    "val": "fold2",
    "test": "fold3"
}


class PannukeDataset(Dataset):

    def __init__(self, path: str, split: str = "train", transforms=None, target_transforms=None):
        self.root_dir = path
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.fold = split_to_fold[split]
        self.dataset_size = None

        print("Loading Pannuke images...")
        image_dir = f"{self.root_dir}/images/{self.fold}"
        self.dataset_size = len(os.listdir(
            image_dir)) - 1

        print("Loading Pannuke types...")
        self.types = np.load(
            f"{self.root_dir}/images/{self.fold}/types.npy")

        print(
            f"Finished loading Pannuke. {self.dataset_size} samples.")

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx):
        image_path = f"{self.root_dir}/images/{self.fold}/{idx}.png"
        mask_path = f"{self.root_dir}/masks/{self.fold}/{idx}.png"
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.transforms is not None:
            image = self.transforms(image)

        if self.target_transforms is not None:
            mask = self.target_transforms(mask)

        return image, mask


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, '../Pannuke')
    ds = PannukeDataset(data_dir)

    print(ds[0])
