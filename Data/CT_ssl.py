import glob
import os
import pandas as pd
from typing import Any, Optional
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    Orientationd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    CropForegroundd,
    SpatialPadd,
    LoadImaged,
    EnsureChannelFirstd,
)
from pytorch_lightning import LightningDataModule
from monai.data import Dataset
from torch.utils.data import DataLoader

from .utils import AsTupled
from .ssl_utils import apply_transforms


CT_DATA_DIR = "/lhome/andrhhaa/work/Data/CT/Images/"


class CTSSLDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = CT_DATA_DIR,
        num_workers: int = 1,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last

        self.input_dims = (1, 96, 96, 96)
        self.target_dims = (1, 96, 96, 96)

        self.labels = []
        self.num_classes = 0

        self.transforms = Compose(
            [
                LoadImaged(keys="image"),
                EnsureChannelFirstd(keys="image"),
                Orientationd(keys="image", axcodes="RAS"),
                Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                CropForegroundd(keys="image", source_key="image", margin=1),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandSpatialCropd(
                    keys="image", roi_size=[96, 96, 96], random_size=False
                ),
                SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                EnsureTyped(keys="image"),
                AsTupled(keys="image"),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        images = sorted(glob.glob(os.path.join(self.data_dir, "**/*.nii.gz")))
        img_dicts = [{"image": img} for img in images]
        if stage == "fit" or stage is None:
            # Assign train/val datasets for use in dataloaders
            self.ct_train = Dataset(img_dicts[:-100], transform=self.transforms,)

            self.ct_val = Dataset(img_dicts[-100:], transform=self.transforms,)

    def train_dataloader(self):
        return DataLoader(
            self.ct_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ct_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        batch = apply_transforms(batch)

        return batch


if __name__ == "__main__":
    dm = CTSSLDataModule()
    dm.setup()

    val_ds = dm.train_dataloader().dataset

    print(len(val_ds))

    for i in range(20):
        image = val_ds[i]
        layer = 44
        print(f"image shape: {image.shape}")
        plt.figure("image", (24, 6))
        plt.title(f"image channel 0")
        plt.imshow(image[0, layer, :].detach().cpu(), cmap="gray")
        # also visualize the 3 channels label corresponding to this image
        plt.show()
