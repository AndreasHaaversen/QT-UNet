from typing import Any

import torch

from Data.ssl_utils import apply_transforms
from .Brats import BRATS_PATH, BratsDataModule

from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    RandSpatialCropd,
    SpatialPadd,
    RandCoarseDropout,
)

from .utils import AsTupled


class BraTSSSLDataModule(BratsDataModule):
    def __init__(
        self,
        data_dir: str = BRATS_PATH,
        num_workers: int = 1,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = True,
        **kwargs
    ):
        super().__init__(
            data_dir,
            num_workers,
            batch_size,
            shuffle,
            pin_memory,
            persistent_workers,
            drop_last,
            **kwargs
        )

        self.train_transform = Compose(
            [
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear"),),
                CropForegroundd(keys=["image"], source_key="image", margin=1),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                SpatialPadd(keys=["image"], spatial_size=[128, 128, 128]),
                RandSpatialCropd(
                    keys=["image"], roi_size=[128, 128, 128], random_size=False,
                ),
                AsTupled(keys=["image"]),
            ]
        )

        self.val_transform = Compose(
            [
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear"),),
                CropForegroundd(keys=["image"], source_key="image", margin=1),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                SpatialPadd(keys=["image"], spatial_size=[128, 128, 128]),
                RandSpatialCropd(
                    keys=["image"], roi_size=[128, 128, 128], random_size=False,
                ),
                AsTupled(keys=["image"]),
            ]
        )

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        batch = apply_transforms(batch)

        return batch

