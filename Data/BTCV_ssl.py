from typing import Any


from Data.ssl_utils import apply_transforms
from .BTCV import BTCV_PATH, BTCVDataModule

from monai.transforms import (
    Compose,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    CropForegroundd,
    SpatialPadd,
    RandSpatialCropd,
    AddChanneld,
)

from .utils import AsTupled


class BTCVSSSLDataModule(BTCVDataModule):
    def __init__(
        self,
        data_dir: str = BTCV_PATH,
        num_workers: int = 1,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = False,
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
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear"),),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image", margin=1),
                SpatialPadd(keys=["image"], spatial_size=[96, 96, 96]),
                RandSpatialCropd(
                    keys=["image"], roi_size=[96, 96, 96], random_size=False,
                ),
                AsTupled(keys=["image"]),
            ]
        )

        self.val_transform = Compose(
            [
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear"),),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image", margin=1),
                SpatialPadd(keys=["image"], spatial_size=[96, 96, 96]),
                RandSpatialCropd(
                    keys=["image"], roi_size=[96, 96, 96], random_size=False,
                ),
                AsTupled(keys=["image"]),
            ]
        )

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        batch = apply_transforms(batch)

        return batch
