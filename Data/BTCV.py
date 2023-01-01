from typing import Optional
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    CropForegroundd,
    SpatialPadd,
    AddChanneld,
    ToTensord,
    ScaleIntensityRanged,
    RandRotate90d,
    LoadImaged,
)
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .utils import AsOneHotd, AsTupled

from .Datasets.BTCV import BTCVDataset


BTCV_PATH = "/cluster/home/andrhhaa/Data/BTCV/RawData"


class BTCVDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = BTCV_PATH,
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
        self.target_dims = (14, 96, 96, 96)
        self.num_classes = 14
        self.labels = [
            "background",
            "spleen",
            "right kidney",
            "left kidney",
            "gallbladder",
            "esophagus",
            "liver",
            "stomach",
            "aorta",
            "inferior vena cava",
            "portal vein and splenic vein",
            "pancreas",
            "right adrenal gland",
            "left adrenal gland",
        ]

        self.train_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image", margin=1),
                RandSpatialCropd(
                    keys=["image", "label"], roi_size=[96, 96, 96], random_size=False
                ),
                SpatialPadd(keys=["image", "label"], spatial_size=[96, 96, 96]),
                RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
                RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3,),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                ToTensord(keys="label"),
                AsOneHotd(keys="label", num_classes=self.num_classes),
                EnsureTyped(keys=["image", "label"]),
                AsTupled(keys=["image", "label"]),
            ]
        )

        self.val_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image", margin=1),
                ToTensord(keys="label"),
                AsOneHotd(keys="label", num_classes=self.num_classes),
                EnsureTyped(keys=["image", "label"]),
                AsTupled(keys=["image", "label"]),
            ]
        )

        self.predict_transform = Compose(
            [
                LoadImaged(keys=["image"]),
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
                EnsureTyped(keys=["image"]),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Assign train/val datasets for use in dataloaders
            self.btcv_train = BTCVDataset(self.data_dir, "train", self.train_transform)
            self.btcv_val = BTCVDataset(self.data_dir, "validation", self.val_transform)
        if stage == "test" or stage is None:
            self.btcv_test = BTCVDataset(self.data_dir, "test", self.val_transform)

        if stage == "predict" or stage is None:
            self.btcv_predict = BTCVDataset(
                self.data_dir, "predict", self.predict_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.btcv_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.btcv_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.btcv_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.btcv_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


if __name__ == "__main__":
    dm = BTCVDataModule(num_workers=3)
    dm.setup(stage="fit")
    print("num classes: ", dm.num_classes)

    val_ds = dm.val_dataloader().dataset

    print("Length: ", len(val_ds))

    # pick one image from DecathlonDataset to visualize and check the 4 channels
    for data in val_ds:
        image, label = data
        print("image shape: ", image.shape)
        print("label shape: ", label.shape)
        print("unique values in label:", np.unique(label))

        fig = plt.figure("image and label", (4, 12))
        start_layer = 10
        layer_step = 20
        dim_index = 1
        num_steps = int((image.shape[dim_index] - start_layer) / layer_step)
        print(num_steps)
        for i, slice_index in zip(
            range(1, num_steps * 2, 2),
            range(start_layer, image.shape[dim_index], layer_step),
        ):
            # plot the slice 50 - 100 of image, label and blend result
            fig.add_subplot(num_steps, 2, i)
            plt.title(f"image slice {slice_index}")
            plt.imshow(image[0, slice_index, :, :], cmap="gray")
            fig.add_subplot(num_steps, 2, i + 1)
            plt.title(f"label slice {slice_index}")
            plt.imshow(label[slice_index, :, :])
        plt.show()
