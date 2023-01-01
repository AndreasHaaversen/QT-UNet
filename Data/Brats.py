from typing import Optional
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
)
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .utils import AsTupled, ConvertToMultiChannelBasedOnBratsClassesd

from .Datasets.Brats import BratsDataset

BRATS_PATH = "/lhome/andrhhaa/work/Data/BraTS2021"


class BratsDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = BRATS_PATH,
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

        self.input_dims = (4, 128, 128, 128)
        self.target_dims = (3, 128, 128, 128)
        self.num_classes = 3
        self.labels = ["ET", "TC", "WT"]

        self.train_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                CropForegroundd(keys=["image", "label"], source_key="image", margin=1),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandSpatialCropd(
                    keys=["image", "label"], roi_size=[128, 128, 128], random_size=False
                ),
                SpatialPadd(keys=["image", "label"], spatial_size=[128, 128, 128]),
                EnsureTyped(keys=["image", "label"]),
                AsTupled(keys=["image", "label"]),
            ]
        )

        self.val_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                CropForegroundd(keys=["image", "label"], source_key="image", margin=1),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                EnsureTyped(keys=["image", "label"]),
                AsTupled(keys=["image", "label"]),
            ]
        )

        self.predict_transform = Compose(
            [
                LoadImaged(keys="image"),
                Orientationd(keys="image", axcodes="RAS"),
                Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                CropForegroundd(keys="image", source_key="image", margin=1),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Assign train/val datasets for use in dataloaders
            self.brats_train = BratsDataset(
                self.data_dir, "train", self.train_transform
            )
            self.brats_val = BratsDataset(self.data_dir, "val", self.val_transform)

        if stage == "test" or stage is None:
            self.brats_test = BratsDataset(self.data_dir, "test", self.val_transform)

        if stage == "predict" or stage is None:
            self.brats_predict = BratsDataset(
                self.data_dir, "predict", self.predict_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.brats_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.brats_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.brats_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.brats_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


if __name__ == "__main__":
    dm = BratsDataModule()
    dm.setup()

    val_ds = dm.val_dataloader().dataset

    print(len(val_ds))

    # pick one image from DecathlonDataset to visualize and check the 4 channels
    for i in range(20):
        image, label = val_ds[i]
        layer = 100
        print(f"image shape: {image.shape}")
        plt.figure("image", (24, 6))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.title(f"image channel {i}")
            plt.imshow(image[i, :, layer, :].detach().cpu(), cmap="gray")
        # also visualize the 3 channels label corresponding to this image
        print(f"label shape: {label.shape}")
        plt.figure("label", (18, 6))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(f"label channel {i}")
            plt.imshow(label[i, :, layer, :].detach().cpu())
        plt.show()
