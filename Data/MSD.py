from typing import Optional
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    Orientationd,
    EnsureTyped,
    LoadImaged,
    EnsureChannelFirstd,
    ToTensord,
    Identityd,
)
from monai.data.decathlon_datalist import load_decathlon_properties
from monai.apps.datasets import DecathlonDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from monai.data import list_data_collate

from .MSD_utils import (
    TASK_LOOKUP,
    TASK_TO_ROI_SIZE,
    TASK_TO_TRANSFORM_MAP,
    TASK_TO_VAL_TRANSFORM_MAP,
    TASKS_WITH_POS_NEG_SAMPLING,
)

from .utils import AsTupled, AsOneHotd


MSD_PATH = "/lhome/andrhhaa/work/Data/MSD"


LIST_COLLATE_TASKS = [TASK_LOOKUP[task] for task in TASKS_WITH_POS_NEG_SAMPLING]


class MSDDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = MSD_PATH,
        num_workers: int = 1,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = False,
        task: str = "Task02",
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
        self.task = TASK_LOOKUP[task]

        self._properties = load_decathlon_properties(
            f"{self.data_dir}/{self.task}/dataset.json",
            ["labels", "tensorImageSize", "modality"],
        )
        num_channels = len(self._properties["modality"])
        self.input_dims = (num_channels, *TASK_TO_ROI_SIZE[task])
        self.target_dims = TASK_TO_ROI_SIZE[task]
        self.num_classes = len(self._properties["labels"]) if task != "Task01" else 3
        self.labels = (
            list(self._properties["labels"].values())
            if task != "Task01"
            else ["ET", "TC", "WT"]
        )

        self.train_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                *TASK_TO_TRANSFORM_MAP[task],
                ToTensord(keys="label")
                if task != "Task01"
                else Identityd(keys="label"),
                AsOneHotd(keys="label", num_classes=self.num_classes)
                if task != "Task01"
                else Identityd(keys="label"),
                EnsureTyped(keys=["image", "label"]),
                AsTupled(keys=["image", "label"]),
            ]
        )

        self.val_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                *TASK_TO_VAL_TRANSFORM_MAP[task],
                ToTensord(keys="label")
                if task != "Task01"
                else Identityd(keys="label"),
                AsOneHotd(keys="label", num_classes=self.num_classes)
                if task != "Task01"
                else Identityd(keys="label"),
                EnsureTyped(keys=["image", "label"]),
                AsTupled(keys=["image", "label"]),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Assign train/val datasets for use in dataloaders
            self.msd_train = DecathlonDataset(
                self.data_dir,
                self.task,
                "training",
                self.train_transform,
                num_workers=self.num_workers,
                cache_num=0,
            )
            self.msd_val = DecathlonDataset(
                self.data_dir,
                self.task,
                "validation",
                self.val_transform,
                num_workers=self.num_workers,
                cache_num=0,
            )
        if stage == "test" or stage is None:
            # Test labels aren't available
            self.msd_test = DecathlonDataset(
                self.data_dir,
                self.task,
                "validation",
                self.val_transform,
                num_workers=self.num_workers,
                cache_num=0,
            )

    def train_dataloader(self):
        return DataLoader(
            self.msd_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=list_data_collate if self.task in LIST_COLLATE_TASKS else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.msd_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.msd_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


if __name__ == "__main__":
    dm = MSDDataModule(num_workers=3)
    dm.setup(stage="fit")
    print("num classes: ", dm.num_classes)

    val_ds = dm.val_dataloader().dataset

    print("Length: ", len(val_ds))

    # pick one image from DecathlonDataset to visualize and check the 4 channels
    for data in val_ds:
        image, label = data
        print("image shape: ", image.shape)
        print("label shape: ", label.shape)

        fig = plt.figure("image and label", (4, 12))
        start_layer = 50
        layer_step = 75
        dim_index = 2
        num_steps = int((image.shape[dim_index] - start_layer) / layer_step)
        print(num_steps)
        for i, slice_index in zip(
            range(1, num_steps * 2, 2),
            range(start_layer, image.shape[dim_index], layer_step),
        ):
            # plot the slice 50 - 100 of image, label and blend result
            fig.add_subplot(num_steps, 2, i)
            plt.title(f"image slice {slice_index}")
            plt.imshow(image[0, :, slice_index, :], cmap="gray")
            fig.add_subplot(num_steps, 2, i + 1)
            plt.title(f"label slice {slice_index}")
            plt.imshow(label[:, slice_index, :])
        plt.show()
