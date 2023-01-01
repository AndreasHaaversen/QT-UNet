from typing import Any
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    Orientationd,
    EnsureTyped,
    LoadImaged,
    EnsureChannelFirstd,
    SpatialPadd,
    RandSpatialCropd,
    CenterSpatialCropd,
)

from .MSD import MSD_PATH, MSDDataModule
from .ssl_utils import apply_transforms

from .MSD_utils import (
    TASK_TO_ROI_SIZE,
    TASK_TO_VAL_TRANSFORM_MAP,
)

from .utils import AsTupled


class MSDSSLDataModule(MSDDataModule):
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
        super(MSDSSLDataModule, self).__init__(
            data_dir,
            num_workers,
            batch_size,
            shuffle,
            pin_memory,
            persistent_workers,
            drop_last,
            task,
            **kwargs,
        )

        self.train_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                *TASK_TO_VAL_TRANSFORM_MAP[task],
                SpatialPadd(keys=["image"], spatial_size=TASK_TO_ROI_SIZE[task]),
                RandSpatialCropd(
                    keys=["image"], roi_size=TASK_TO_ROI_SIZE[task], random_size=False,
                ),
                EnsureTyped(keys="image"),
                AsTupled(keys="image"),
            ]
        )

        self.val_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                *TASK_TO_VAL_TRANSFORM_MAP[task],
                SpatialPadd(keys=["image"], spatial_size=TASK_TO_ROI_SIZE[task]),
                CenterSpatialCropd(keys=["image"], roi_size=TASK_TO_ROI_SIZE[task]),
                EnsureTyped(keys="image"),
                AsTupled(keys="image"),
            ]
        )

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        batch = apply_transforms(batch)
        return batch


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
