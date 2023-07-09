import math
from monai.transforms import (
    Spacingd,
    CropForegroundd,
    NormalizeIntensityd,
    RandSpatialCropd,
    SpatialPadd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandRotated,
    ScaleIntensityd,
    RandAxisFlipd,
    RandAffined,
    RandCropByPosNegLabeld,
    SqueezeDimd,
)


from .utils import Clipd, ConvertToMultiChannelBasedOnBratsClassesFromMSDd

TASK_LOOKUP = {
    "Task01": "Task01_BrainTumour",
    "Task02": "Task02_Heart",
    "Task03": "Task03_Liver",
    "Task04": "Task04_Hippocampus",
    "Task05": "Task05_Prostate",
    "Task06": "Task06_Lung",
    "Task07": "Task07_Pancreas",
    "Task08": "Task08_HepaticVessel",
    "Task09": "Task09_Spleen",
    "Task10": "Task10_Colon",
}

TASK_TO_ROI_SIZE = {
    "Task01": [128 for _ in range(3)],
    "Task02": [96 for _ in range(3)],
    "Task03": [96 for _ in range(3)],
    "Task04": [96 for _ in range(3)],
    "Task05": [96 for _ in range(3)],
    "Task06": [96 for _ in range(3)],
    "Task07": [96 for _ in range(3)],
    "Task08": [96 for _ in range(3)],
    "Task09": [96 for _ in range(3)],
    "Task10": [96 for _ in range(3)],
}

TASKS_WITH_POS_NEG_SAMPLING = [
    "Task02",
    "Task03",
    "Task06",
    "Task07",
    "Task08",
    "Task10",
]

TASK_TO_TEST_TRANSFORM = {
    "Task01": [
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear"),
        ),
        CropForegroundd(keys=["image"], source_key="image", margin=1),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ],
    "Task02": [
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear"),
        ),
        CropForegroundd(keys=["image"], source_key="image", margin=1),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ],
    "Task03": [
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear"),
        ),
        CropForegroundd(keys=["image"], source_key="image", margin=1),
        ScaleIntensityd(keys="image", minv=-21, maxv=189),
        NormalizeIntensityd(keys="image", nonzero=True),
    ],
    "Task04": [
        Spacingd(
            keys=["image"],
            pixdim=(0.2, 0.2, 0.2),
            mode=("bilinear"),
        ),
        CropForegroundd(keys=["image"], source_key="image", margin=1),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ],
    "Task05": [
        Spacingd(
            keys=["image"],
            pixdim=(0.5, 0.5, 0.5),
            mode=("bilinear"),
        ),
        CropForegroundd(keys=["image"], source_key="image", margin=1),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ],
    "Task06": [
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear"),
        ),
        CropForegroundd(keys=["image"], source_key="image", margin=1),
        Clipd(keys="image", minv=-1000, maxv=1000),
        NormalizeIntensityd(keys="image", nonzero=True),
    ],
    "Task07": [
        CropForegroundd(keys=["image"], source_key="image", margin=1),
        Clipd(keys="image", minv=-87, maxv=199),
    ],
    "Task08": [
        CropForegroundd(keys=["image"], source_key="image", margin=1),
        Clipd(keys="image", minv=0, maxv=230),
    ],
    "Task09": [
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear"),
        ),
        CropForegroundd(keys=["image"], source_key="image", margin=1),
        Clipd(keys="image", minv=-125, maxv=275),
    ],
    "Task10": [
        CropForegroundd(keys=["image"], source_key="image", margin=1),
        Clipd(keys="image", minv=-57, maxv=175),
        NormalizeIntensityd(keys="image", nonzero=True),
    ],
}

TASK_TO_VAL_TRANSFORM_MAP = {
    "Task01": [
        SqueezeDimd(keys="label"),
        ConvertToMultiChannelBasedOnBratsClassesFromMSDd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=1),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ],
    "Task02": [
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=1),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ],
    "Task03": [
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=1),
        ScaleIntensityd(keys="image", minv=-21, maxv=189),
        NormalizeIntensityd(keys="image", nonzero=True),
    ],
    "Task04": [
        Spacingd(
            keys=["image", "label"],
            pixdim=(0.2, 0.2, 0.2),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=1),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ],
    "Task05": [
        Spacingd(
            keys=["image", "label"],
            pixdim=(0.5, 0.5, 0.5),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=1),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ],
    "Task06": [
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=1),
        Clipd(keys="image", minv=-1000, maxv=1000),
        NormalizeIntensityd(keys="image", nonzero=True),
    ],
    "Task07": [
        CropForegroundd(keys=["image", "label"], source_key="image", margin=1),
        Clipd(keys="image", minv=-87, maxv=199),
    ],
    "Task08": [
        CropForegroundd(keys=["image", "label"], source_key="image", margin=1),
        Clipd(keys="image", minv=0, maxv=230),
    ],
    "Task09": [
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=1),
        Clipd(keys="image", minv=-125, maxv=275),
    ],
    "Task10": [
        CropForegroundd(keys=["image", "label"], source_key="image", margin=1),
        Clipd(keys="image", minv=-57, maxv=175),
        NormalizeIntensityd(keys="image", nonzero=True),
    ],
}

TASK_TO_TRANSFORM_MAP = {
    "Task01": [
        *TASK_TO_VAL_TRANSFORM_MAP["Task01"],
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=TASK_TO_ROI_SIZE["Task01"],
            random_size=False,
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=TASK_TO_ROI_SIZE["Task01"]),
        RandAxisFlipd(keys=["image", "label"], prob=0.5),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
    ],
    "Task02": [
        *TASK_TO_VAL_TRANSFORM_MAP["Task02"],
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            spatial_size=TASK_TO_ROI_SIZE["Task02"],
            label_key="label",
            image_key="image",
            pos=2.0,
            num_samples=4,
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=TASK_TO_ROI_SIZE["Task02"]),
        RandAxisFlipd(keys=["image", "label"], prob=0.5),
        RandRotated(keys=["image", "label"], prob=0.1),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ],
    "Task03": [
        *TASK_TO_VAL_TRANSFORM_MAP["Task03"],
        SpatialPadd(keys=["image", "label"], spatial_size=TASK_TO_ROI_SIZE["Task03"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            spatial_size=TASK_TO_ROI_SIZE["Task03"],
            label_key="label",
            image_key="image",
            num_samples=4,
        ),
        RandAxisFlipd(keys=["image", "label"], prob=0.2),
        RandRotated(keys=["image", "label"], prob=0.2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
    ],
    "Task04": [
        *TASK_TO_VAL_TRANSFORM_MAP["Task04"],
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=TASK_TO_ROI_SIZE["Task04"],
            random_size=False,
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=TASK_TO_ROI_SIZE["Task04"]),
        RandAxisFlipd(keys=["image", "label"], prob=0.1),
        RandRotated(keys=["image", "label"], prob=0.1),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
    ],
    "Task05": [
        *TASK_TO_VAL_TRANSFORM_MAP["Task05"],
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=TASK_TO_ROI_SIZE["Task05"],
            random_size=False,
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=TASK_TO_ROI_SIZE["Task05"]),
        RandAxisFlipd(keys=["image", "label"], prob=0.5),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        RandAffined(
            keys=["image", "label"],
            prob=1,
            scale_range=[0.3, 0.3, 0.0],
            rotate_range=[0, 0, math.pi],
        ),
    ],
    "Task06": [
        *TASK_TO_VAL_TRANSFORM_MAP["Task06"],
        SpatialPadd(keys=["image", "label"], spatial_size=TASK_TO_ROI_SIZE["Task06"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            spatial_size=TASK_TO_ROI_SIZE["Task06"],
            label_key="label",
            image_key="image",
            num_samples=4,
            pos=2,
            neg=1,
        ),
        RandAxisFlipd(keys=["image", "label"], prob=0.5),
        RandRotated(keys=["image", "label"], prob=0.3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
    ],
    "Task07": [
        *TASK_TO_VAL_TRANSFORM_MAP["Task07"],
        SpatialPadd(keys=["image", "label"], spatial_size=TASK_TO_ROI_SIZE["Task07"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            spatial_size=TASK_TO_ROI_SIZE["Task07"],
            label_key="label",
            image_key="image",
            num_samples=4,
        ),
        RandAxisFlipd(keys=["image", "label"], prob=0.5),
        RandRotated(keys=["image", "label"], prob=0.25),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
    ],
    "Task08": [
        *TASK_TO_VAL_TRANSFORM_MAP["Task08"],
        SpatialPadd(keys=["image", "label"], spatial_size=TASK_TO_ROI_SIZE["Task08"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            spatial_size=TASK_TO_ROI_SIZE["Task08"],
            label_key="label",
            image_key="image",
            num_samples=4,
        ),
        RandAxisFlipd(keys=["image", "label"], prob=0.5),
        RandRotated(keys=["image", "label"], prob=0.25),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
    ],
    "Task09": [
        *TASK_TO_VAL_TRANSFORM_MAP["Task09"],
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=TASK_TO_ROI_SIZE["Task09"],
            random_size=False,
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=TASK_TO_ROI_SIZE["Task09"]),
        RandAxisFlipd(keys=["image", "label"], prob=0.15),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
    ],
    "Task10": [
        *TASK_TO_VAL_TRANSFORM_MAP["Task10"],
        SpatialPadd(keys=["image", "label"], spatial_size=TASK_TO_ROI_SIZE["Task10"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            spatial_size=TASK_TO_ROI_SIZE["Task10"],
            label_key="label",
            image_key="image",
            num_samples=4,
        ),
        RandAxisFlipd(keys=["image", "label"], prob=0.5),
        RandRotated(keys=["image", "label"], prob=0.25),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
    ],
}
