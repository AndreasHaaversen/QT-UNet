from typing import Tuple
from monai.transforms import RandRotate90, RandCoarseDropout
from monai.config.type_definitions import NdarrayOrTensor
import torch


class CustomRandRotate90(RandRotate90):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`, additionally returning the number k of rotations.
    """

    def __init__(
        self, prob: float = 0.1, max_k: int = 3, spatial_axes: Tuple[int, int] = (0, 1)
    ) -> None:
        super().__init__(prob, max_k, spatial_axes)

    def __call__(
        self, img: NdarrayOrTensor, randomize: bool = True
    ) -> Tuple[NdarrayOrTensor, int]:
        out = super().__call__(img, randomize)

        return out, self._rand_k


def apply_transforms(batch):
    rotate = CustomRandRotate90(prob=1)
    cutout = RandCoarseDropout(
        holes=1, spatial_size=32, max_holes=3, max_spatial_size=64, prob=1
    )
    img_1_rot_tmp = [rotate(i) for i in batch]
    img_1_rot = torch.stack([img for img, _ in img_1_rot_tmp])
    k_1 = torch.tensor([rot for _, rot in img_1_rot_tmp])
    img_2_rot_tmp = [rotate(i) for i in batch]
    img_2_rot = torch.stack([img for img, _ in img_2_rot_tmp])
    k_2 = torch.tensor([rot for _, rot in img_2_rot_tmp])

    transformed_img_1 = torch.stack([cutout(i) for i in img_1_rot])
    transformed_img_2 = torch.stack([cutout(i) for i in img_2_rot])
    return k_1, k_2, transformed_img_1, transformed_img_2, batch

