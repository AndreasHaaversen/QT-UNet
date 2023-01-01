import torch
import torch.nn.functional as F
from monai.transforms import MapTransform
from monai.config import KeysCollection
from monai.networks.utils import one_hot
import numpy as np


def squeeze_to_long(t: torch.Tensor):
    return t.squeeze().long()


def squeeze_to_tensor_long(t: torch.Tensor):
    return torch.tensor(t.squeeze(), dtype=torch.long)


def squeeze(t: torch.Tensor):
    return t.squeeze()


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    label 1 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # label 4 is ET
            ET = d[key] == 4
            # merge label 4 and label 1 to construct TC
            TC = np.logical_or(ET, d[key] == 1)
            # merge labels 1, 2 and 4 to construct WT
            WT = np.logical_or(TC, d[key] == 2)

            d[key] = np.stack([ET, TC, WT], axis=0).astype(np.bool)
        return d


class ConvertToMultiChannelBasedOnBratsClassesFromMSDd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # label 2 is ET
            result.append(d[key] == 2)
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1)
            )
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


class AsTupled(MapTransform):
    def __call__(self, data):
        d = dict(data)
        result = []
        if len(self.keys) == 1:
            return d[self.keys[0]]

        for key in self.keys:
            result.append(d[key])
        return result


class AsOneHotd(MapTransform):
    def __init__(
        self, keys: KeysCollection, num_classes: int, allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.num_classes = num_classes

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            one_hot_enc = one_hot(d[key], num_classes=self.num_classes, dim=0)
            d[key] = one_hot_enc
        return d


class Clipd(MapTransform):
    def __init__(
        self, keys: KeysCollection, minv, maxv, allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.minv = minv
        self.maxv = maxv

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            d[key] = np.clip(d[key], self.minv, self.maxv)
        return d


mapping_20_classes = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 1,
    8: 2,
    9: 0,
    10: 0,
    11: 3,
    12: 4,
    13: 5,
    14: 0,
    15: 0,
    16: 0,
    17: 6,
    18: 0,
    19: 7,
    20: 8,
    21: 9,
    22: 10,
    23: 11,
    24: 12,
    25: 13,
    26: 14,
    27: 15,
    28: 16,
    29: 0,
    30: 0,
    31: 17,
    32: 18,
    33: 19,
    -1: 0,
}

mapping_7_categories = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    17: 3,
    18: 3,
    19: 3,
    20: 3,
    21: 4,
    22: 4,
    23: 5,
    24: 6,
    25: 6,
    26: 7,
    27: 7,
    28: 7,
    29: 7,
    30: 7,
    31: 7,
    32: 7,
    33: 7,
    -1: 7,
}

mapping_ntnu_classes = {
    0: 0,
    1: 19,
    2: 3,
    3: 16,
    4: 14,
    5: 19,
    6: 5,
    7: 18,
    8: 12,
    9: 6,
    10: 13,
    11: 1,
    12: 2,
    13: 11,
    14: 10,
    15: 7,
    16: 8,
    17: 17,
    18: 15,
    19: 9,
    20: 4,
}

mapping_ntnu_categories = {
    0: 0,
    1: 7,
    2: 2,
    3: 7,
    4: 7,
    5: 7,
    6: 2,
    7: 7,
    8: 6,
    9: 3,
    10: 6,
    11: 1,
    12: 1,
    13: 5,
    14: 4,
    15: 3,
    16: 3,
    17: 7,
    18: 7,
    19: 4,
    20: 2,
}


valid_CityScapes_classes = [
    key for (key, value) in mapping_20_classes.items() if value != 0
]


class ConvertToMultiChannelBasedOnCityScapesClasses(object):
    def _mask_transform(self, mask):
        label_mask = torch.zeros_like(mask, dtype=torch.long)
        for k in mapping_20_classes:
            label_mask[mask == k] = mapping_20_classes[k]
        return label_mask

    def __call__(self, data):
        data = self._mask_transform(data[0])
        data = F.one_hot(data, len(valid_CityScapes_classes) + 1).movedim(-1, 0)
        return data


class ConvertToMultiChannelBasedOnCityScapesCategories(object):
    def _mask_transform(self, mask):
        label_mask = torch.zeros_like(mask, dtype=torch.long)
        for k in mapping_7_categories.keys():
            label_mask[mask == k] = mapping_7_categories[k]
        return label_mask

    def __call__(self, data):
        data = self._mask_transform(data[0])
        data = F.one_hot(data, 8).movedim(-1, 0)
        return data


class ConvertToMultiChannelBasedOnNTNUClasses(object):
    def _mask_transform(self, mask):
        label_mask = torch.zeros_like(mask, dtype=torch.long)
        for k in mapping_ntnu_classes:
            label_mask[mask == k] = mapping_ntnu_classes[k]
        return label_mask

    def __call__(self, data):
        data = self._mask_transform(data[0])
        data = F.one_hot(data, len(valid_CityScapes_classes) + 1).movedim(-1, 0)
        return data


class ConvertToMultiChannelBasedOnNTNUCategories(object):
    def _mask_transform(self, mask):
        label_mask = torch.zeros_like(mask, dtype=torch.long)
        for k in mapping_ntnu_categories.keys():
            label_mask[mask == k] = mapping_ntnu_categories[k]
        return label_mask

    def __call__(self, data):
        data = self._mask_transform(data[0])
        data = F.one_hot(data, 8).movedim(-1, 0)
        return data
