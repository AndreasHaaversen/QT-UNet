from pathlib import Path
import SimpleITK as sitk
from torch.utils.data.dataset import Dataset
import re

from yaml import scan

train_range = range(1, 35)
val_range = range(35, 40)


class BTCVDataset(Dataset):
    def __init__(self, base_path, split="train", transforms=None):
        super(BTCVDataset, self).__init__()
        self.transforms = transforms
        self.split = "Testing" if split == "predict" else "Training"
        self.id_range = train_range if split == "train" else val_range

        self.data = []

        base_folder = Path(base_path).joinpath(self.split, "img").resolve()
        print(base_folder)
        assert base_folder.exists()
        scans = [x for x in base_folder.iterdir() if x.is_file()]
        for scan in scans:
            if (
                not self.split == "Testing"
                and not int(re.findall(r"\d+", scan.name)[0]) in self.id_range
            ):
                continue
            image_path = str(scan)
            label_path = image_path.replace("img", "label")
            patient = {
                "image": image_path,
                "label": label_path,
            }
            self.data.append(patient)

    def __getitem__(self, idx):
        data = {}
        scans = self.data[idx]
        data["image"] = scans["image"]
        data["label"] = scans["label"]
        data["name"] = scans["image"].split("/")[-1]
        if self.transforms is not None:
            data = self.transforms(data)

        return data

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.data)
