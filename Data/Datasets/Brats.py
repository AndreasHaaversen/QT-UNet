import pathlib
import SimpleITK as sitk
import numpy as np
from torch.utils.data.dataset import Dataset


class BratsDataset(Dataset):
    def __init__(self, base_path, split="train", transforms=None):
        super(BratsDataset, self).__init__()
        self.transforms = transforms
        self.split = split

        self.datas = []
        self.patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        if self.split != "predict":
            self.patterns.append("_seg")

        base_folder = pathlib.Path(base_path).joinpath(split).resolve()
        print(base_folder)
        assert base_folder.exists()
        patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])
        for patient_dir in patients_dir:
            patient_id = patient_dir.name
            paths = [
                patient_dir / f"{patient_id}{value}.nii.gz" for value in self.patterns
            ]
            patient = dict(
                id=patient_id, t1=paths[0], t1ce=paths[1], t2=paths[2], flair=paths[3],
            )
            if self.split != "predict":
                patient["seg"] = paths[4]
            self.datas.append(patient)

    def __getitem__(self, idx):
        _patient = self.datas[idx]
        data = {}
        patient_image = [_patient[key] for key in _patient if key not in ["id", "seg"]]

        data["image"] = patient_image

        if self.split != "predict":
            data["label"] = _patient["seg"]

        data["name"] = f"{_patient['id']}.nii.gz"

        if self.transforms is not None:
            data = self.transforms(data)

        return data

    @staticmethod
    def load_nii(path_folder):
        return sitk.ReadImage(str(path_folder))

    def __len__(self):
        return len(self.datas)
