import os
from typing import Any, Callable, Dict, Optional, Sequence, Union

from monai.data import CSVDataset
from monai.apps import download_and_extract
from monai.utils import ensure_tuple

# Modified from https://github.com/Project-MONAI/tutorials/blob/master/modules/tcia_csv_processing.ipynb
class TCIADataset(CSVDataset):
    """
    Dataset to automatically download TCIA data and load.
    Args:
        filename: the filename to load downloaded csv file. if providing a list
            of filenames, it will load all the files and join tables.
        img_dir: root directory to save downloaded raw DICOM images.
        row_indices: indices of the expected rows to load. it should be a list,
            every item can be a int number or a range `[start, end)` for the indices.
            for example: `row_indices=[[0, 100], 200, 201, 202, 300]`. if None,
            load all the rows in the file.
        col_names: names of the expected columns to load. if None, load all the columns.
        col_types: `type` and `default value` to convert the loaded columns, if None, use original data.
            it should be a dictionary, every item maps to an expected column, the `key` is the column
            name and the `value` is None or a dictionary to define the default value and data type.
            the supported keys in dictionary are: ["type", "default"]. for example::
                col_types = {
                    "subject_id": {"type": str},
                    "label": {"type": int, "default": 0},
                    "ehr_0": {"type": float, "default": 0.0},
                    "ehr_1": {"type": float, "default": 0.0},
                    "image": {"type": str, "default": None},
                }
        col_groups: args to group the loaded columns to generate a new column,
            it should be a dictionary, every item maps to a group, the `key` will
            be the new column name, the `value` is the names of columns to combine. for example:
            `col_groups={"ehr": [f"ehr_{i}" for i in range(10)], "meta": ["meta_1", "meta_2"]}`
        transform: transform to apply on the loaded items of a dictionary data.
        kwargs: additional arguments for `pandas.merge()` API to join tables.

    """

    def __init__(
        self,
        filename: Union[str, Sequence[str]],
        img_dir: str,
        row_indices: Optional[Sequence[Union[int, str]]] = None,
        col_names: Optional[Sequence[str]] = None,
        col_types: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
        col_groups: Optional[Dict[str, Sequence[str]]] = None,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        filename = ensure_tuple(filename)

        for f in filename:
            assert os.path.exists(f)

        super().__init__(
            filename=filename,
            row_indices=row_indices,
            col_names=col_names,
            col_types=col_types,
            col_groups=col_groups,
            transform=transform,
        )
        self.img_dir = img_dir

    def _get_image(self, series_uid: str):
        # download raw DICOM series based on `Series UID`
        data_dir = os.path.join(self.img_dir, f"{series_uid}")
        url = (
            "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID="
            + series_uid
        )
        if not os.path.exists(data_dir):
            download_and_extract(
                url=url, filepath=data_dir + ".zip", output_dir=data_dir, progress=False
            )
            if os.path.exists(data_dir + ".zip"):
                os.remove(data_dir + ".zip")
        return data_dir

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        if isinstance(index, int):
            series_uid = self.data[index]["Series ID"]
            self.data[index]["image"] = self._get_image(series_uid)
        return super().__getitem__(index=index)
