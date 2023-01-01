# A program to download data from TCIA, convert it to nifti and save it to a directory.

import os
import shutil
import pandas as pd
from monai.apps import download_and_extract
import dicom2nifti
from joblib import Parallel, delayed
import multiprocessing

data_root = "/lhome/andrhhaa/work/Data/CT"
image_dir = os.path.join(data_root, "Images")
tmp_dir = os.path.join(data_root, "tmp")


def download(series_uid):
    # Download the data
    # Convert to nifti
    # Save to directory
    dicom_dir = os.path.join(tmp_dir, f"{series_uid}")
    nifti_dir = os.path.join(image_dir, f"{series_uid}")
    if os.path.exists(nifti_dir):
        print("Nifti dir allready exists")
        return
    os.makedirs(nifti_dir, exist_ok=True)
    print(dicom_dir)
    url = (
        "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID="
        + series_uid
    )

    if not os.path.exists(dicom_dir):
        try:
            download_and_extract(
                url=url,
                filepath=dicom_dir + ".zip",
                output_dir=dicom_dir,
                progress=False,
            )
        except:
            print("Failed to get image zip")
            return

    if os.path.exists(dicom_dir + ".zip"):
        os.remove(dicom_dir + ".zip")
    os.makedirs(nifti_dir, exist_ok=True)
    try:
        dicom2nifti.convert_directory(dicom_dir, nifti_dir)
    except:
        print("Conversion failed, skipping")
        os.rmdir(nifti_dir)
    if os.path.exists(dicom_dir):
        shutil.rmtree(dicom_dir)


if __name__ == "__main__":
    df = pd.read_csv("/lhome/andrhhaa/work/MPEx/Data/CSVs/CT_meta.csv")
    SeriesUIDs = df["Series ID"]

    num_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=num_cores)(
        delayed(download)(series_uid) for series_uid in SeriesUIDs
    )
