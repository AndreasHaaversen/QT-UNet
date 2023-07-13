from argparse import Namespace
import os
from tkinter import Image
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import torch
from PIL import Image

import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms import Compose, ToPILImage, PILToTensor

from Data import data_modules
from Models.QT_UNet.QT_UNet import QTUNet
import SimpleITK as sitk

from monai.transforms import BatchInverseTransform

from shutil import make_archive


models = {
    "QT-UNet": QTUNet,
}


def run_inference(
    selected_experiment: str,
    fast_dev_run: bool,
    task: str,
    selected_model: str,
    checkpoint_path: str,
    output_path: str,
    data_args: Namespace,
    trainer_args: dict,
    metric_args: dict,
):
    if task is not None and "MSD" in selected_experiment and "task" in data_args:
        data_args.task = task.strip()
    dm = data_modules[selected_experiment](**vars(data_args))

    if "background" in dm.labels and not metric_args["include_background"]:
        dm.labels.remove("background")

    print("Checkpoint path: ", checkpoint_path)
    # get model name from checkpoint path
    model_name = checkpoint_path.split("/")[1]

    model_class = models[selected_model]

    model = model_class.load_from_checkpoint(
        checkpoint_path,
        input_shape=dm.input_dims,
        max_epochs=350,
        num_classes=dm.num_classes,
    )

    trainer = pl.Trainer(
        gpus=1,
        strategy=DDPPlugin(find_unused_parameters=False),
        fast_dev_run=fast_dev_run,
        **trainer_args,
    )

    predictions_list = trainer.predict(model, dm)

    dir_name = os.path.abspath(output_path)
    os.makedirs(dir_name, exist_ok=True)
    i = 0
    if selected_experiment in ["MSD", "BTCV", "BraTS"]:
        batch_inverter = BatchInverseTransform(
            dm.predict_transform, dm.predict_dataloader()
        )

    for data_sample in predictions_list:
        data_sample["image"] = data_sample["pred"]
        if selected_experiment in ["MSD", "BTCV", "BraTS"]:
            data_sample = batch_inverter(data_sample)[0]

        pred = data_sample["image"]
        name = data_sample["image_meta_dict"]["filename_or_obj"].split("/")[-1]
        if selected_experiment == "BraTS" or (
            selected_experiment == "MSD" and "Task01" in task
        ):
            if selected_experiment == "MSD":
                pred = torch.tensor(pred)
            ET = pred[0].bool()
            NET = torch.logical_and(pred[1], torch.logical_not(ET))
            ED = torch.logical_and(pred[2], torch.logical_not(pred[1]))

            pred = torch.zeros(pred.shape[1:], device=pred.device)
            pred[ET] = 4
            pred[NET] = 1
            pred[ED] = 2
        else:
            pred = np.argmax(pred, axis=0)

        if selected_experiment == "BraTS" or (
            selected_experiment == "MSD" and "Task01" in task
        ):
            pred = sitk.GetImageFromArray(
                pred.cpu().numpy().astype(np.uint8).swapaxes(0, -1)
            )
        else:
            pred = sitk.GetImageFromArray(pred.astype(np.uint8).swapaxes(0, -1))

        if not selected_experiment == "MSD":
            ref_path = data_sample["image_meta_dict"]["filename_or_obj"]
            ref_img = sitk.ReadImage(ref_path)

            pred.CopyInformation(ref_img)

        if selected_experiment in ["MSD", "BTCV", "BraTS"]:
            filename = f"{dir_name}/{name}"
            print(filename)
            sitk.WriteImage(pred, filename)

    zip_name = f"{model_name}-{selected_experiment}"
    if task is not None:
        zip_name += f"-{task}"
    zip_path = os.path.join(dir_name, "..", zip_name)
    print(
        f"Inference complete, results are beeing zipped to {os.path.abspath(zip_path)}.zip"
    )
    make_archive(zip_path, "zip", dir_name)
    print("Done!")
