from argparse import Namespace
import os
import torch

import yaml
from Data.BTCV import BTCVDataModule
from Data.Brats import BratsDataModule
from Data.MSD import MSDDataModule
from Models import get_model
from fvcore.nn import FlopCountAnalysis
from monai.inferers import sliding_window_inference
from torch import nn

from yamlinclude import YamlIncludeConstructor

YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.FullLoader, base_dir="./Configs"
)

datamodules = [
    BratsDataModule(batch_size=1),
    BTCVDataModule(batch_size=1),
    MSDDataModule(batch_size=1, task="Task01"),
    MSDDataModule(batch_size=1, task="Task02"),
    MSDDataModule(batch_size=1, task="Task03"),
    MSDDataModule(batch_size=1, task="Task04"),
    MSDDataModule(batch_size=1, task="Task05"),
    MSDDataModule(batch_size=1, task="Task06"),
    MSDDataModule(batch_size=1, task="Task07"),
    MSDDataModule(batch_size=1, task="Task08"),
    MSDDataModule(batch_size=1, task="Task09"),
    MSDDataModule(batch_size=1, task="Task10"),
]

# A list of paths to model yaml config files
model_config_paths = [
    "./Configs/Models/QT-UNet/Tiny.yaml",
    "./Configs/Models/QT-UNet/Small.yaml",
    "./Configs/Models/QT-UNet/Base.yaml",
    "./Configs/Models/VT-UNet/Tiny.yaml",
    "./Configs/Models/VT-UNet/Small.yaml",
    "./Configs/Models/VT-UNet/Base.yaml",
    "./Configs/Models/QT-UNet-A/Tiny.yaml",
    "./Configs/Models/QT-UNet-A/Small.yaml",
    "./Configs/Models/QT-UNet-A/Base.yaml",
    "./Configs/Models/VT-UNet-A/Tiny.yaml",
    "./Configs/Models/VT-UNet-A/Small.yaml",
    "./Configs/Models/VT-UNet-A/Base.yaml",
]


import os, sys


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


if __name__ == "__main__":
    report = ""

    for dm_i, dm in enumerate(datamodules):
        dm.setup()
        for model_config_path in model_config_paths:
            if os.path.isfile(model_config_path):
                with open(model_config_path) as file:
                    model_config = Namespace(**yaml.full_load(file))

                with HiddenPrints():
                    model = get_model(
                        model_config.type,
                        model_config,
                        dm,
                        100,
                        {"list": ["Dice"], "include_background": True},
                    )

                    sample = dm.train_dataloader().dataset[0]
                    if dm_i > 2 and isinstance(sample[0], list):
                        sample[0] = sample[0][0]
                    if len(sample[0].shape) == 3:
                        sample[0] = torch.unsqueeze(sample[0], dim=0)

                    flop_count = FlopCountAnalysis(
                        model, [torch.unsqueeze(sample[0], 0), sample[1]]
                    )

                task_no = dm.task if hasattr(dm, "task") else ""
                report += f"{model_config.type}-{model_config.variant} in {dm.__class__.__name__} {task_no}: {flop_count.total():,}\n"

        report += "\n"

    print(report)
