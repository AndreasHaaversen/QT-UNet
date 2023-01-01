from argparse import Namespace
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch

from Data import data_modules
from Models import get_model
from SSL import SSLHarness


def run_experiment(
    selected_experiment: str,
    test: bool,
    ssl: bool,
    fast_dev_run: bool,
    resume_from: str,
    task: str,
    selected_model: str,
    model_args: Namespace,
    data_args: Namespace,
    trainer_args: dict,
    metric_args: dict,
):
    selected_experiment = selected_experiment + "-SSL" if ssl else selected_experiment
    if task is not None and "MSD" in selected_experiment and "task" in data_args:
        data_args.task = task.strip()
    dm = data_modules[selected_experiment](**vars(data_args))

    if "background" in dm.labels and not metric_args["include_background"]:
        dm.labels.remove("background")

    if "void" in dm.labels and not metric_args["include_background"]:
        dm.labels.remove("void")

    model = get_model(
        selected_model, model_args, dm, trainer_args["max_epochs"], metric_args
    )

    if ssl:
        encoder = model.model.encoder

        model = SSLHarness(
            base_encoder=encoder,
            encoder_out_dim=model.model.encoder_output_dim,
            in_data_dim=dm.input_dims,
            max_epochs=trainer_args["max_epochs"],
            compress_encoder_output="CityScapes" in selected_experiment,
        )

    try:
        model_name = f"{model_args.type}-{model_args.variant}"
    except:
        model_name = model_args.type

    if ssl:
        model_name += "-SSL"

    if "pretrained_encoder_checkpoint" in model_args:
        model_name += "-pretrained"

    if "task" in data_args:
        selected_experiment += f"-{data_args.task}"

    logger = TensorBoardLogger("tb_logs", name=f"{model_name}/{selected_experiment}")

    value_to_monitor = "val/Dice_epoch" if not ssl else "val/train_loss"
    callbacks = [
        ModelCheckpoint(
            monitor=value_to_monitor,
            auto_insert_metric_name=False,
            filename="epoch={epoch:02d}-dice={" + value_to_monitor + ":.2f}",
            save_last=True,
            mode="max",
        )
    ]
    trainer = pl.Trainer(
        gpus=-1,
        strategy=DDPPlugin(find_unused_parameters=ssl or "UNETR" in model_name),
        logger=logger,
        fast_dev_run=fast_dev_run,
        callbacks=callbacks,
        weights_save_path="checkpoints",
        **trainer_args,
    )

    if not test:
        trainer.fit(model, dm, ckpt_path=resume_from)

    if ssl:
        save_state_dict(
            model, f"{model_args.type}/{model_args.variant}", selected_experiment
        )
    elif test and resume_from is not None:
        trainer.test(model, dm, ckpt_path=resume_from)
    else:
        trainer.test(model, dm)


@rank_zero_only
def save_state_dict(model, model_name, dataset_name):
    state_dict = model.state_dict()
    encoder_dict = {
        k.replace("online_network.", ""): state_dict[k]
        for k in state_dict.keys()
        if "online_network.encoder" in k
    }

    save_path = f"./out/{model_name}/{dataset_name}_pretrained.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(encoder_dict, save_path)

    print(f"Saved state dict to {save_path}")
