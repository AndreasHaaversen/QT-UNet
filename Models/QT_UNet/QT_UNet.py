from argparse import Namespace
import copy
from pytorch_lightning import LightningModule
from monai.losses import DiceLoss
from monai.transforms import AsDiscrete, Compose, Activations, EnsureType

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch_optimizer as extra_optim
from monai.inferers import sliding_window_inference

from Models.utils import common_predict_step, log_gif, get_metrics, step_log

from .components import SwinTransformerSys3D


class QTUNet(LightningModule):
    def __init__(
        self,
        args,
        input_shape,
        max_epochs,
        num_classes,
        labels=[],
        zero_head=False,
        metric_args={"list": ["Dice"], "include_background": True},
        **kwargs,
    ):
        super(QTUNet, self).__init__()

        exclusive_targets = args.exclusive_targets if hasattr(args, "exclusive_targets") else False
        include_background = args.include_background if hasattr(args, "include_background") else True

        self.roi_size = input_shape[1:]
        self.labels = labels

        self.save_hyperparameters(args)
        self.zero_head = zero_head
        self.max_epochs = max_epochs
        self.window_size = (
            self.hparams.window_size,
            self.hparams.window_size,
            self.hparams.window_size,
        )

        self.model = SwinTransformerSys3D(
            img_size=input_shape[1:],
            patch_size=self.hparams.patch_size,
            in_chans=input_shape[0],
            num_classes=num_classes,
            embed_dim=self.hparams.embed_dim,
            depths=self.hparams.depths,
            depths_decoder=self.hparams.decoder_depths,
            num_heads=self.hparams.num_heads,
            window_size=self.window_size,
            mlp_ratio=self.hparams.mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            final_upsample="expand_first",
            use_ca=self.hparams.use_ca if hasattr(self.hparams, "use_ca") else True,
        )

        self.loss = DiceLoss(
            include_background=include_background,
            sigmoid=not exclusive_targets,
            softmax=exclusive_targets,
            squared_pred=True,
        )
        self.metric_names = metric_args["list"]
        self.metrics = get_metrics(self.metric_names, metric_args["include_background"])

        self.post_trans = Compose(
            [
                EnsureType(),
                Activations(sigmoid=not exclusive_targets, softmax=exclusive_targets),
                AsDiscrete(threshold=0.5),
            ]
        )

        if "pretrained_checkpoint" in self.hparams:
            self.load_from()

        if (
            "pretrained_encoder_checkpoint" in self.hparams
            and self.hparams.pretrained_encoder_checkpoint != ""
        ):
            self.load_pretrained_encoder()

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.hparams.optimizer, None)
        if optimizer_class is None:
            optimizer_class = getattr(extra_optim, self.hparams.optimizer)

        optimizer = optimizer_class(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = {
            "scheduler": CosineAnnealingLR(optimizer, self.max_epochs),
            "name": "CosineLR",
        }
        return [optimizer], [lr_scheduler]

    def forward(self, batch):
        x, y = batch
        pred = self.model(x)
        return pred, y

    def step(self, batch):
        y_hat, y = self(batch)
        return self.loss(y_hat, y)

    def sliding_window_step(self, batch):
        x = batch[0]
        y = batch[1]
        y_hat = sliding_window_inference(
            inputs=x,
            roi_size=self.roi_size,
            sw_batch_size=x.shape[0],
            predictor=self.model,
            overlap=0.5,
        )

        loss = self.loss(y_hat, y)

        y_hat = torch.stack([self.post_trans(i) for i in y_hat])

        metrics = [metric(y_hat, y) for metric in self.metrics]

        return y_hat, loss, metrics

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        step_log(self.log, "train", loss, None, self.metric_names, self.labels)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, loss, metrics = self.sliding_window_step(batch)
        step_log(self.log, "val", loss, metrics, self.metric_names, self.labels)
        if batch_idx == 0:
            log_gif(self.logger, "val", self.current_epoch, pred, batch, self.device)

    def test_step(self, batch, batch_idx):
        pred, loss, metrics = self.sliding_window_step(batch)
        step_log(self.log, "test", loss, metrics, self.metric_names, self.labels)

        if batch_idx == 0:
            log_gif(self.logger, "test", self.current_epoch, pred, batch, self.device)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        return common_predict_step(batch, self.roi_size, self.model, self.post_trans)

    def load_from(self):
        pretrained_path = self.hparams.pretrained_checkpoint
        print("pretrained_path:{}".format(pretrained_path))
        device = self.device
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        if "model" not in pretrained_dict:
            print("---start load pretrained modle by splitting---")
            pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
            for k in list(pretrained_dict.keys()):
                if "output" in k:
                    print("delete key:{}".format(k))
                    del pretrained_dict[k]
            self.model.load_state_dict(pretrained_dict, strict=False)

            return
        pretrained_dict = pretrained_dict["model"]
        print("---start load pretrained model of swin encoder---")

        model_dict = self.model.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():

            if "layers." in k:
                current_layer_num = 3 - int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k: v})

                full_dict[
                    f"encoder.layer.{str(current_layer_num)}{k[8:]}"
                ] = full_dict.pop(k)
            else:
                full_dict[f"encoder.{k}"] = full_dict.pop(k)

        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print(
                        "delete:{};shape pretrain:{};shape model:{}".format(
                            k, full_dict[k].shape, model_dict[k].shape
                        )
                    )
                    del full_dict[k]

        self.model.load_state_dict(full_dict, strict=False)

    def load_pretrained_encoder(self):
        pretrained_path = self.hparams.pretrained_encoder_checkpoint
        print("pretrained_encoder_path:{}".format(pretrained_path))
        device = self.device
        pretrained_dict = torch.load(pretrained_path, map_location=device)

        print("---start load pretrained encoder---")

        model_dict = self.model.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print(
                        "delete:{};shape pretrain:{};shape model:{}".format(
                            k, full_dict[k].shape, model_dict[k].shape
                        )
                    )
                    del full_dict[k]

        self.model.load_state_dict(full_dict, strict=False)

