from typing import Any, Tuple
from copy import deepcopy
from functools import reduce

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam

import pytorch_lightning as pl
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from Models.QT_UNet.components import SwinTransformerSys3D


torch.backends.cudnn.enabled = False


class SSLHarness(pl.LightningModule):
    """PyTorch Lightning harness for SSL, using BYOL, reconstruction loss and rotation loss.

    Based on PTLightnings BYOL module and the procedure described in Swin-UNETR.

    Assumes inputs from datasource are square/perfect qubes.
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        encoder_out_dim: Tuple[int, ...],
        in_data_dim: Tuple[int, ...],
        learning_rate: float = 4e-4,
        weight_decay: float = 1.5e-6,
        warmup_epochs: int = 10,
        max_epochs: int = 1000,
        projector_hidden_size: int = 4096,
        projector_out_dim: int = 256,
        compress_encoder_output: bool = False,
    ):
        super(SSLHarness, self).__init__()
        self.save_hyperparameters(ignore="base_encoder")

        if compress_encoder_output:
            flattened_encoder_dim = encoder_out_dim[0]
            for i in range(1, len(encoder_out_dim)):
                flattened_encoder_dim *= encoder_out_dim[i] // 4
        else:
            flattened_encoder_dim = reduce(lambda x, y: x * y, encoder_out_dim)

        self.online_network = SiameseArm(
            base_encoder,
            flattened_encoder_dim,
            projector_hidden_size,
            projector_out_dim,
            compress_encoder_output,
        )
        self.target_network = deepcopy(self.online_network)
        self.weight_callback = BYOLMAWeightUpdate()

        stride = 32
        kernel_size = in_data_dim[1] - (encoder_out_dim[1] - 1) * stride
        self.reconstruction_head = nn.ConvTranspose3d(
            encoder_out_dim[0], in_data_dim[0], kernel_size, stride
        )

        self.reconstruction_loss = nn.L1Loss()

        self.rotation_head = nn.Sequential(
            nn.Flatten(), MLP(flattened_encoder_dim, output_dim=4,),
        )
        self.rotation_loss = nn.CrossEntropyLoss()

    def on_train_batch_end(
        self, outputs, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self.weight_callback.on_train_batch_end(
            self.trainer, self, outputs, batch, batch_idx, dataloader_idx
        )

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def shared_step(self, batch, batch_idx):
        k_1, k_2, img_1, img_2, raw_img = batch

        # Image 1 to image 2 loss
        y1_1_orig, y1_1, z1_1, h1_1 = self.online_network(img_1)
        with torch.no_grad():
            y1_2_orig, y1_2, z1_2, h1_2 = self.target_network(img_2)
        loss_a = -2 * F.cosine_similarity(h1_1, z1_2).mean()

        # Image 2 to image 1 loss
        y2_1_orig, y2_1, z2_1, h2_1 = self.online_network(img_2)
        with torch.no_grad():
            y2_2_orig, y2_2, z2_2, h2_2 = self.target_network(img_1)
        # L2 normalize
        loss_b = -2 * F.cosine_similarity(h2_1, z2_2).mean()

        # Reconstruction
        img_1_rec = self.reconstruction_head(y1_1_orig)
        img_2_rec = self.reconstruction_head(y2_1_orig)

        img_1_rec_loss = self.reconstruction_loss(img_1_rec, raw_img)
        img_2_rec_loss = self.reconstruction_loss(img_2_rec, raw_img)
        rec_loss = img_1_rec_loss + img_2_rec_loss

        # Rotation
        img_1_rot = self.rotation_head(y1_1)
        img_2_rot = self.rotation_head(y2_1)

        img_1_rot_loss = self.rotation_loss(img_1_rot, k_1)
        img_2_rot_loss = self.rotation_loss(img_2_rot, k_2)
        rot_loss = img_1_rot_loss + img_2_rot_loss

        # Final loss
        total_loss = loss_a + loss_b + rec_loss + rot_loss

        return loss_a, loss_b, rec_loss, rot_loss, total_loss

    def training_step(self, batch, batch_idx):
        loss_a, loss_b, rec_loss, rot_loss, total_loss = self.shared_step(
            batch, batch_idx
        )

        # log results
        self.log_dict(
            {
                "train/1_2_loss": loss_a,
                "train/2_1_loss": loss_b,
                "train/rec_loss": rec_loss,
                "train/rot_loss": rot_loss,
                "train/train_loss": total_loss,
            }
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        loss_a, loss_b, rec_loss, rot_loss, total_loss = self.shared_step(
            batch, batch_idx
        )

        # log results
        self.log_dict(
            {
                "val/1_2_loss": loss_a,
                "val/2_1_loss": loss_b,
                "val/rec_loss": rec_loss,
                "val/rot_loss": rot_loss,
                "val/train_loss": total_loss,
            }
        )

        return total_loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
        )
        return [optimizer], [scheduler]


class MLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_size=4096, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SiameseArm(nn.Module):
    def __init__(
        self,
        encoder="resnet50",
        encoder_out_dim=2048,
        projector_hidden_size=4096,
        projector_out_dim=256,
        compress_encoder_output=False,
    ):
        super().__init__()
        self.compress_encoder_output = compress_encoder_output

        if isinstance(encoder, str):
            encoder = torchvision_ssl_encoder(encoder)
        # Encoder
        self.encoder = encoder
        # Projector
        self.projector = MLP(encoder_out_dim, projector_hidden_size, projector_out_dim)
        # Predictor
        self.predictor = MLP(
            projector_out_dim, projector_hidden_size, projector_out_dim
        )

    def forward(self, x):
        y_orig = self.encoder(x)[0]
        if self.compress_encoder_output:
            y = F.interpolate(y_orig, scale_factor=0.25, mode="bilinear")
        else:
            y = y_orig
        y_flat = torch.flatten(y, start_dim=1)
        z = self.projector(y_flat)
        h = self.predictor(z)
        return y_orig, y, z, h


if __name__ == "__main__":
    x = torch.randn((3, 2, 4, 96, 96, 96)).to("cuda")
    k = torch.randint(low=0, high=2, size=(2, 2, 4)).to("cuda")
    x = [k[0], k[1], x[0], x[1], x[2]]

    model = SwinTransformerSys3D(
        img_size=x[2].shape[2:], in_chans=x[2].shape[1], embed_dim=96
    )
    encoder = model.encoder

    ssl = SSLHarness(encoder, model.encoder_output_dim, x[2].shape[1:]).to("cuda")

    ssl.training_step(x, 0)

