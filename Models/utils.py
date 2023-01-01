from typing import List
from pyparsing import col
import torch

import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms import Compose, ToPILImage, PILToTensor, Normalize
from monai.visualize.utils import blend_images
from monai.transforms import rescale_array, repeat
from einops import rearrange
from monai.metrics import compute_meandice, compute_hausdorff_distance
from torchmetrics.functional import jaccard_index
from monai.inferers import sliding_window_inference

CITYSCAPES_CLASS_COLOURS = [
    (0, 0, 0),
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]


def log_image(logger, stage, step_no, prediction, batch, device):
    writer = logger.experiment
    im_transforms = Compose(
        [
            Denormalize(
                mean=[0.28689554, 0.32513303, 0.28389177]
                if "Cityscapes" in logger.name
                else [0.7746, 0.5997, 0.8150],
                std=[0.18696375, 0.19017339, 0.18720214]
                if "Cityscapes" in logger.name
                else [1.1175, 1.1769, 1.2329],
            ),
            ToPILImage(),
            PILToTensor(),
        ]
    )

    x, y = batch
    image = im_transforms(x[0])
    gt = y[0].bool()

    normalized_masks = F.softmax(prediction[0], dim=0)

    class_dim = 0
    num_classes = normalized_masks.shape[0]
    all_classes_masks = (
        normalized_masks.argmax(class_dim)
        == torch.arange(num_classes, device=device)[:, None, None]
    )

    prediction_with_masks = draw_segmentation_masks(
        image,
        masks=all_classes_masks,
        alpha=0.5,
        colors=CITYSCAPES_CLASS_COLOURS
        if "CityScapes" in logger.name or "NTNU" in logger.name
        else None,
    )

    writer.add_image(
        f"{stage}/Example prediction", prediction_with_masks, global_step=step_no
    )
    if step_no == 0 or stage == "test":
        gt_with_masks = draw_segmentation_masks(
            image,
            masks=gt,
            alpha=0.5,
            colors=CITYSCAPES_CLASS_COLOURS
            if "CityScapes" in logger.name or "NTNU" in logger.name
            else None,
        )
        writer.add_image(f"{stage}/Ground truth", gt_with_masks, global_step=step_no)
        writer.add_image(f"{stage}/Raw image", image, global_step=step_no)


def log_gif(logger, stage, epoch_no, prediction, batch, device):
    writer = logger.experiment

    x, y = batch
    x = x[0]
    y = y[0]

    if "BraTS" in logger.name or "Task01" in logger.name:
        y = prepeare_brats_label_for_viz(y, device)
        pred_img = prepeare_brats_label_for_viz(prediction[0], device)

    else:
        y = torch.unsqueeze(torch.argmax(y, dim=0), 0)
        pred_img = torch.unsqueeze(torch.argmax(prediction[0], dim=0), 0)

    if x.shape[0] == 1:
        blended_gt, blended_pred, prepped_x = prepeare_images_for_logging(
            pred_img, x, y
        )
        if epoch_no == 0 or stage == "test":
            writer.add_video(
                tag=f"{stage}/Raw image",
                vid_tensor=prepped_x,
                fps=24,
                global_step=epoch_no,
            )
            writer.add_video(
                tag=f"{stage}/Ground truth",
                vid_tensor=blended_gt,
                fps=24,
                global_step=epoch_no,
            )
        writer.add_video(
            tag=f"{stage}/Prediction",
            vid_tensor=blended_pred,
            fps=24,
            global_step=epoch_no,
        )
    else:
        for i in range(x.shape[0]):
            image = torch.unsqueeze(x[i], 0)
            blended_gt, blended_pred, prepped_x = prepeare_images_for_logging(
                pred_img, image, y
            )
            if epoch_no == 0 or stage == "test":
                writer.add_video(
                    tag=f"{stage}/Raw image/ch{i}",
                    vid_tensor=prepped_x,
                    fps=24,
                    global_step=epoch_no,
                )
                writer.add_video(
                    tag=f"{stage}/Ground truth/ch{i}",
                    vid_tensor=blended_gt,
                    fps=24,
                    global_step=epoch_no,
                )
            writer.add_video(
                tag=f"{stage}/Prediction/ch{i}",
                vid_tensor=blended_pred,
                fps=24,
                global_step=epoch_no,
            )


def prepeare_brats_label_for_viz(y, device):
    et = y[0].bool()
    net = torch.logical_and(y[1], torch.logical_not(et))
    ed = torch.logical_and(y[2], torch.logical_not(y[1]))

    new_y = torch.zeros(y.shape[1:], device=device)
    new_y[et] = 4
    new_y[net] = 1
    new_y[ed] = 2
    y = torch.unsqueeze(new_y, 0)
    return y


def prepeare_images_for_logging(pred_img, x, y):
    blended_gt = blend_images(x, y)
    blended_pred = blend_images(x, pred_img)
    blended_gt = rearrange(blended_gt, "c h w d -> () d c w h")
    blended_pred = rearrange(blended_pred, "c h w d -> () d c w h")
    rearranged_x = rearrange(repeat(rescale_array(x), 3, 0), "c h w d -> () d c w h")
    return blended_gt, blended_pred, rearranged_x


class Denormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, x):
        for t, m, s in zip(x, self.mean, self.std):
            t.mul_(s).add_(m)
        return torch.clamp(x, 0, 1)


def calculate_dice(pred: torch.Tensor, target: torch.Tensor, include_background: bool):
    raw_dice_scores = compute_meandice(pred, target, include_background)
    for i in range(raw_dice_scores.shape[0]):
        for j in range(raw_dice_scores[i].shape[0]):
            if torch.isnan(raw_dice_scores[i][j]):
                if torch.sum(pred[i][j]) == torch.sum(target[i][j]) == 0:
                    raw_dice_scores[i][j] = 1
                else:
                    raw_dice_scores[i][j] = 0

    return raw_dice_scores


def calculate_HD(pred: torch.Tensor, target: torch.Tensor, include_background: bool):
    raw_hd_scores = compute_hausdorff_distance(
        pred, target, include_background, percentile=95
    )
    for i in range(raw_hd_scores.shape[0]):
        for j in range(raw_hd_scores[i].shape[0]):
            if torch.isnan(raw_hd_scores[i][j]) or torch.isinf(raw_hd_scores[i][j]):
                raw_hd_scores[i][j] = torch.nan

    return raw_hd_scores


available_metrics = {
    "Dice": lambda include_background: lambda pred, target: torch.mean(
        calculate_dice(pred, target, include_background), dim=0
    ),
    "IoU": lambda include_background: lambda pred, target: jaccard_index(
        torch.argmax(pred, dim=1),
        torch.argmax(target, dim=1),
        ignore_index=0 if not include_background else None,
        reduction="none",
    ),
    "HD": lambda include_background: lambda pred, target: torch.mean(
        calculate_HD(pred, target, include_background), dim=0
    ),
}


def get_metrics(metrics: List[str], include_background: bool = False):
    out = []
    for metric in metrics:
        out.append(available_metrics[metric](include_background=include_background))

    return out


def step_log(log, stage, loss, metrics, metric_names, labels):
    log(
        f"{stage}/loss",
        loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
        sync_dist=True,
    )

    if metrics is None:
        return

    for name, metric in zip(metric_names, metrics):

        avg_score = torch.nanmean(metric)
        if not torch.isnan(avg_score):
            log(
                f"{stage}/{name}",
                avg_score,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        for label_metric, label in zip(metric, labels):
            if torch.isnan(label_metric):
                continue
            log(
                f"{stage}/{label}_{name}", label_metric, on_epoch=True, sync_dist=True,
            )


def common_predict_step(batch, roi_size, predictor, post_trans):
    x = batch["image"]
    y_hat = sliding_window_inference(
        inputs=x,
        roi_size=roi_size,
        sw_batch_size=x.shape[0],
        predictor=predictor,
        overlap=0.5,
    )

    y_hat = torch.stack([post_trans(i) for i in y_hat])

    batch["pred"] = y_hat

    return batch


if __name__ == "__main__":
    pred = torch.randint(high=2, low=0, size=(1, 20, 32, 32, 32))
    target = torch.randint(high=2, low=0, size=(1, 20, 32, 32, 32))

    target[0][1][:][:] = 0
    pred[0][1][:][:] = 1

    calculate_HD(pred, target, True)

