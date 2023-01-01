from argparse import Namespace

from .VT_UNet import VTUNet
from .VT_UNet_A import VTUNetA
from .QT_UNet import QTUNet
from .QT_UNet_A import QTUNetA

models = {
    "QT-UNet": QTUNet,
    "QT-UNet-A": QTUNetA,
    "VT-UNet": VTUNet,
    "VT-UNet-A": VTUNetA,
}


def get_model(selected_model, model_args, dm, max_epochs, metric_args):
    if selected_model in [
        "QT-UNet",
        "QT-UNet-A",
        "VT-UNet",
        "VT-UNet-A",
    ]:
        common = model_args.common

        args_dict = vars(model_args)

        del args_dict["common"]

        combined = {**common, **args_dict}

        model_args = Namespace(**combined)

    model = models[selected_model](
        args=model_args,
        metric_args=metric_args,
        input_shape=dm.input_dims,
        output_shape=dm.target_dims,
        num_classes=dm.num_classes,
        labels=dm.labels,
        max_epochs=max_epochs,
    )

    return model
