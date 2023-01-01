from Experiments import run_experiment
import os
from argparse import ArgumentParser, Namespace

import yaml

from yamlinclude import YamlIncludeConstructor

from Infer import run_inference

YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.FullLoader, base_dir="./Configs"
)


def parse_args_and_get_dict() -> dict:
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default="./Configs/base.yaml")
    parser.add_argument("-t", "--test", default=False, action="store_true")
    parser.add_argument("-s", "--ssl", default=False, action="store_true")
    parser.add_argument("-d", "--dev", default=False, action="store_true")
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument("-r", "--resume_from", default=None)
    parser.add_argument("-a", "--task", default=None)
    parser.add_argument("-p", "--pretrained_encoder", default=None)
    parser.add_argument("-i", "--infer", default=False, action="store_true")
    parser.add_argument("-o", "--output_path", default=None)
    args = parser.parse_args()

    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, args.config.strip())

    if not os.path.isfile(path):
        raise ValueError(f"{args.config} is not a valid file path")

    with open(path) as file:
        print(f"Loading config from {args.config}")
        config = yaml.full_load(file)

    if args.model is not None and os.path.isfile(args.model):
        with open(args.model) as file:
            model_config = yaml.full_load(file)

        config["model"] = model_config

    return args, config


if __name__ == "__main__":
    args, config = parse_args_and_get_dict()
    if "model_extra" in config:
        model_args = Namespace(**config["model"], **config["model_extra"])
    else:
        model_args = Namespace(**config["model"])

    if args.pretrained_encoder is not None:
        model_args.pretrained_encoder_checkpoint = args.pretrained_encoder

    data_args = Namespace(**config["data"])
    trainer_args = config["trainer"]
    metric_args = config["metrics"]
    selected_experiment = config["experiment"]

    if args.test and args.ssl:
        raise ValueError(
            "Invalid option configuration: Cannot run with both test and SSL flags enabled."
        )

    if args.infer:
        run_inference(
            selected_experiment,
            args.dev,
            args.task,
            model_args.type,
            args.resume_from,
            args.output_path,
            data_args,
            trainer_args,
            metric_args,
        )
    else:
        run_experiment(
            selected_experiment,
            args.test,
            args.ssl,
            args.dev,
            args.resume_from,
            args.task,
            model_args.type,
            model_args,
            data_args,
            trainer_args,
            metric_args,
        )
