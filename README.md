# QT-UNet

Getting started requires [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to be installed, and you need to be on a Linux distro. Once available from the command line, simply run the following command from the root of the project:

```shell
     conda env create -f environment.yml
```

This will download and install all the required packages to run the models. Applying the following command will activate the Conda environment and allow the project files to be executed:

```shell
     conda activate QT-UNet
```

Data for the CT-SSL and MSD datasets can be downloaded using the included download scripts. The remaining datasets must be downloaded manually from their respective portals. Please pay special attention to the path variables in the datamodules and downloader files, as they control where the system looks for data and downloads it to.

To perform inference with for example BraTS and QT-UNet-B, invoke the following command:

```shell
     python main.py -i -c Configs/Experiments/BraTS.yaml -m Configs/Models/QT-UNet/Base.yaml -r <<Path to QT-UNet-B checkpoint>> -o <<desired output folder>>
```

Note that the model specified in the configuration file passed to the "-m" flag and the model in the checkpoint passed to the "-r" flag must match each-other. Predictions are zipped after inference for ease of transfer.

To perform model training from scratch, issue the following command:

```shell
     python main.py -c <<Path to desired experiment config file>> -m <<Path to desired model config file>>

     // A example for training QT-UNet-B on BraTS would look like
     python main.py -c Configs/Experiments/BraTS.yaml -m Configs/Models/QT-UNet/Base.yaml
```

To perform training with a pretrained encoder, simply add the "-p" flag with a path to the desired pretrained weights. Note that the model supplied in "-m" and the model the weights in "-p" were trained on must match.

Example again with QT-UNet and BraTS:

```shell
     python main.py -c Configs/Experiments/BraTS.yaml -m Configs/Models/QT-UNet/Base.yaml -p Pretrained_weights/Base/BraTS-SSL_pretrained.pt
```
