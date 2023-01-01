#!/bin/sh

conda activate MPEx
tensorboard --logdir tb_logs/ --reload_multifile True