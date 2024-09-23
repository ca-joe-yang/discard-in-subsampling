#!/usr/bin/env bash

imagenet_dir="${1:-.}" # Directory that contains ILSVRC2012_img_train.tar or ILSVRC2012_img_val.tar

bash tools/prepare_imagenet_val.sh $imagenet_dir
bash tools/prepare_imagenet_train.sh $imagenet_dir