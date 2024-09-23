#!/bin/sh -l
dataset=imagenet
policy=expanded
gpu_id=6
MODELS=(
    resnet18
    resnet50
    mobilenet_v2
    inception_v3
    resnext50_32x4d
    shufflenet_v2_x1_0
    swin_t
    swin_v2_t
)
TTA=(
    MeanTTA
    MaxTTA
    GPS
    ClassTTA
    AugTTA
)

mkdir -p results

for tta in ${TTA[@]}; do
    for model in ${MODELS[@]}; do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 main_cls_tta.py \
            --seed 42 --tta $tta \
            --config configs/cls/torchvision/$model.yaml \
            DATA.NAME $dataset \
            TTA.POLICY $policy \
            MODEL.BACKBONE $model
    done
done