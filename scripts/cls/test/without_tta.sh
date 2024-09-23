#!/bin/sh -l
dataset=imagenet
gpu_id=6
MODELS=(
    timm/resnet18
    timm/resnet50
    timm/convnext_v2
    timm/efficientnet
    timm/efficientformer
    timm/mobilenet_v2
    timm/mobilenet_v3
    timm/mobilevit_v2
    timm/pvt_v2
)
BUDGETS=$(seq 1 16)

mkdir -p results

for model in ${MODELS[@]}; do
    for budget in ${BUDGETS[@]}; do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 main_cls.py \
            --seed 42 \
            --config configs/cls/$model.yaml \
            DATA.NAME $dataset \
            SEARCH.CRITERION min_entropy \
            SEARCH.AGGREGATE.ALIGN True \
            SEARCH.AGGREGATE.OP entropy \
            SEARCH.BUDGET ${budget}
    done
done