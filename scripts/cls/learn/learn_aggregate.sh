seed=42
dataset=imagenet
MODELS=(
    torchvision/resnet18
    torchvision/resnet50
    torchvision/mobilenet_v2
    torchvision/inception_v3
    torchvision/resnext50_32x4d
    torchvision/shufflenet_v2_x1_0
    torchvision/swin_t
    torchvision/swin_v2_t
)
gpu_id=7

for model in ${MODELS[@]};
do
    CUDA_VISIBLE_DEVICES=$gpu_id python3 train_poolsearch.py \
        --config configs/cls/$model.yaml \
        --seed $seed \
        --evaluate-split val \
        DATA.NAME $dataset \
        SEARCH.BUDGET 30 \
        SEARCH.AGGREGATE.ALIGN True \
        SEARCH.AGGREGATE.FN learn \
        SEARCH.CRITERION random-max_learn \
        SEARCH.OPTIM.MAX_EPOCH 5 \
        SEARCH.OPTIM.LR 1e-6 \
        DATALOADER.BATCH_SIZE 32 \
        ${@:1}
done
