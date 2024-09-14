ROOT=$1
CODE_BASE=$2
TAG=gr_vit_mini_throughput

cd $CODE_BASE && CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 train.py \
    --cfg /opt/data/private/zhousai/imagenet1k_classification/configs/gr_vit_mini.yaml \
    --data-path $ROOT/opt/data/private/zhousai/imagenet \
    --batch-size 128 \
    --throughput \
    --disable_amp