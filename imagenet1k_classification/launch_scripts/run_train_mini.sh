ROOT=$1
CODE_BASE=$2
TAG=gr_vit_mini

cd $CODE_BASE &&  python3 -m torch.distributed.launch --nproc_per_node 4 --nnodes 1 --use_env /opt/data/private/zhousai/imagenet1k_classification/train.py \
  --data-path $ROOT/opt/data/private/zhousai/imagenet \
  --batch-size 256 \
  --output $ROOT/opt/data/private/zhousai/output_grvit \
  --cfg /opt/data/private/zhousai/imagenet1k_classification/configs/gr_vit_mini.yaml \
  --model-type gr_vit \
  --model-file GR_ViT.py \
  --tag $TAG
