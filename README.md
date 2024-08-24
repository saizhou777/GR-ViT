# GR-ViT
will be submitted to ...


## 🌟 1. Trianing Scripts
To train GR-ViT-mini on the custom plankton dataset with one gpu, please run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

To train GR-ViT-mini on the ImageNet-1K dataset with four gpus, please run:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node 4 --nnodes 1 --use_env train.py --data-path /opt/data/private/zhousai/imagenet --batch-size 256 --output /opt/data/private/zhousai/output_grvit --cfg /opt/data/private/zhousai/imagenet1k_classification/configs/gr_vit_mini.yaml --model-type gr_vit --model-file GR_ViT.py --tag gr_vit_mini
```


## ✨ 2. Inference Scripts
To eval GR-ViT-mini on the custom plankton dataset on a single gpu, please identify the path of pretrained weight and run:
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py
```
This will give
```bash
Validation Loss: 0.1782, Validation Accuracy: 0.9517
```


## 👏 3. Acknowledgement
This repository is built using the [Groupmixformer](https://github.com/AILab-CVC/GroupMixFormer) and [deep-learning-for-image-processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing) repositories. We particularly appreciate their open-source efforts.


## 📖 4. Citation
If you find this repository helpful, please consider citing:
```bash
@Article{xxx
}
```
