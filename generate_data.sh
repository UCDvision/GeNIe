#!/bin/sh

echo "run on gpu $1: start seed $2 : end seed $3"

DATASET_DIR="./datasets/mini_imagenet_fs"


# Mini ImageNet
CUDA_VISIBLE_DEVICES=$1 python genie.py  --output_dir $DATASET_DIR/5way_1shot/  $DATASET_DIR/5way_1shot/ --kshot 1 --nway 5 --noise 0.3 --start_seed $2 --end_seed $3 --method img2img

CUDA_VISIBLE_DEVICES=$1 python genie.py  --output_dir $DATASET_DIR/5way_1shot/  $DATASET_DIR/5way_1shot/ --kshot 1 --nway 5 --noise 0.7 --start_seed $2 --end_seed $3 --method img2img

CUDA_VISIBLE_DEVICES=$1 python genie.py  --output_dir $DATASET_DIR/5way_1shot/  $DATASET_DIR/5way_1shot/ --kshot 1 --nway 5 --noise 0.8 --start_seed $2 --end_seed $3 --method genie

CUDA_VISIBLE_DEVICES=$1 python genie.py  --output_dir $DATASET_DIR/5way_1shot/  $DATASET_DIR/5way_1shot/ --kshot 1 --nway 5 --noise 0.8 --start_seed $2 --end_seed $3 --method txt2img


CUDA_VISIBLE_DEVICES=$1 python genie.py  --output_dir $DATASET_DIR/5way_5shot/  $DATASET_DIR/5way_5shot/ --kshot 5 --nway 5 --noise 0.3 --start_seed $2 --end_seed $3 --method img2img

CUDA_VISIBLE_DEVICES=$1 python genie.py  --output_dir $DATASET_DIR/5way_5shot/  $DATASET_DIR/5way_5shot/ --kshot 5 --nway 5 --noise 0.7 --start_seed $2 --end_seed $3 --method img2img

CUDA_VISIBLE_DEVICES=$1 python genie.py  --output_dir $DATASET_DIR/5way_5shot/  $DATASET_DIR/5way_5shot/ --kshot 5 --nway 5 --noise 0.8 --start_seed $2 --end_seed $3 --method genie

CUDA_VISIBLE_DEVICES=$1 python genie.py  --output_dir $DATASET_DIR/5way_5shot/  $DATASET_DIR/5way_5shot/ --kshot 5 --nway 5 --noise 0.8 --start_seed $2 --end_seed $3 --method txt2img



DATASET_DIR="./datasets/tiered_imagenet_fs"

# tiered-ImageNet
CUDA_VISIBLE_DEVICES=$1 python genie.py  --output_dir $DATASET_DIR/5way_1shot/  $DATASET_DIR/5way_1shot/ --kshot 1 --nway 5 --noise 0.3 --start_seed $2 --end_seed $3 --method img2img

CUDA_VISIBLE_DEVICES=$1 python genie.py  --output_dir $DATASET_DIR/5way_1shot/  $DATASET_DIR/5way_1shot/ --kshot 1 --nway 5 --noise 0.7 --start_seed $2 --end_seed $3 --method img2img

CUDA_VISIBLE_DEVICES=$1 python genie.py  --output_dir $DATASET_DIR/5way_1shot/  $DATASET_DIR/5way_1shot/ --kshot 1 --nway 5 --noise 0.8 --start_seed $2 --end_seed $3 --method genie

CUDA_VISIBLE_DEVICES=$1 python genie.py  --output_dir $DATASET_DIR/5way_1shot/  $DATASET_DIR/5way_1shot/ --kshot 1 --nway 5 --noise 0.8 --start_seed $2 --end_seed $3 --method txt2img


# tiered-ImageNet
CUDA_VISIBLE_DEVICES=$1 python genie.py  --output_dir $DATASET_DIR/5way_5shot/  $DATASET_DIR/5way_5shot/ --kshot 5 --nway 5 --noise 0.3 --start_seed $2 --end_seed $3 --method img2img

CUDA_VISIBLE_DEVICES=$1 python genie.py  --output_dir $DATASET_DIR/5way_5shot/  $DATASET_DIR/5way_5shot/ --kshot 5 --nway 5 --noise 0.7 --start_seed $2 --end_seed $3 --method img2img

CUDA_VISIBLE_DEVICES=$1 python genie.py  --output_dir $DATASET_DIR/5way_5shot/  $DATASET_DIR/5way_5shot/ --kshot 5 --nway 5 --noise 0.8 --start_seed $2 --end_seed $3 --method genie

CUDA_VISIBLE_DEVICES=$1 python genie.py  --output_dir $DATASET_DIR/5way_5shot/  $DATASET_DIR/5way_5shot/ --kshot 5 --nway 5 --noise 0.8 --start_seed $2 --end_seed $3 --method txt2img
