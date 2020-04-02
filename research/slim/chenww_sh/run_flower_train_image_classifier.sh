DATASET_DIR=/home-ex/tclsz/yangshun/chenww/Dataset/flowers/
TRAIN_DIR=/home-ex/tclsz/yangshun/chenww/models-master/research/slim/train_log/train_flowers
MODEL_NAME=inception_resnet_v2


# CUDA_VISIBLE_DEVICES=3 python train_image_classifier.py \
    # --train_dir=${TRAIN_DIR} \
    # --dataset_name=flowers \
    # --dataset_split_name=train \
    # --dataset_dir=${DATASET_DIR} \
    # --model_name=${MODEL_NAME}
    


CUDA_VISIBLE_DEVICES=3 python train_image_classifier.py \
    --train_dir=/home-ex/tclsz/yangshun/chenww/models-master/research/slim/train_log/holeCls \
    --dataset_dir=/home-ex/tclsz/yangshun/datasets/holeClass_4350 \
    --dataset_name=holeC1s \
    --dataset_split_name=train \
    --model_name=inception_v4 \
    --checkpoint_path=/home-ex/tclsz/yangshun/chenww/models-master/research/slim/models/inception_v4/inception_v4 \
    --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
    --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits