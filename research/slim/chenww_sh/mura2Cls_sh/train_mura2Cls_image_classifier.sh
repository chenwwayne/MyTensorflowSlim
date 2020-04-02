DATASET_DIR=/home-ex/tclhk/chenww/datasets/T4/mura/2cls/train
TRAIN_DIR=/home-ex/tclhk/chenww/MY_TRAINING/T4/model/mura2Cls/1126/
CHECKPOINT_PATH=/home-ex/tclhk/chenww/tensorflow-model/research/slim/models/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt
CUDA_VISIBLE_DEVICES=1 python ../../train_image_classifier.py \
--train_dir=${TRAIN_DIR} \
--dataset_dir=${DATASET_DIR} \
--dataset_name=mura2Cls \
--dataset_split_name=train \
--model_name=inception_resnet_v2 \
--checkpoint_path=${CHECKPOINT_PATH} \
--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--save_interval_secs=600 \
2>&1 | tee  /home-ex/tclhk/chenww/MY_TRAINING/T4/model/mura2Cls/1126/log.txt &
#--max_number_of_steps=30000 \