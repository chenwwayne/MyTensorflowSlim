######################################### fine tune initialization #############################################
#DATASET_DIR=/home-ex/tclhk/chenww/MY_TRAINING/T4/DATASET/MASK/crop_1024_718/train/
#TRAIN_DIR=/home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1125_whole_finetune/
#CHECKPOINT_PATH=/home-ex/tclhk/chenww/tensorflow-model/research/slim/models/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt
#CUDA_VISIBLE_DEVICES=0 python ../train_image_classifier.py \
#--train_dir=${TRAIN_DIR} \
#--dataset_dir=${DATASET_DIR} \
#--dataset_name=maskCls \
#--dataset_split_name=train \
#--model_name=inception_resnet_v2 \
#--checkpoint_path=${CHECKPOINT_PATH} \
#--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
#--max_number_of_steps=30000 \
#2>&1 | tee  /home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1125_whole_finetune/log.txt &

########################################### fine tune from imgnet ckpt ######################################################
#DATASET_DIR=/home-ex/tclhk/chenww/MY_TRAINING/T4/DATASET/MASK/crop_center/train/
#TRAIN_DIR=/home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1126_center_finetune_10000/
#CHECKPOINT_PATH=/home-ex/tclhk/chenww/tensorflow-model/research/slim/models/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt
#CUDA_VISIBLE_DEVICES=1 python ../train_image_classifier.py \
#--train_dir=${TRAIN_DIR} \
#--dataset_dir=${DATASET_DIR} \
#--dataset_name=maskCls \
#--dataset_split_name=train \
#--model_name=inception_resnet_v2 \
#--checkpoint_path=${CHECKPOINT_PATH} \
#--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
#--max_number_of_steps=10000 \
#2>&1 | tee  /home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1126_center_finetune_10000/log.txt &

########################################### fine tune from exist ckpt ######################################################
DATASET_DIR=/home-ex/tclhk/chenww/MY_TRAINING/T4/DATASET/MASK/crop_center/train/
TRAIN_DIR=/home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1126_center_2W+1W+1W_without_exclude/
CHECKPOINT_PATH=/home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1126_center_finetune_20000+10000/model.ckpt-10000
CUDA_VISIBLE_DEVICES=0 python ../train_image_classifier.py \
--train_dir=${TRAIN_DIR} \
--dataset_dir=${DATASET_DIR} \
--dataset_name=maskCls \
--dataset_split_name=train \
--model_name=inception_resnet_v2 \
--checkpoint_path=${CHECKPOINT_PATH} \
#--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
--max_number_of_steps=10000 \
2>&1 | tee  /home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1126_center_2W+1W+1W_without_exclude/log.txt &
