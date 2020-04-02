#######################################  whole #########################################################
#CHECKPOINT_FILE=/home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1125_whole_finetune/model.ckpt-9109
#DATASET_DIR=/home-ex/tclhk/chenww/MY_TRAINING/T4/DATASET/MASK/crop_1024_718/validation/
#CUDA_VISIBLE_DEVICES=0 python ../eval_image_classifier.py \
#--alsologtostderr \
#--checkpoint_path=${CHECKPOINT_FILE} \
#--dataset_dir=${DATASET_DIR} \
#--dataset_name=maskCls \
#--dataset_split_name=validation \
#--model_name=inception_resnet_v2
#######################################  leftdown #########################################################
#CHECKPOINT_FILE=/home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1125_leftdown_finetune_10000/model.ckpt-10000
#DATASET_DIR=/home-ex/tclhk/chenww/MY_TRAINING/T4/DATASET/MASK/crop_left_down/validation/
#CUDA_VISIBLE_DEVICES=0 python ../eval_image_classifier.py \
#--alsologtostderr \
#--checkpoint_path=${CHECKPOINT_FILE} \
#--dataset_dir=${DATASET_DIR} \
#--dataset_name=maskCls \
#--dataset_split_name=validation \
#--model_name=inception_resnet_v2
#######################################  center #########################################################
CHECKPOINT_FILE=/home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1126_center_2W+1W+1W/model.ckpt-10000
DATASET_DIR=/home-ex/tclhk/chenww/MY_TRAINING/T4/DATASET/MASK/crop_center/validation/
CUDA_VISIBLE_DEVICES=0 python ../eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=${CHECKPOINT_FILE} \
--dataset_dir=${DATASET_DIR} \
--dataset_name=maskCls \
--dataset_split_name=validation \
--model_name=inception_resnet_v2
