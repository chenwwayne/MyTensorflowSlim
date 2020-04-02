CHECKPOINT_FILE=/home-ex/tclhk/chenww/MY_TRAINING/T4/model/mura2Cls/1126/model.ckpt-11601
DATASET_DIR=/home-ex/tclhk/chenww/datasets/T4/mura/2cls/validation
CUDA_VISIBLE_DEVICES=3 python ../eval_image_classifier.py \
--alsologtostderr \
--checkpoint_path=${CHECKPOINT_FILE} \
--dataset_dir=${DATASET_DIR} \
--dataset_name=mura2Cls \
--dataset_split_name=validation \
--model_name=inception_resnet_v2