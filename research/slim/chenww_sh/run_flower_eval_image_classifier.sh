CUDA_VISIBLE_DEVICES=1 python  eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=/home-ex/tclsz/yangshun/chenww/models-master/research/slim/train_log/train_flowers/model.ckpt-100876  \
    --dataset_dir=/home-ex/tclsz/yangshun/chenww/Dataset/flowers/ \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --model_name=inception_resnet_v2