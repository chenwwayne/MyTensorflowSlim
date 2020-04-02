CUDA_VISIBLE_DEVICES=3  python ../../TEST_mura2Cls_predict_with_pb_from_source_image.py \
--frozen_pb_file=/home-ex/tclhk/chenww/MY_TRAINING/T4/model/mura2Cls/1126/frozen_inception_resnet_v2_13203.pb  \
--result_img_clsfy_folder=/home-ex/tclhk/chenww/MY_TRAINING/T4/model/mura2Cls/1126/13203_result_img/  \
--test_dataset_path=/home-ex/tclhk/chenww/datasets/T4/mura/ori/test
