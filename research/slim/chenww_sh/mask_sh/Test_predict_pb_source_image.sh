#################################################         TEST_DEFECT-FREE-ONLY-FILTER               ###################################################################
#CUDA_VISIBLE_DEVICES=0  python ../TEST_predict_with_pb_from_source_image.py \
#--frozen_pb_file=/home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1125_whole_finetune/frozen_inception_resnet_v2_10018.pb \
#--test_dataset_path=/home-ex/tclhk/chenww/datasets/T4/T4_TEST_1000/MASK/M_EDMAI_TXT_CROPED/ \
#--output_all_info=/home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1125_whole_finetune/m_test_filter_obj0-0004767.csv \
#--mask_or_glass=mask \
#--json_file_path=/home-ex/tclhk/eric/T4_update/POC_result/Mask/mask.json \
#--obj_th=0.19028 \
#--cls_th=0
#################################################     TEST_DEFECT-FREE-ONLY_FILTER   Leftdown ##################################################
#CUDA_VISIBLE_DEVICES=0  python ../TEST_predict_with_pb_from_source_image.py \
#--frozen_pb_file=/home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1125_leftdown_finetune_10000/frozen_inception_resnet_v2_10000.pb \
#--test_dataset_path=/home-ex/tclhk/chenww/datasets/T4/T4_TEST_1000/MASK/M_EDMAI_leftdown/  \
#--output_all_info=/home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1125_leftdown_finetune_10000//m_test_filter_obj0-0004767.csv \
#--mask_or_glass=mask \
#--json_file_path=/home-ex/tclhk/eric/T4_update/POC_result/Mask/mask.json \
#--obj_th=0.19028 \
#--cls_th=0
#################################################     TEST_DEFECT-FREE-ONLY_FILTER   center ##################################################
CUDA_VISIBLE_DEVICES=0  python ../TEST_predict_with_pb_from_source_image.py \
--frozen_pb_file=/home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1126_center_2W+1W+1W/frozen_inception_resnet_v2_40000.pb \
--test_dataset_path=/home-ex/tclhk/chenww/datasets/T4/T4_TEST_1000/MASK/M_EDMAI_center/  \
--output_all_info=/home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1126_center_2W+1W+1W/m_test_filter_obj0-0004767.csv \
--mask_or_glass=mask \
--json_file_path=/home-ex/tclhk/eric/T4_update/POC_result/Mask/mask.json \
--obj_th=0.19028 \
--cls_th=0


