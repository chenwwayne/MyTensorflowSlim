#python ../export_inference_graph.py  \
#--model_name=inception_resnet_v2 \
#--output_file=/home-ex/tclhk/chenww/MY_TRAINING/T4/DATASET/MASK/inception_resnet_v2_inf_graph_5cls.pb \
#--alsologtostderr \
#--dataset_name=maskCls
#########################################  mura2Cls  ##########################################################
python ../export_inference_graph.py  \
--model_name=inception_resnet_v2 \
--output_file=/home-ex/tclhk/chenww/MY_TRAINING/T4/model/mura2Cls/inception_resnet_v2_inf_graph_mura2cls.pb \
--alsologtostderr \
--dataset_name=mura2Cls


