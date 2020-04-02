#CUDA_VISIBLE_DEVICES=0 /home-ex/tclhk/anaconda3/envs/pycenternet/bin/freeze_graph \
#--input_graph=/home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/inception_resnet_v2_inf_graph_5cls.pb  \
#--input_checkpoint=/home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1126_center_2W+1W+1W/model.ckpt-10000   \
#--output_graph=/home-ex/tclhk/chenww/tensorflow-model/research/MASK_MODEL/1126_center_2W+1W+1W/frozen_inception_resnet_v2_40000.pb   \
#--input_binary=true \
#--output_node_names=InceptionResnetV2/Logits/Predictions

###################################################    mura2Cls    ###################################################################
CUDA_VISIBLE_DEVICES=3 /home-ex/tclhk/anaconda3/envs/pycenternet/bin/freeze_graph \
--input_graph=/home-ex/tclhk/chenww/MY_TRAINING/T4/model/mura2Cls/inception_resnet_v2_inf_graph_mura2cls.pb  \
--input_checkpoint=/home-ex/tclhk/chenww/MY_TRAINING/T4/model/mura2Cls/1126/model.ckpt-13203   \
--output_graph=/home-ex/tclhk/chenww/MY_TRAINING/T4/model/mura2Cls/1126/frozen_inception_resnet_v2_13203.pb   \
--input_binary=true \
--output_node_names=InceptionResnetV2/Logits/Predictions

#


