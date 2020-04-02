import tensorflow as tf
import numpy as np

import os
import sys
import pdb
import time
import json
import cv2
import csv
import shutil

from tqdm import tqdm
from preprocessing import inception_preprocessing
from tensorflow.python.platform import gfile
from collections import Counter
# from scipy.misc import imread, imresize

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "frozen_pb_file",
    "/home-ex/tclhk/syang/tensorflow/research/train_maskCls_uni/maskCls_inceptionResnetV2_40000_ROI/frozen_inception_resnet_v2_40000.pb","")
flags.DEFINE_string(
    "result_img_clsfy_folder",
    "/home-ex/tclhk/syang/tensorflow/research/train_maskCls_uni/maskCls_inceptionResnetV2_40000_ROI/result_img/",None)
flags.DEFINE_string(
    "test_dataset_path",
    "/home-ex/tclhk/chenww/M_EDMAI_ROI/","")

    # "/home-ex/tclhk/syang/chenww/T4_VAL_FROM_OBJ_DETEC/all_mask_val/","")
# flags.DEFINE_string(
#     "output_all_info",
#     "/home-ex/tclhk/syang/tensorflow/research/train_maskCls_uni/maskCls_inceptionResnetV2_40000_ROI/m_test_info_40000_2.csv","")
# flags.DEFINE_string(
#     "mask_or_glass",
#     'glass',"")
# flags.DEFINE_string(
#     "json_file_path",
#     "","")
############################
# flags.DEFINE_float("obj_th",None,None)
# flags.DEFINE_float("cls_th",None,None)

# def JsonDictToList(json_file_path):
#     with open(json_file_path) as json_file:
#         classes = json.load(json_file)
#     json_labels = [None] * len(classes)
#     for key, value in classes.items():
#         json_labels[value] = key
#     return json_labels


def ShowDataStatistics(mat,labels_name,time):
    colsum = np.sum(mat, axis=0).tolist()
    rowsum = np.sum(mat, axis=1).tolist()
    total = np.sum(colsum, axis=0)
    diag = np.trace(mat)
    recall = np.diagonal(mat) / rowsum
    precision = np.diagonal(mat) / colsum
    acc = float("{0:.2f}".format(diag / total)) * 100

    results = np.array([labels_name, recall, precision]).transpose()

    print('======================= Results ============================')
    print('\n[0] Model name:', FLAGS.frozen_pb_file.split('/')[-2]+'/'+FLAGS.frozen_pb_file.split('/')[-1])
    print('\n[1] Labels:', labels_name)
    print('\n[2] Total test images:', total)
    print('\n[3] Total image processing time: %.2fs' % time)
    print('\n[4] FPS: %.2f' % (total / time))
    print('\n[5] Confusion matrix:')
    confusion_mat = np.vstack([labels_name, mat.transpose()]).transpose()
    line = '\t'
    for i in range(len(labels_name)):
        line += labels_name[i] + '\t'
    print('\t(Predictions)')
    print(line)
    for i in range(confusion_mat.shape[0]):
        line = ''
        for j in range(confusion_mat.shape[1]):
            val = confusion_mat[i][j]
            line += val + '\t'
        print(line)

    print('\n[6] Acc: %.2f%%' % acc)
    print('\n[7] Recall & Precision:')
    print('\tRecall\tPrecision')
    for i in range(results.shape[0]):
        line = ''
        for j in range(results.shape[1]):
            val = results[i][j]
            if j > 0:
                val = float(val) * 100
                val = "{0:.2f}%".format(val)
            line += val + '\t'
            # line += str(results[i][j]) + '\t'
        print(line)
    print('\n=============================================================')

def process_to_sqr(img,x1,y1,x2,y2):
    img_rows=img.shape[0]
    img_cols=img.shape[1]

    length = x2 - x1
    width = y2 - y1
    center = [x1 + round((x2 - x1) / 2), y1 + round((y2 - y1) / 2)]

    extend = round(max(length, width) / 2 + max(length, width) * 0.15)
    new_x1, new_x2, new_y1, new_y2 = center[0] - extend, center[0] + extend, center[1] - extend, center[1] + extend  # try use lambda ,map

    new_x1 = 0 if new_x1 < 0 else new_x1
    new_y1 = 0 if new_y1 < 0 else new_y1
    new_x2 = img_cols if new_x2 > img_cols else new_x2
    new_y2 = img_rows if new_y2 > img_rows else new_y2

    return new_x1, new_y1, new_x2, new_y2


def draw_and_save_dect_box(image, image_name, cls_id, cls, conf, json_f, img_saved_path):
    if not os.path.exists(img_saved_path):
        os.makedirs(img_saved_path)

    with open(json_f) as f:
        impath_class_box_dict = json.load(f)
        for path, mess in impath_class_box_dict.items():
            if mess == []:  # 没有检测框
                continue

            # img_name = path.split('/')[-1]

            xmin, ymin, xmax, ymax = mess[0][0], mess[0][1], mess[0][2], mess[0][3]

            try:
                image.shape
            except:
                print('fail to read xxx.jpg')

            if 1:
                xmin, ymin, xmax, ymax = process_to_sqr(image, xmin, ymin, xmax, ymax)

            bbox = image[ymin:ymax, xmin:xmax]
            rec_image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4)

            font = cv2.FONT_HERSHEY_SUPLEX
            text = '001'
            cv2.putText(rec_image, text, (212, 310), font, 4, (0, 0, 255), 3)
            cv2.imwrite('001_new.jpg', img)
            # cv.imwrite(os.path.join(img_saved_path, img_name), bbox)


def norm_and_batch(image):
    img_batch = []
    single_img_np = cv2.resize(np.array(image), (299, 299))
    single_img_np = single_img_np.astype("float")
    single_img_np /= 127.5
    single_img_np -= 1.
    img_batch.append(single_img_np)
    final_imgs = np.stack(img_batch, axis=0)
    return final_imgs


def predict(sess, image, out_node, in_node='import/input:0'):
    probabilities = sess.run(out_node, {in_node: image})
    probabilities = np.squeeze(probabilities)  # predictions = (batch_size, 4)
    max_probabilities = np.max(probabilities)
    # max_prob_id = probabilities.argsort()[0:1][::-1]  # Getting top 3 predictions, reverse order
    max_prob_id = np.argmax(probabilities)
    # answer = batch_reader.labels[top_k[0]]
    return probabilities, max_probabilities, max_prob_id

def get_bbox_and_predict(json_f, out_node):
    mask_class_list = ['EMDFBM', 'EMDFJS', 'EMDFKL', 'EMDFET', 'EMDFFD', 'EMOTHER']
    glass_class_list = ["EPEIBL", "EPEIPI", "EPEITP", "AXXIAP", "AXXIAL", "EAOIFE", "EAOTHER"]
    with open(json_f) as f:
        impath_class_box_dict = json.load(f)
        for path, mess in impath_class_box_dict.items():
            if not mess:  # 没有检测框
                continue
            img_name = path.split('/')[-1]

            print(mess)

            xmin = int(mess[0])
            ymin = int(mess[1])
            xmax = int(mess[2])
            ymax = int(mess[3])

            # img = imread(os.path.join(FLAGS.test_dataset_path, img_name))
            img = cv2.imread(os.path.join(FLAGS.test_dataset_path, img_name))
            try:
                img.shape
            except:
                print('fail to read xxx.jpg')

            if 1:
                n_xmin, n_ymin, n_xmax, n_ymax = process_to_sqr(img, xmin, ymin, xmax, ymax)

            bbox = img[n_ymin:n_ymax, n_xmin:n_xmax]
            confidences, max_confidences, max_conf_id = predict(persisted_sess, norm_and_batch(bbox), out_node)

            # test_info_writer.writerow([str(img_name), str(predict_label), str(xmin), str(ymin), str(xmax), str(ymax), str(max_confidences)])


            sorted_confidences = np.sort(confidences)[::-1]
            top2_predict_id = np.where(confidences == sorted_confidences[1])
            # a = str(confidences.tolist())
            # pdb.set_trace()

            if FLAGS.mask_or_glass == 'mask':
                predict_label = mask_class_list[max_conf_id]
                top2_predict_label = mask_class_list[int(top2_predict_id[0])]
            else:
                predict_label = glass_class_list[max_conf_id]
                top2_predict_label = glass_class_list[int(top2_predict_id[0])]

            # pdb.set_trace()

            test_info_writer.writerow([str(img_name), str(predict_label), str(top2_predict_label), str(xmin), str(ymin), str(xmax), str(ymax), str(max_confidences), str(sorted_confidences[1]), str(confidences.tolist())])
            # test_info_writer.writerow([str(img_name), str(predict_label), str(top2_predict_label), str(confidences)])

def defect_filter_from_json(json_f, defect_free_th):
    defect_img_list = []
    with open(json_f) as f:
        impath_class_box_dict = json.load(f)
        for path, mess in impath_class_box_dict.items():
            if not mess:  continue
            img_name = path.split('/')[-1]
            detec_conf = mess[4]

            if detec_conf > defect_free_th:
                defect_img_list.append(img_name)

    return defect_img_list


def count_cls_num(csv_file, cls_idx_in_csv):
    cls = []
    with open(csv_file) as f:
        csv_f = csv.reader(f)
        for idx, row in enumerate(csv_f):
            if idx != 0:
                cls.append(row[cls_idx_in_csv])
        cls_counter = Counter(cls)
        return cls_counter


def clsfy_img_from_csv_result(csv_file, src_img_folder, tgt_folder):
    with open(csv_file) as f:
        csv_f = csv.reader(f)
        for idx, row in enumerate(csv_f):
            if idx != 0:
                predict_label = row[1]
                img_name = row[0]
                xmin, ymin, xmax, ymax = int(row[3]), int(row[4]), int(row[5]), int(row[6])

                img_folder = os.path.join(tgt_folder, predict_label)
                if not os.path.exists(img_folder):
                    os.makedirs(img_folder)

                img = cv2.imread(os.path.join(src_img_folder, img_name))
                img_roi = img[ymin:ymax, xmin:xmax]
                cv2.imwrite(os.path.join(img_folder, img_name), img_roi)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.Session(config=config) as persisted_sess:
    print('------------------')
    with gfile.FastGFile(FLAGS.frozen_pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_sess.graph.as_default()
        tf.import_graph_def(graph_def)

        # test_info_f = open(FLAGS.output_all_info, 'w')
        # test_info_writer = csv.writer(test_info_f)

        out = persisted_sess.graph.get_tensor_by_name('import/InceptionResnetV2/Logits/Predictions:0')

        #########################################     directly read image    #########################################
        mura2Cls_list = ['FOLD', 'GRID']
        img_file = os.listdir(FLAGS.test_dataset_path)
        for img_name in img_file:
            img = cv2.imread(os.path.join(FLAGS.test_dataset_path, img_name))
            confidences, max_confidences, max_conf_id = predict(persisted_sess, norm_and_batch(img), out)
            predict_label = mura2Cls_list[max_conf_id]

            result_img_saved_dir = os.path.join(FLAGS.result_img_clsfy_folder, predict_label)
            if not os.path.exists(result_img_saved_dir): os.makedirs(result_img_saved_dir)

            cv2.imwrite(os.path.join(result_img_saved_dir, img_name), img)

        #########################################     READ JSON DIRECTLY FROM JSON     ######################################################
        # test_info_writer.writerow(['FILE', 'PRED_LABEL', 'TOP2_PRED_LABEL', 'XMIN', 'YMIN', 'XMAX', 'YMAX', 'TOP1_CONFIDENCES','TOP2_CONFIDENCES', 'ALL_CONFIDENCE'])
        # get_bbox_and_predict(FLAGS.json_file_path, out)
        # test_info_f.close()

        #########################################     TEST DEFECT-FREE WHOLE-IMG FILTER    ######################################################
        # test_info_writer.writerow(['FILE', 'PRED_LABEL', 'MAX_CONFIDENCES', 'ALL_CONFIDENCE' ])
        #
        # mask_class_list = ['EMDFBM', 'EMDFJS', 'EMDFKL', 'EMDFET', 'EMDFFD', 'EMOTHER']
        # mask_class_list_5cls = ['EMDFBM', 'EMDFJS', 'EMDFKL', 'EMDFET', 'EMOTHER', 'EMDFFD']
        # glass_class_list = ["EPEIBL", "EPEIPI", "EPEITP", "AXXIAP", "AXXIAL", "EAOIFE", "EAOTHER"]
        #
        #
        # defect_free_img_list = defect_filter_from_json(FLAGS.json_file_path, FLAGS.obj_th)
        #
        # for root, _, images_name in os.walk(FLAGS.test_dataset_path):
        #     for idx, image_name in enumerate(images_name):
        #         if image_name in defect_free_img_list:  # only defect img
        #             img = cv2.imread(os.path.join(root, image_name))
        #
        #             confidences, max_confidences, max_conf_id = predict(persisted_sess, norm_and_batch(img), out)
        #
        #             if max_conf_id == 0:
        #                 # pdb.set_trace()
        #                 pass
        #
        #             # if max_confidences < FLAGS.cls_th :  # weed out low conf
        #             #     max_conf_id = 5 if FLAGS.mask_or_glass == "mask" else 6
        #
        #         else:  # defect_free_img
        #             max_conf_id = 5 if FLAGS.mask_or_glass == "mask" else 5
        #             max_confidences = -1
        #             confidences = np.array([-1])
        #
        #         predict_label = mask_class_list_5cls[max_conf_id] if FLAGS.mask_or_glass == "mask" else glass_class_list[max_conf_id]
        #
        #         test_info_writer.writerow([str(image_name), str(predict_label), str(max_confidences), str(confidences.tolist())])
        #
        # test_info_f.close()
        # print("************************************************************************************************************************")
        # print("obj_th: %f,cls_th: %f" %(FLAGS.obj_th,FLAGS.cls_th))
        # print(count_cls_num(FLAGS.output_all_info, 1))
        # print("************************************************************************************************************************")
