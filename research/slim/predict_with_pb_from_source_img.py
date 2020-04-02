# ************************************************  Keypoint ***************************************************
# data(np.ndarray) -> processed_image(tensor) -> np_image(np.ndarray)
# Doing this convertion to reshape source img to 299*299 by using "inception_preprocessing.preprocess_image()"
# use imresize() will result in wrong prediction
# **************************************************************************************************************
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "frozen_pb_file",
    "/home-ex/tclsz/yangshun/tensorflow/models/research/train_holeCls__models/finetune_holeCls_models_from_inceptionResnetV2/frozen_inception_resnet_v2.pb","")
flags.DEFINE_string(
    "Validation_dataset_path",
    "/home-ex/tclsz/yangshun/datasets/holeTEST/source_img","")
flags.DEFINE_string(
    "Validation_label_file",
    "/home-ex/tclsz/yangshun/datasets/holeTEST/labels.txt","")
flags.DEFINE_string(
    "json_file_path",
    "/home-ex/tclsz/yangshun/datasets/holeTEST/labels.json","")
flags.DEFINE_integer(
    "batch_size",
    900,
    "image batch size for infering")

class ImageFolderDataSource:
    def __init__(self, folder, batch_size, labels):
        if not os.path.exists(folder):
            raise Exception("Folder doesn't exist: {}".format(folder))
        if set(labels) != set(os.listdir(folder)):
            raise Exception("The labels are not consistent with folder structure.")
        self.labels = labels
        self.n_classes = len(self.labels)
        self.labels_index = np.arange(self.n_classes)
        self.index_map = dict(zip(self.labels, self.labels_index))
        self.label_map = dict(zip(self.labels_index, self.labels))
        self.folder = folder
        self.batch_size = batch_size
        self.label_pool = []
        self.file_pool = []
        for label in self.labels:
            label_one_hot = self.one_hot(self.index_map[label])
            label_folder = os.path.join(folder, label)
            label_files = list(map(
                lambda f: os.path.join(label_folder, f),
                os.listdir(label_folder)
            ))
            self.file_pool.extend(label_files)
            self.label_pool.extend(np.repeat([label_one_hot], len(label_files), axis=0))
        self.n_files = len(self.file_pool)
        self.label_pool = np.array(self.label_pool)
        self.file_pool = np.array(self.file_pool)
    def one_hot(self, index):
        res = np.repeat(0, self.n_classes).astype(np.float32)
        res[index] = 1.0
        return  res
    def rand_index(self):
        return np.random.choice(np.arange(self.n_files), self.batch_size)
    def sequential_index(self):
        return np.arange(self.batch_size)
    def get_batch(self):
        index = self.rand_index()
        # index = self.sequential_index()
        batch_labels = self.label_pool[index]
        batch_files = self.file_pool[index]
        batch_data = np.array(list(
            map(
                lambda f: imread(f),
                batch_files
            )
        ))
        return batch_labels, batch_data, batch_files

def JsonDictToList(json_file_path):
    with open(json_file_path) as json_file:
        classes = json.load(json_file)
    json_labels = [None] * len(classes)
    for key, value in classes.items():
        json_labels[value] = key
    return json_labels


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


with tf.Session() as persisted_sess:
    json_labels = JsonDictToList(FLAGS.json_file_path)
    class_num = len(json_labels)
    mat = np.zeros((class_num, class_num), dtype='int32')
    mat_conf = [[[] for i in range(class_num)] for j in range(class_num)]

    err_f = open('./logs/wrong_prediction_imgs.csv', 'w', newline='')
    err_writer = csv.writer(err_f)
    err_writer.writerow(['INDEX', 'GT', 'PREDICTION', 'CONFIDENCES'])

    csv_f = open('./logs/prediction_HOLE_cls.csv', 'w', newline='')
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(['INDEX', 'GT', 'PREDICTION', 'CONFIDENCES'])

    def run_inference_on_image():
        with open(FLAGS.Validation_label_file) as fh:
            label_names = [x.strip() for x in fh.readlines()]
        batch_reader = ImageFolderDataSource(FLAGS.Validation_dataset_path, FLAGS.batch_size, label_names) #validate
        onehot_label, data, files = batch_reader.get_batch()

        label_list = np.array(np.argmax(onehot_label, axis=1))
        data = np.divide(data, np.float32(255.0))
        # answer = None

        # # Print all operators in the graph
        # for op in persisted_sess.graph.get_operations():
            # print(op)
        # # Print all tensors produced by each operator in the graph
        # for op in persisted_sess.graph.get_operations():
            # print(op.values())
        # tensor_names = [[v.name for v in op.values()] for op in persisted_sess.graph.get_operations()]
        # tensor_names = np.squeeze(tensor_names)
        # print(tensor_names)

        out = persisted_sess.graph.get_tensor_by_name('import/InceptionResnetV2/Logits/Predictions:0')

        def predict(img):
            probabilities = persisted_sess.run(out, {'import/input:0': img})
            probabilities = np.squeeze(probabilities) # predictions = (batch_size, 4)
            top_k = probabilities.argsort()[:][::-1]  # Getting top 3 predictions, reverse order
            # for node_id in top_k:
                # human_string = batch_reader.labels[node_id]
                # score = predictions[node_id]
                #print('%s (conf = %.5f)' % (human_string, score))
            # answer = batch_reader.labels[top_k[0]]
            return probabilities

        start_time = time.time()

        print('\n=======================cropping images==============================')
        img_tensor = [None] * len(label_list)
        for index in tqdm(range(len(label_list))):
            single_img = data[index] # type numpy.ndarray
            single_img_tensor = tf.convert_to_tensor(single_img, tf.float32)  # ndarray to tensor
            processed_single_img_tensor = inception_preprocessing.preprocess_image(single_img_tensor, 299, 299, is_training=False)
            processed_single_img_tensor = tf.expand_dims(processed_single_img_tensor, 0)
            img_tensor[index] = processed_single_img_tensor
        print('\n DONE.')

        print('\n================concatenating images in batch=======================')
        img_tensor = tf.concat(img_tensor, 0)
        img_np = img_tensor.eval()
        print('\n DONE.')

        print('\n======================predicting in batch===========================')
        confidences = predict(img_np)

        predicted_label = np.argmax(confidences, axis=1)

        for index in range(FLAGS.batch_size):
            mat[label_list[index], predicted_label[index]] += 1
            csv_writer.writerow([str(index), str(label_list[index]), str(predicted_label[index]), str(confidences[index])])

            if label_list[index] != predicted_label[index]:
                # print(index, '\t Actual:', np_labels[index], '\tPrediction:', predicted_label[index], '\tConfidences: ', np_probabilities)
                err_writer.writerow([str(index), str(label_list[index]), str(predicted_label[index]), str(confidences[index])])

        err_f.close()
        csv_f.close()

        elapsed_process_time = time.time() - start_time

        ShowDataStatistics(mat, json_labels, elapsed_process_time)



    with gfile.FastGFile(FLAGS.frozen_pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_sess.graph.as_default()
        tf.import_graph_def(graph_def)
        run_inference_on_image()
