import tensorflow as tf
import os
import sys
from tensorflow.python.platform import gfile
import numpy as np
from scipy.misc import imread
import glob
import pdb
import time
import csv
import json

from preprocessing import inception_preprocessing
from datasets import mydata

slim = tf.contrib.slim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("frozen_pb_file",
                    "/home-ex/tclsz/yangshun/tensorflow/models/research/train_holeCls__models/finetune_holeCls_models_from_inceptionResnetV2/frozen_inception_resnet_v2.pb","")
flags.DEFINE_string("Validation_dataset_path",
                    "/home-ex/tclsz/yangshun/datasets/holeTEST/","")
flags.DEFINE_string("Validation_label_file",
                    "/home-ex/tclsz/yangshun/chenww/Classification/Dataset/validation4350/labels.txt","")
flags.DEFINE_string("json_file_path",
                    "/home-ex/tclsz/yangshun/datasets/holeTEST/labels.json","")
flags.DEFINE_integer("batch_size", 600, "image batch size for infering")


NCHANNEL = 3
WIDTH = 299
HEIGHT = 299


def get_Image_Batch_From_Tfrecord_1(filenames, batch_size=FLAGS.batch_size, shuffle=True,enqueue_many=False):
    filenameQ = tf.train.string_input_producer(filenames, num_epochs=None)
    recordReader = tf.TFRecordReader()
    key, fullExample = recordReader.read(filenameQ)
    features = tf.parse_single_example(
        fullExample,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            #'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })
    label = features['image/class/label']
    image_buffer = features['image/encoded']
    with tf.name_scope('decode_png', [image_buffer], None):
        image = tf.image.decode_png(image_buffer, channels=NCHANNEL)
        #image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    pdb.set_trace()
    image = inception_preprocessing.preprocess_image(image, 299, 299, is_training=False)
    # label = tf.stack(tf.one_hot(label - 1, NCLASS))

    if shuffle:
        imageBatch, labelBatch = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size,
            capacity=2 * batch_size,
            min_after_dequeue=1, enqueue_many=enqueue_many)
    else:
        imageBatch, labelBatch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=1,
            capacity=2 * batch_size)
    return imageBatch, labelBatch


def get_Image_Batch_From_Tfrecord_2(dataset,batch_size=FLAGS.batch_size, shuffle=True, enqueue_many=False):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])

    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, 299, 299, is_training=False)

    if shuffle:
        imageBatch, labelBatch = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size,
            capacity=2 * batch_size,
            min_after_dequeue=1, enqueue_many=enqueue_many)
    else:
        imageBatch, labelBatch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=1,
            capacity=2 * batch_size)
    return imageBatch, labelBatch


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


def main(_):
    start = time.time()

    json_labels = JsonDictToList(FLAGS.json_file_path)
    class_num = len(json_labels)
    mat = np.zeros((class_num, class_num), dtype='int32')
    mat_conf = [[[] for i in range(class_num)] for j in range(class_num)]

    err_f = open('./logs/wrong_prediction_imgs.csv', 'w', newline='')
    err_writer = csv.writer(err_f)
    err_writer.writerow(['LABEL', 'PREDICTION', 'CONFIDENCES'])

    csv_f = open('./logs/prediction_HOLE_cls.csv', 'w', newline='')
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(['LABEL', 'PREDICTION', 'CONFIDENCES'])

    with gfile.FastGFile(FLAGS.frozen_pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        with tf.Session() as sess:
            sess.graph.as_default()
            tf.import_graph_def(graph_def)
            tf.global_variables_initializer().run()

            file_list = list(glob.glob(FLAGS.Validation_dataset_path + "holeCls_validation*"))
            batch_image_tensor, batch_label_tensor = get_Image_Batch_From_Tfrecord_1(file_list)

            # dataset = mydata.get_split('validation', FLAGS.Validation_dataset_path)
            # batch_image_tensor, batch_label_tensor = get_Image_Batch_From_Tfrecord_2(dataset)

            coord = tf.train.Coordinator() # 协调器
            threads = tf.train.start_queue_runners(sess=sess, coord=coord) # 入队线程启动器
            np_image, np_labels = sess.run([batch_image_tensor, batch_label_tensor])

            # softmax_tensor = sess.graph.get_tensor_by_name('import/InceptionResnetV2/Logits/Predictions:0')
            # predictions = sess.run(softmax_tensor, {'import/input:0': np_image})
            # predictions = np.squeeze(predictions)

            logit_tensor = sess.graph.get_tensor_by_name("import/InceptionResnetV2/Logits/Logits/BiasAdd:0")
            logits = sess.run(logit_tensor, {'import/input:0': np_image})
            probabilities = tf.nn.softmax(logits)
            np_probabilities = sess.run(probabilities)
            predicted_label = np.argmax(np_probabilities, axis=1)

            coord.request_stop() # 终止所有线程
            coord.join(threads) # 把线程加入主线程，等待threads结束

            for index in range(FLAGS.batch_size):
                mat[np_labels[index], predicted_label[index]] += 1
                csv_writer.writerow([str(index), str(np_labels[index]), str(predicted_label[index]), str(np_probabilities[index])])

                if np_labels[index] != predicted_label[index]:
                    # print(index, '\t Actual:', np_labels[index], '\tPrediction:', predicted_label[index],'\tConfidences: ', np_probabilities)
                    err_writer.writerow([str(index), str(np_labels[index]), str(predicted_label[index]), str(np_probabilities[index])])

    err_f.close()
    csv_f.close()

    end = time.time()
    elapsed_process_time = end - start

    ShowDataStatistics(mat, json_labels, elapsed_process_time)



if __name__ == '__main__':
    tf.app.run()