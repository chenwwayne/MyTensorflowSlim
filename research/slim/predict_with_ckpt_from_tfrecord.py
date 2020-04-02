from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import json
import numpy as np
import time
import pdb
import csv
import glob

from nets import nets_factory
from datasets import mydata
from preprocessing import inception_preprocessing

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'} for shielding useless output information
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'checkpoint_path',
    '/home-ex/tclsz/yangshun/chenww/models-master/research/slim/train_log/train_flowers',
    'The directory where the model was written to or an absolute path to a ''checkpoint file.')
tf.app.flags.DEFINE_string(
    'Validation_dataset_path',
    '/home-ex/tclsz/yangshun/chenww/Dataset/flowers/',
    'The directory where the dataset label file and info file are stored.')
tf.app.flags.DEFINE_string(
    'model_name',
    'inception_resnet_v2', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string(
    "json_file_path",
    "/home-ex/tclsz/yangshun/chenww/Dataset/flowers/labels.json","")
tf.app.flags.DEFINE_integer(
    'predict_image_size',
    299,
    'Predict image size')
tf.app.flags.DEFINE_integer(
    'batch_size',
    600,
    'batch size for inference')
FLAGS = tf.app.flags.FLAGS


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
    with tf.name_scope('decode_jpg', [image_buffer], None):
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = inception_preprocessing.preprocess_image(image, 299, 299, is_training=False)
    #label = tf.stack(tf.one_hot(label - 1, NCLASS))

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

    err_f = open('./train_log/train_flowers/wrong_prediction_imgs.csv', 'w',newline='')
    err_writer = csv.writer(err_f)
    err_writer.writerow(['LABEL', 'PREDICTION', 'CONFIDENCES'])

    csv_f = open('./train_log/train_flowers/prediction_HOLE_cls.csv', 'w', newline='')
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(['LABEL', 'PREDICTION', 'CONFIDENCES'])

    with tf.Graph().as_default():
    #   tf.logging.set_verbosity(tf.logging.INFO)

        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=class_num,
            is_training=False)
        image_size = FLAGS.predict_image_size or network_fn.default_image_size

        dataset = mydata.get_split('validation', FLAGS.Validation_dataset_path)
        batch_image_tensor, batch_label_tensor = get_Image_Batch_From_Tfrecord_2(dataset)

        # file_list = list(glob.glob(FLAGS.Validation_dataset_path + "flowers_validation*"))
        # batch_image_tensor, batch_label_tensor = get_Image_Batch_From_Tfrecord_1(file_list)

        logits, _ = network_fn(batch_image_tensor) # logits = Tensor("InceptionResnetV2/Logits/Logits/BiasAdd:0", shape=(64, 4), dtype=float32)
        probabilities = tf.nn.softmax(logits) # Tensor("Softmax:0", shape=(64, 4), dtype=float32)
        init_fn = slim.assign_from_checkpoint_fn(tf.train.latest_checkpoint(FLAGS.checkpoint_path), slim.get_variables_to_restore())

        with tf.Session() as sess:
            with slim.queues.QueueRunners(sess):
                init_fn(sess)
                np_probabilities, np_labels = sess.run([probabilities, batch_label_tensor])
                predicted_label = np.argmax(np_probabilities,axis=1)

        for index in range(FLAGS.batch_size):
            mat[np_labels[index], predicted_label[index]] += 1
            csv_writer.writerow([str(index), str(np_labels[index]), str(predicted_label[index]), str(np_probabilities[index])])

            if np_labels[index] != predicted_label[index]:
                # print(index, '\t Actual:', np_labels[index], '\tPrediction:', predicted_label[index], '\tConfidences: ', np_probabilities)
                err_writer.writerow([str(index), str(np_labels[index]), str(predicted_label[index]), str(np_probabilities[index])])

    err_f.close()
    csv_f.close()

    end = time.time()
    elapsed_process_time = end - start

    ShowDataStatistics(mat, json_labels, elapsed_process_time)

if __name__ == '__main__':
    tf.app.run()





# ****************** TESTING A PICTURE FROM .PNG  ************************
# image_string = tf.gfile.FastGFile(FLAGS.predict_file, 'rb').read()
# image = tf.image.decode_png(image_string, channels=3)
# processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
# processed_images  = tf.expand_dims(processed_image, 0)
# ****************** TESTING A PICTURE FROM .PNG  ************************