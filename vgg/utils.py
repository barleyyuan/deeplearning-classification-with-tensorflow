"""
Utility functions
"""


import os, random, time
import tensorflow as tf


def train_test_split(data_path, seed=15, test_ratio=0.1):
    """
    Split data to training set and test set
    :param data_path: A path
    :param seed: A random seed
    :param test_ratio: A float value in [0.0, 1.0]
    :return: A list with filenames in training set, a list with filenames in test set
    """
    random.seed(seed)
    train = []
    test = []
    for classname in os.listdir(data_path):
        img_name_list = os.listdir(os.path.join(data_path, classname))
        random.shuffle(img_name_list)
        train_list = img_name_list[:int(len(img_name_list)*(1-test_ratio))]
        test_list = img_name_list[int(len(img_name_list)*(1-test_ratio)):]
        train.append(train_list)
        test.append(test_list)
    return train, test


def cal_num(dataset):
    n = 0
    for sublist in dataset:
        n += len(sublist)
    return n


def get_filename_label(main_path, file_list):
    """
    Get filenames and labels of data
    :param main_path: A path
    :param file_list: A list with filenames
    :return: A tf.constant of filenames, a tf.constant of labels
    """
    filenames = []
    labels = []
    for label, classname in enumerate(os.listdir(main_path)):
        sub_list = file_list[label]
        for filename in sub_list:
            filenames.append(os.path.join(main_path, classname, filename))
            labels.append(label)
    return tf.constant(filenames), tf.constant(labels)


def _parse_function(filename, label, num_classes):
    '''
    Read an image from a file, decode it into a dense tensor, and preprocess it.
    :param filename: A tf.constant of filename
    :param label: A tf.constant of label
    :param num_class: An int number of class
    :return: A tensor of image, a tensor of label
    '''
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_data = tf.image.resize_images(image_decoded, [224, 224])
    image_data = tf.image.per_image_standardization(image_data)
    label = tf.one_hot(label, num_classes)
    return image_data, label


def create_optimizer(opt, learning_rate, momentum):
    """
    Create optimizer operation
    :param opt: A string which can be one of 'sgd', 'momentum' or 'adam'
    :param learning_rate: A float value
    :param momentum: A float value
    :return: An optimizer operation
    """
    assert (opt == 'sgd') or (opt == 'momentum') or (opt == 'adam')
    if opt == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='SGD')
    elif opt == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, name='Momentum')
    elif opt == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate, name='Adam')
    else:
        optimizer = None
    return optimizer


def train(logits, preds, labels, optimizer, learning_rate, momentum):
    """
    Training process
    :param logits: A tensor
    :param preds: A tensor
    :param labels: A tensor
    :param optimizer: A operation
    :param learning_rate: A float value
    :param momentum: A float value
    :return: A operation, a operation, a operation, a operation
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels, name='cross_entropy')
    loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    optimizer = create_optimizer(optimizer, learning_rate, momentum)
    train_op = optimizer.minimize(loss)
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    correct_count = tf.reduce_sum(tf.cast(correct_prediction, 'float'))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    return train_op, correct_count, loss, accuracy


def print_time():
    return time.strftime('%y-%m-%d %H:%M:%S', time.localtime(time.time()))


def record_time():
    return time.strftime('%y%m%d%H%M%S', time.localtime(time.time()))


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        pass


def early_stop(train_accuracy, train_loss, val_accuracies=None, val_losses=None, n=3):
    if (train_accuracy == 1.0) and (train_loss == 0.0):
        return True
    else:
        if (val_accuracies is not None) \
                and (len(val_accuracies) >= 3) \
                and(val_accuracies[-n:] == sorted(val_accuracies[-n:], reverse=True)) \
                and (val_losses[-n:] == sorted(val_losses[-n:])):
            assert len(val_accuracies) == len(val_losses)
            return True
        else:
            return False
