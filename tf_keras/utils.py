"""
Utility functions
"""


import os
import shutil
import random
import time
from PIL import ImageDraw
from PIL import ImageFont
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def data_split(data_path, new_main_path, test_ratio=0.1, val_ratio=0.1, seed=15):
    """
    Split data to training set and test set
    :param data_path: A path
    :param seed: A random seed
    :param test_ratio: A float value in [0.0, 1.0]
    :param validation: True if setting validation set
    :param val_ratio: A float value in [0.0, 1.0]
    :return: Three list with filenames in training, test, validation set
    """
    random.seed(seed)
    for classname in os.listdir(data_path):
        img_name_list = os.listdir(os.path.join(data_path, classname))
        random.shuffle(img_name_list)
        total_num = len(img_name_list)
        test_num = int(total_num * test_ratio)
        test_list = img_name_list[:test_num]
        val_num = int(total_num * val_ratio)
        val_list = img_name_list[test_num:test_num + val_num]
        train_list = img_name_list[val_num + test_num:]
        cp_filename(data_path, os.path.join(new_main_path, 'train'), classname, train_list)
        cp_filename(data_path, os.path.join(new_main_path, 'val'), classname, val_list)
        cp_filename(data_path, os.path.join(new_main_path, 'test'), classname, test_list)


def cp_filename(main_path, new_path, classname, file_list):
    create_directory(os.path.join(new_path, classname))
    for filename in file_list:
        srcfile = os.path.join(main_path, classname, filename)
        dstfile = os.path.join(new_path, classname, filename)
        shutil.copyfile(srcfile, dstfile)


def train_data_generator(train_directory, batch_size, preprocess_function=None, target_size=(224, 224)):
    train_datagen = ImageDataGenerator(rotation_range=0.,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       zoom_range=0.1,
                                       horizontal_filp=True,
                                       preprocess_function=preprocess_function,
                                       )
    train_generator = train_datagen.flow_from_directory(directory=train_directory,
                                                        target_size=target_size,
                                                        classes=None,
                                                        class_mode="categorical",
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        )
    return train_generator


def test_data_generator(test_directory, preprocess_function=None, target_size=(224, 224)):
    test_datagen = ImageDataGenerator(preprocess_function=preprocess_function)
    test_generator = test_datagen.flow_from_directory(directory=test_directory,
                                                      target_size=target_size,
                                                      classes=None,
                                                      class_mode="categorical",
                                                      )
    return test_generator


def show_result_pic(image, preds):
    # Ubuntu font
    # font = ImageFont.truetype('LiberationSans-Regular.ttf', 30)
    # Windows font
    font = ImageFont.truetype("C:\Windows\Fonts\Verdana.ttf", 30)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), '%s: %f' % (preds[0][1], preds[0][2]), (255, 0, 0), font=font)
    image.show()


def create_optimizer(opt, learning_rate, momentum=0.9, decay=0.0, nesterov=False):
    """
    Create optimizer operation
    :param opt: A string which can be one of 'sgd', 'momentum' or 'adam'
    :param learning_rate: A float value
    :param momentum: A float value
    :return: An optimizer operation
    """
    assert opt in ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']:
    if opt == 'sgd':
        optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=nesterov)
    elif opt == 'rmsprop':
        optimizer = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-06)
    elif opt == 'adagrad':
        optimizer = optimizers.Adagrad(lr=learning_rate, epsilon=1e-06)
    elif opt == 'adadelta':
        optimizer = optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-06)
    elif opt == 'adam':
        optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif opt == 'adamax':
        optimizer = optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif opt == 'nadam':
        optimizer = optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    else:
        optimizer = None
    return optimizer


def print_time():
    return time.strftime('%y-%m-%d %H:%M:%S', time.localtime(time.time()))


def record_time():
    return time.strftime('%y%m%d%H%M%S', time.localtime(time.time()))


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        pass


def early_stop(train_accuracies, train_losses, val_accuracies=None, val_losses=None, n=3):
    if (len(train_accuracies) >= 3) \
            and (train_accuracies[-n:] == [1.0]*n) \
            and (train_losses[-n:] == [0.0]*n):
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
