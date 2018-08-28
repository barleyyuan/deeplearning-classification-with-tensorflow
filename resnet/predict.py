from tf_keras import utils
import time
import numpy as np
from PIL import Image
import tensorflow as tf


def predict_img(sess, classes):
    while True:
        img = input('Input image filename: ')
        t1 = time.time()
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
        else:
            image_data = utils.process_image(image)
            pred = sess.run(tf.get_default_graph().get_tensor_by_name('ResNet/prob:0'), feed_dict={'x:0': image_data})
            t2 = time.time()
            print('%s sec' % (t2-t1))
            pred_class_index = np.argmax(pred, axis=1)[0]
            pred_prob = pred[0, pred_class_index]
            pred_class = classes[pred_class_index]
            utils.show_result_pic(image, pred_class, pred_prob)
            print(pred_class, pred_prob)


def main():
    model_path = 'path/to/model/'
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
    classes = ['clean', 'dirty']
    with tf.Session() as sess:
        t1 = time.time()
        saver.restore(sess, ckpt.model_checkpoint_path)
        t2 = time.time()
        print('restore graph: %s sec' % (t2 - t1))
        predict_img(sess, classes)


if __name__ == '__main__':
    main()
