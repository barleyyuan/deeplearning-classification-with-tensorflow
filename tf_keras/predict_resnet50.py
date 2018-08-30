from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import time
from PIL import Image
import utils


def predict(model, class_list):
    while True:
        img_path = input('Input image filename: ')
        try:
            image_ = Image.open(img_path)
        except:
            print("Open Error! Try again!")
        else:
            t1 = time.time()
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = model.predict(x)
            preds = utils.decode_prediction(preds, class_list)
            t2 = time.time()
            utils.show_result_pic(image_, preds)
            print("Predicted: ", preds)
            print("%s sec" % (t2-t1))


def main():
    filepath = "/home/deeplearning/Projects/Muck/models/resnet50_keras_model.h5"
    class_list = ["non_truck", "truck_with_no_slag", "truck_with_slag"]
    t1 = time.time()
    model = load_model(filepath)
    t2 = time.time()
    print("Load model: %s sec" % (t2-t1))
    predict(model, class_list)


if __name__ == '__main__':
    main()