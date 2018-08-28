from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import time
from PIL import Image
from tf_keras import utils


def predict(model):
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
            t2 = time.time()
            preds = decode_predictions(preds, top=3)[0]
            utils.show_result_pic(image_, preds)
            print("Predicted: ", preds)
            print("%s sec" % (t2-t1))


def main():
    t1 = time.time()
    model = ResNet50(include_top=True, weights='imagenet')
    t2 = time.time()
    print("Load model: %s sec" % (t2-t1))
    predict(model)


if __name__ == '__main__':
    main()