import utils
import os
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Hyper-parameters
num_classes = 3
batch_size = 32
epochs = 500
validation_steps = 10
# training parameters
opt = 'sgd'
learning_rate = 1e-04
momentum = 0.0 # necessary if opt is 'sgd'
# model path
model_path = "/home/deeplearning/Projects/Muck/models/"
utils.create_directory(model_path)
model_filepath = os.path.join(model_path, "resnet50_keras_model.h5")

# data_path
data_path = "/home/deeplearning/Projects/Muck/data/keras_data/"
train_directory = os.path.join(data_path, 'train')
validation_directory = os.path.join(data_path, 'validation')
test_directory = os.path.join(data_path, 'test')


# create data train generator
train_generator = utils.train_data_generator(train_directory, batch_size, preprocess_input, target_size=(224, 224))
train_image_numbers = train_generator.samples
# create validation generator
validation_generator = utils.test_data_generator(validation_directory, preprocess_input, (224, 224))
# create test generator
test_generator = utils.test_data_generator(test_directory, preprocess_input, (224, 224))

# create the base pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False, pooling="avg")
# add a logistic layer for predictions
x = base_model.output
# x = Flatten()(x)
predictions = Dense(num_classes, activation='softmax')(x)

# this the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# if train only the top layers, uncomment the following two lines
# for layer in base_model.layers:
#    layer.trainable = False

# configure the optimizer
optimizer = utils.create_optimizer(opt, learning_rate, momentum)

# compile the model (should be done after setting layers to non-trainable)
model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=['accuracy'],
              )

# train the model
# define callback function
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=10,
                               )
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              )

model.fit_generator(train_generator,
                    steps_per_epoch=train_image_numbers//batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    callbacks=[early_stopping, reduce_lr],
                    )

# test the model
test_metrics = model.evaluate_generator(test_generator,
                         steps=1,
                         )
for i in zip(model.metrics_names, test_metrics):
    print(i)

# save the model
model.save(model_filepath)