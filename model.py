# https://classroom.udacity.com/nanodegrees/nd013/parts/edf28735-efc1-4b99-8fbb-ba9c432239c8/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/b6356fc5-5191-40ae-a2d9-3c8d2c2b37bb
import numpy as np
from keras.callbacks import ModelCheckpoint

from preprocess import create_Cropping2D, get_input_shape, get_images_and_measurements, \
    get_augmented_images_and_measurements


def get_X_train_y_train():
    images, measurements = get_augmented_images_and_measurements(*get_images_and_measurements())
    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D
from keras.layers.convolutional import Convolution2D


# val_loss: 0.02299
def create_model_LeNet():
    model = Sequential()
    model.add(Lambda(lambda image: image / 255.0 - 0.5, input_shape=get_input_shape()))
    model.add(create_Cropping2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model


# val_loss: 0.02381
def create_model_Nvidia():
    model = Sequential()
    model.add(Lambda(lambda image: image / 255.0 - 0.5, input_shape=get_input_shape()))
    model.add(create_Cropping2D())
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    # model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    # model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


X_train, y_train = get_X_train_y_train()
model = create_model_LeNet()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train,
          y_train,
          validation_split=0.2,
          shuffle=True,
          callbacks=[ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)],
          epochs=5,
          verbose=1)
