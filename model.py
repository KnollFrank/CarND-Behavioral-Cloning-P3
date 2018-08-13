# https://classroom.udacity.com/nanodegrees/nd013/parts/edf28735-efc1-4b99-8fbb-ba9c432239c8/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/b6356fc5-5191-40ae-a2d9-3c8d2c2b37bb

import numpy as np
import pandas as pd
from scipy import ndimage


def get_driving_log():
    base_path = '../CarND-Behavioral-Cloning-P3-data_from_udacity/data/'

    def correct_path(path):
        filename = path.split('/')[-1]
        return base_path + 'IMG/' + filename

    def correct_path_in_column(df, column):
        df[column] = df[column].map(correct_path)

    df = pd.read_csv(base_path + 'driving_log.csv')
    correct_path_in_column(df, 'center')
    correct_path_in_column(df, 'left')
    correct_path_in_column(df, 'right')
    return df


def get_images_and_measurements(size):
    df = get_driving_log()[:size]
    return pd.DataFrame(
        {'image': df['center'].map(ndimage.imread).values,
         'measurement': df['steering'].values})


def get_data(df, column):
    return np.array(df[column].values.tolist())


images_measurements = get_images_and_measurements(8036)
X_train = get_data(images_measurements, 'image')
y_train = get_data(images_measurements, 'measurement')

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
model.save('model.hd5')

# TODO: Data Augmentation (https://classroom.udacity.com/nanodegrees/nd013/parts/edf28735-efc1-4b99-8fbb-ba9c432239c8/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/580c6a1d-9d20-4d2e-a77d-755e0ca0d4cd)
# TODO: videos following Data Augmentation
