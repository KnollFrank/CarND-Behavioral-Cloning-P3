# https://classroom.udacity.com/nanodegrees/nd013/parts/edf28735-efc1-4b99-8fbb-ba9c432239c8/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/b6356fc5-5191-40ae-a2d9-3c8d2c2b37bb

import numpy as np
import pandas as pd
from scipy import ndimage

def get_driving_log():
    base_path = '../CarND-Behavioral-Cloning-P3-data_from_udacity/data/'
    def correct_path(path):
        filename = path.split('/')[-1]
        return base_path + 'IMG/' + filename

    df = pd.read_csv(base_path + 'driving_log.csv')
    # TODO: DRY
    df['center'] = df['center'].map(correct_path)
    df['left'] = df['left'].map(correct_path)
    df['right'] = df['right'].map(correct_path)
    return df


def get_images_and_measurements(size):
    df = get_driving_log()[:size]
    return df['center'].map(ndimage.imread).values, df['steering'].values


images, measurements = get_images_and_measurements(50)
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import  Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 6)
model.save('model.hd5')

