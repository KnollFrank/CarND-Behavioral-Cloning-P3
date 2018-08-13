# https://classroom.udacity.com/nanodegrees/nd013/parts/edf28735-efc1-4b99-8fbb-ba9c432239c8/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/b6356fc5-5191-40ae-a2d9-3c8d2c2b37bb
import csv

import numpy as np
from scipy import ndimage


def get_driving_log():
    def read_driving_log():
        with open('../CarND-Behavioral-Cloning-P3-data_from_udacity/data/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                yield line

    return list(read_driving_log())[1:]

images = []
measurements = []
for line in get_driving_log()[:50]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../CarND-Behavioral-Cloning-P3-data_from_udacity/data/IMG/' + filename
    image = ndimage.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

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

