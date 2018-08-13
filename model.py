# https://classroom.udacity.com/nanodegrees/nd013/parts/edf28735-efc1-4b99-8fbb-ba9c432239c8/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/b6356fc5-5191-40ae-a2d9-3c8d2c2b37bb

from preprocess import create_Cropping2D, get_input_shape, get_images_and_measurements, get_data


def get_X_train_y_train():
    print('get_images_and_measurements ...')
    images_measurements = get_images_and_measurements(8036)
    print('... get_images_and_measurements')
    X_train = get_data(images_measurements, 'image')
    y_train = get_data(images_measurements, 'measurement')
    return X_train, y_train


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D
from keras.layers.convolutional import Convolution2D


def create_model():
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


X_train, y_train = get_X_train_y_train()
model = create_model()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
model.save('model.hd5')

# TODO: Data Augmentation (https://classroom.udacity.com/nanodegrees/nd013/parts/edf28735-efc1-4b99-8fbb-ba9c432239c8/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/580c6a1d-9d20-4d2e-a77d-755e0ca0d4cd)
# TODO: videos following Data Augmentation
