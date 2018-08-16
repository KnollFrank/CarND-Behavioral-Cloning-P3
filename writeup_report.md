# **Behavioral Cloning**

## Writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* preprocess.py to preprocess the recorded camera images before feeding them to the neural network
* drive.py for driving the car in autonomous mode
* Behavioral-Cloning.ipynb, a jupyter notebook for training some model architectures
* model.h5 containing a trained convolutional neural network
* writeup_report.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model.

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with three convolutional layers (`Convolution2D`) of increasing depths having filter sizes between 3x3 and 5x5 (model.py function `create_model_Nvidia()`)

The model includes RELU layers each of them immediately following a convolutional layer to introduce nonlinearity (see `activation='relu'`), and the data is normalized in the model using a Keras lambda layer (`Lambda(lambda image: image / 255.0 - 0.5)`).

### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. This was accomplished through the statement `model.fit(X_train, y_train, validation_split=0.2, ...)` which sets 20% of the training data `X_train` apart for validation. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py statement `model.compile(... optimizer='adam')`).

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple neural network and gradually adding more layers in order to drive successfully around test track one.

#### Simple Neural Network

My first step was to use a simple neural network model (see model.py, function `create_model_simple()`) consisting of a `Flatten` layer and a `Dense` layer.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

I found that my first model had a high mean squared error on the training set and a high mean squared error on the validation set:

![LeNet model](examples/loss_model_simple.png)

This implied that the model was underfitting.

#### LeNet

To combat the underfitting, I added convolutional layers among others by switching to [LeNet-5](http://yann.lecun.com/exdb/lenet/) (see model.py, function `create_model_LeNet()`). It's best validation loss was 0.02058 within 5 epochs of training, which is much better than the simple network's validation loss:

![LeNet model](examples/loss_model_lenet.png)

#### Nvidia's Model

I tested another model from Nvidia (see model.py, function `create_model_Nvidia()`) having more convolutional layers and more fully connected layers than LeNet. It's validation loss was 0.02452, which is almost equal to LeNet's validation loss of 0.02058:

![Nvidia model](examples/loss_model_nvidia.png)

At the end of this process, the vehicle is able to drive autonomously around the track without leaving the road using Nvidias model.

### 2. Final Model Architecture

The final model architecture (model.py, function `create_model_Nvidia()`) is a convolutional neural network adapted from Nvidia. Here is a visualization of the architecture:

![Model Visualization (LeNet)](examples/model_nvidia.jpg)

The first layer receives a RGB camera image having 40 rows and 80 columns, which has a size a quarter of the original camera image in each direction due to the preprocessing steps described in the section "Resizing Images to a Quarter in Each Dimension".

The next layer is a Lambda layer normalizing the image to values between -0.5 and +0.5.

Then a cropping layer follows. It removes the top portion of the image containing trees and hills and sky, and the bottom portion of the image containing the hood of the car as described in the section "Cropping Images".

Then three convolutional layers follow having decreasing filter sizes and increasing depths. This sequence of convolutional layers recognizes low level features of the camera image like edges in it's first layer and higher level features like lane lines and their curvatures in it's higher layers.

Then after flattening the image four fully connected layers follow to finally compute a suitable steering angle from the high level features of the image.

### 3. Creation of the Training Set & Training Process

#### Training Process

To capture good driving behavior, I used the [sample driving data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided in the lecture. Here is an example image of center lane driving:

![center lane driving](examples/center_2016_12_01_13_31_15_308.jpg)

#### Resizing Images to a Quarter in Each Dimension

The training process on my local computer having no GPU was quite slow. A tip from a former udacity student I have found on the net was to resize the images. So I resized the images by adding a keras `AveragePooling2D` layer, which resulted in no performance gain. But resizing the images to a quarter in each dimension using `cv2.resize()` prior to feeding them to the neural network had the desired positive effect on training speed without worsening validation loss. Here is an example of a resized image:

![quartered image](examples/center_2016_12_01_13_31_15_308_quartered.jpg)

#### Recovery Driving

I then simulated the car recovering from the left side and right side of the road back to center so that the car would learn to steer if it drifts off to the left or the right. Following the lecture "Using Multiple Cameras" I exchanged the center camera image with the left (respectively right) camera image and adapted the steering angle in order to steer more to the right (respectively left).

These images show what a recovery looks like starting from the left using a steering angle of 0.2° clockwise:

![left camera](examples/left_2016_12_01_13_31_15_308.jpg)

or starting from the right using a steering angle of 0.2° counterclockwise:

![right camera](examples/right_2016_12_01_13_31_15_308.jpg)

#### Flipping Images And Steering Angles

As the trained car often tries to steer to the left (named 'left turn bias' in the lecture "Data Augmentation") the images and steering angles were flipped in order to present the neural network with more right turns and suitable steering angles.

For example, here is an image captured by the center camera showing a left turn:

![Image captured by the center camera](examples/center_2016_12_01_13_31_15_308.jpg)

And then the image has been flipped vertically in order to simulate a right turn:

![Flipped image from the center camera](examples/center_2016_12_01_13_31_15_308_flipped.jpg)

#### Cropping Images

As explained in the lecture "Cropping Images in Keras ", not all of the pixels in a camera image contain useful information for predicting steering angles. So the top portion of the image containing trees and hills and sky, and the bottom portion of the image containing the hood of the car are removed using a `Cropping2D` layer within the neural network. Here is an example of a cropped image:

![cropped image](examples/center_2016_12_01_13_31_15_308_cropped.jpg)

After the collection, preprocessing and augmentation process, I had 38572 images.

I finally randomly shuffled the data set and put 20% of the data into a validation set (see model.py: `model.fit(... validation_split=0.2, shuffle=True ...`).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was fixed to 5, which is the maximum number of epochs I was willing to wait for results on my local machine. In order to save only the best model params within the 5 epochs I used a `ModelCheckpoint` with `save_best_only=True`. I used an adam optimizer so that manually training the learning rate wasn't necessary.
