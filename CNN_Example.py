#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:30:39 2019

@author: kyle
Office Hour Content - Deep Dive into CNN
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_labels = len(np.unique(y_train))
print(f"Num lables: {num_labels}")

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train.shape
# there are 60,000 observations in 28X28 2D tensors (vector). 
y_train.shape

# resize and normalize
# we are going from dimensions with height and width to height, width, and channels
# we could include more channels if we were using color images (RGB) for 3 channels
# However, since we only have grayscale 1 channel is fine.
# 28 is the image size
image_size = X_train.shape[1]
X_train = np.reshape(X_train, [-1, image_size, image_size, 1])
X_test = np.reshape(X_test, [-1, image_size, image_size, 1])
X_train.shape

# Now we want to normalize the data to reduce the noise and potential for
# overfitting. The intuition for why we sacle is to reduce potential for overfitting
# with a smaller range of values
# Recall that gray scale values range from 0 to 255
# basically going from 0 to 1 indiciating percent of gray scale
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

input_shape = (image_size, image_size, 1)
batch_size = 128 # recall that batch size should be between40 and 256
# the batches are used in mini batching to update the error during an
# epoch instead of waiting until the end of the train epoch when we have eval
# all of the images

kernel_size = 3
# Basically the kernel is a rectangular window that slides through the image
# from top, pixel by pixel. Can do it as a rectangle as well (in our case
# its a 3x3 box). Starts with top 3 rows and goes pixel by pixel, then drops 
# only 1 row. This sliding of the kernel is known as a convolution. 
# Convolutions transform the image into feature maps. 
# After a convolution the images become smaller. 
# you can pad the images by zeroing out the boarders to keep the image the 
# same size if you wish.
# Earlier convoltions find simple things, like edges and lines, later 
# convultions find more advanced features of the image.

pool_size = 2
# We are adding a MaxPooling2D layer to our model as well. MaxPooling2D compresses
# each feature map. Every patch of the size pool_size**2 is reduce by one pixel. The
# value is equal to the maximum pixel value within the map. The main reason
# to use this is to reduce the feature map size which gives us increased kernel coverage.
# So if we had a pool size 2 and a 26x26 image the pooling would reduce it to 13x13

filters = 64 # the numbers of feature maps created per Conv2D is controlled by this variable
# before we go to pooling we do this operation 64 times (kernal step)

dropout = 0.2 # is a form of regularization, that makes nnets more robust to new unseen data.
# not used in the output layer and is only used during training.
# dropout is not present when making predictions with test data.


# model is a stack of CNN-ReLU-MaxPooling
model = Sequential() # first call the keras sequential API
model.add(Conv2D(filters=filters,
          kernel_size=kernel_size,
          activation='relu',
          input_shape=input_shape))

# for the first Conv2D we specify the input shape of the images, the other arguments will stay
# the same in other Conv2D layers. 
# Will go from 28x28 to 26x26 after conv. Then 13 by 13 after pooling

model.add(MaxPooling2D(pool_size))

model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))

# 13x13 to 11x11 after conv. Then we pool to 5x5

model.add(MaxPooling2D(pool_size))

model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))

# 5x5 to 3x3 after conv. No pooling.
# Will have 65 3x3 feature maps that have been vectorized via flatten
# of length 576 (3*3*64 = 576), Now that the feature maps are flattened
# we can use drop out

# (1 - 0.20) * 576 = 461 hidden neurons/units
model.add(Flatten()) 
model.add(Dropout(dropout))

#output layer is a 10 dim one-hot vector
model.add(Dense(num_labels))
# Then we map the output length for the number of labels, 10
# this is the output for one-hot vector

model.add(Activation('softmax'))
# softmax squashes the outputs to predicted probabilities of
# each class that sum to 1. The higest probability wins.
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3, batch_size=batch_size)

loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f"The accuracy was {acc * 100}% and the loss was: {loss}")
# The accuracy was 98.89% and the loss was: 0.03154198095765896