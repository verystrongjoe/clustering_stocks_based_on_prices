from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
import numpy as np
from keras.datasets import mnist
from argslist import *

model = Sequential()

pad3 = 'same'
input_shape = (240, 1, 1)
KERNEL_SIZE = [5, 3]
filters=[32, 64, 128, 10]

if len(input_shape) == 2:
    input_shape = tuple(list(input_shape) + [1])

model.add(Conv2D(filters[0], KERNEL_SIZE[0], strides=2, padding='same', activation='relu', name='conv1',
                 input_shape=input_shape))
model.add(Conv2D(filters[1], KERNEL_SIZE[0], strides=2, padding='same', activation='relu', name='conv2'))
model.add(
    Conv2D(filters[2], KERNEL_SIZE[1], strides=2, padding=pad3, activation='relu', name='conv3'))  # todo : check pad3

model.add(Flatten())
model.add(Dense(units=filters[3], name='embedding'))
model.add(Dense(units=filters[2] * 30, activation='relu'))  # 128*3*3  # todo : why does it divide 8?

model.add(Reshape((-1, 1, filters[2])))

model.add(Conv2DTranspose(filters[1], KERNEL_SIZE[1], strides=(2, 1), padding=pad3, activation='relu', name='deconv3'))
model.add(Conv2DTranspose(filters[0], KERNEL_SIZE[0], strides=(2, 1), padding='same', activation='relu', name='deconv2'))
model.add(Conv2DTranspose(input_shape[2], KERNEL_SIZE[0], strides=(2, 1), padding='same', name='deconv1'))
model.summary()