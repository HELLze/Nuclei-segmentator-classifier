from __future__ import division
import os
import sys
import random
import h5py
import numpy as np
from keras.models import Model, load_model, Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.layers.core import Dropout, Lambda, Layer
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
from keras import callbacks
from keras import optimizers
from keras.utils import Sequence
import matplotlib.pyplot as plt
from keras.utils import np_utils
import logging
logger = logging.getLogger()
console = logging.StreamHandler()
logger.addHandler(console)


dset = h5py.File('../nuclei_images/train_augmented_shuffled.h5', 'r')
X_train = dset['img'][:]
Y_train = dset['lab'][:]
dset.close()

dset = h5py.File('./nuclei_images/validate_shuffled.h5', 'r')
X_val = dset['img'][:]
Y_val = dset['lab'][:]
dset.close()

Y_train = np_utils.to_categorical(Y_train, num_classes = 2)
Y_val = np_utils.to_categorical(Y_val, num_classes = 2)


def batch_generator(X, Y, batch_size):
    nsamples = len(X)
    start_idx = 0
    while True:
        if start_idx + batch_size > nsamples:
            start_idx = 0
        x_batch = X[start_idx:start_idx+batch_size, ...] / 255.
        y_batch = Y[start_idx:start_idx+batch_size, ...]
        start_idx += batch_size
        yield np.asarray(x_batch), np.asarray(y_batch)

numb = random.randint(5000)        
run_name = ''.format('lympho_other_supervised_classifier_v4' + str(numb))
graphPath = './Graph/' + run_name + '/'

checkpointerPath = './model/'


if not os.path.exists('./Graph/'):
    os.mkdir('./Graph/')
    logger.warn('directory for graphs created')
if not os.path.exists(graphPath):
    os.mkdir(graphPath)
    logger.warn('graphPath created')
if not os.path.exists(checkpointerPath):
    os.mkdir(checkpointerPath)
    logger.warn('directory for models created')

model = Sequential()
model.add(Convolution2D(64, (5, 5), input_shape=X_train.shape[1:], strides=1, padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Convolution2D(128, (5, 5), strides=1, padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Convolution2D(64, (3, 3), strides=1, padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3, 3), strides=1, padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(2))
model.add(Activation('softmax'))

opt = optimizers.Adam(lr=0.0001)

logger.warn(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


checkpointer = callbacks.ModelCheckpoint(filepath=''.join([checkpointerPath + run_name + '.h5']), verbose=1, save_best_only=True, mode='min')
tbCallBack = callbacks.TensorBoard(log_dir=graphPath, histogram_freq=0, write_graph=True, write_images=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience=6, verbose=1, min_lr=0.000001)

train_b_size = 32
val_b_size = 16

training_generator = batch_generator(X_train, Y_train, train_b_size)
validation_generator = batch_generator(X_val, Y_val, val_b_size)

history = model.fit_generator(generator=training_generator,
                    epochs=70,
                    steps_per_epoch=len(X_train) // train_b_size,
                    validation_data=validation_generator,
                    validation_steps=len(X_val) // val_b_size,
                    callbacks=[earlystopper,checkpointer,tbCallBack, reduce_lr],
                    verbose=0)
