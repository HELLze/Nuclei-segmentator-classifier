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

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3]))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1]*X_val.shape[2]*X_val.shape[3]))
print(X_train.shape)
print(X_val.shape)

nn_grid = [16, 32]
act_grid = ['relu']
output_act = ['binary_crossentropy']
for nn in nn_grid:
    for act in act_grid:
        for out_act in output_act:
            for exp_id in range(5):
                run_name = 'MLP_classifier_{}_{}_{}_{}'.format(nn, act, out_act, exp_id)
                graphPath = './Graph/' + run_name + '/'
                checkpointerPath = './10fold_MLPmodel/'

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
                model.add(Dense(nn*128, input_shape=X_train.shape[1:]))
                model.add(Activation(act))
                model.add(Dropout(0.4))

                model.add(Dense(nn*64))
                model.add(Activation(act))
                model.add(BatchNormalization())

                model.add(Dense(nn*32))
                model.add(Activation(act))
                model.add(Dropout(0.4))

                model.add(Dense(2))
                model.add(Activation('softmax'))

                opt = optimizers.Adam(lr=0.0001)

                logger.warn(model.summary())

                model.compile(loss=out_act, optimizer=opt, metrics=['accuracy'])


                checkpointer = callbacks.ModelCheckpoint(filepath=''.join([checkpointerPath + run_name + '.h5']), verbose=1, save_best_only=True, mode='min')
                tbCallBack = callbacks.TensorBoard(log_dir=graphPath, histogram_freq=0, write_graph=True, write_images=True)
                earlystopper = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience=6, verbose=1, min_lr=0.000001)

                train_b_size = 32
                val_b_size = 16

                training_generator = batch_generator(X_train, Y_train, train_b_size)
                validation_generator = batch_generator(X_val, Y_val, val_b_size)

                history = model.fit_generator(generator=training_generator,
                                    epochs=90,
                                    steps_per_epoch=len(X_train) // train_b_size,
                                    validation_data=validation_generator,
                                    validation_steps=len(X_val) // val_b_size,
                                    callbacks=[earlystopper,checkpointer,tbCallBack, reduce_lr],
                                    verbose=0)
