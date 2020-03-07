# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 06:04:51 2019

@author: elzbieta
"""

from __future__ import division
import os
import sys
import random
import h5py
import numpy as np
from keras.models import Model, load_model
from keras import metrics
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras import optimizers
from keras.utils import Sequence
import logging

logger = logging.getLogger()
console = logging.StreamHandler()
logger.addHandler(console)

dset = h5py.File('./training_images_augmented.h5', 'r')
X_train = dset['orig'][:]
dset.close()

dset = h5py.File('./training_masks_augmented.h5', 'r')
Y_train = dset['msk'][:, :, :, :2]
dset.close()

dset = h5py.File('./PILNAS_MANO_ANOTACIJOS/validate_70.h5', 'r')
X_val = dset['orig'][:]
Y_val = dset['nuc_msk'][:, :, :, :2]
dset.close()

logger.warn('Loaded Training set of {} images'.format(X_train.shape[0]))
logger.warn('Loaded Validation set of {} images'.format(X_val.shape[0]))

imh,imw = 256,256


def batch_generator(X, y, batch_size):
    nsamples = len(X)
    start_idx = 0
    while True:
        if start_idx + batch_size > nsamples:
            start_idx = 0
        x_batch = X[start_idx:start_idx+batch_size, ...] / 255.
        y_batch = y[start_idx:start_idx+batch_size, ...] / 255.
        start_idx += batch_size
        yield np.asarray(x_batch), np.asarray(y_batch)


def dice_coef2(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) +1)

def dice_coef_loss2(y_true, y_pred):
    return 1-dice_coef2(y_true, y_pred)

def softmax_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) * 0.1 + dice_coef_loss2(y_true, y_pred) * 0.9


def dice_coef_rounded_ch0(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_rounded_ch1(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f = K.flatten(K.round(y_pred[..., 1]))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)



class Texture2D:
    def __init__ (self, t=None):
        self.t = t
    def do(self,nf=16,kernel=(3,3),nl=2,d=0.1):
        nf1 = self.t.get_shape().as_list()[3]*8
        if nf1 > nf:
            nf = nf1
        t1 = Conv2D(1, (1, 1), activation='elu', kernel_initializer='he_normal', padding='same') (self.t)
        t1 = Conv2D(nf, kernel, activation='elu', kernel_initializer='he_normal', padding='same') (t1)
        t1 = Dropout(d) (t1)
        t1 = Conv2D(nf, kernel, activation='elu', kernel_initializer='he_normal', padding='same') (t1)
        return t1

class Color2D:
    def __init__ (self, c=None):
        self.c = c
    def do(self,nf=16,kernel=(3,3),nl=2,d=0.1):
        nf1 = self.c.get_shape().as_list()[3]*8
        if nf1 > nf:
            nf = nf1
        c1 = Conv2D(nf, (1, 1), activation='elu', kernel_initializer='he_normal', padding='same') (self.c)
        c1 = Conv2D(nf, kernel, activation='elu', kernel_initializer='he_normal', padding='same') (c1)
        c1 = Dropout(d) (c1)
        c1 = Conv2D(nf, kernel, activation='elu', kernel_initializer='he_normal', padding='same') (c1)
        return c1

act = 'elu'

don = 0.2
nn = 32
outfname = 'nuclei_segm_active_contour_20x_annotations_revised'
logfolder = ''.join(['./Graph/',outfname])
if not os.path.exists('./Graph/'):
    os.mkdir('./Graph/')
    os.mkdir('./models/')
if not os.path.exists(logfolder):
    os.mkdir(logfolder)
    logger.warn("Logfolder created")
else:
    logger.warn("Logfolder found")

s = Input((imw,imh,3))
c1 = Color2D(s).do()
t1 = Texture2D(s).do()

m1 = concatenate([c1, t1])
m1 = Conv2D(32, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (m1)
m1 = Dropout(don) (m1)
m1 = Conv2D(32, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (m1)
p1 = MaxPooling2D((2, 2)) (m1)

s2 = Lambda(lambda image: tf.image.resize_images(image, (128, 128))) (s)
c2 = Color2D(s2).do()
t2 = Texture2D(s2).do()

m2 = concatenate([c2, t2])
m2 = concatenate([m2, p1])
m2 = Conv2D(32, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (m2)
m2 = Dropout(don) (m2)
m2 = Conv2D(32, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (m2)
p2 = MaxPooling2D((2, 2)) (m2)

s3 = Lambda(lambda image: tf.image.resize_images(image, (64, 64))) (s)
c3 = Color2D(s3).do()
t3 = Texture2D(s3).do()

m3 = concatenate([c3, t3])
m3 = concatenate([m3, p2])
m3 = Conv2D(32, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (m3)
m3 = Dropout(don) (m3)
m3 = Conv2D(32, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (m3)
p3 = MaxPooling2D((2, 2)) (m3)

s4 = Lambda(lambda image: tf.image.resize_images(image, (32, 32))) (s)
c4 = Color2D(s4).do()
t4 = Texture2D(s4).do()

m4 = concatenate([c4, t4])
m4 = concatenate([m4, p3])
m4 = Conv2D(nn, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (m4)
m4 = Dropout(don) (m4)
m4 = Conv2D(nn, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (m4)

u1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (m4)
u1 = concatenate([u1, m3])
m5 = Conv2D(32, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (u1)
m5 = Dropout(don) (m5)
m5 = Conv2D(32, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (m5)


u2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (m5)
u2 = concatenate([u2, m2])
m6 = Conv2D(32, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (u2)
m6 = Dropout(don) (m6)
m6 = Conv2D(32, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (m6)


u3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (m6)
u3 = concatenate([u3, m1])
m7 = Conv2D(32, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (u3)
m7 = Dropout(don) (m7)
m7 = Conv2D(32, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (m7)



o = Conv2D(2, (1, 1), activation='sigmoid') (m7)

# opt = optimizers.Adam(lr=0.0001)
# opt = optimizers.SGD(lr=0.001, momentum=0.985)
model = Model(inputs=[s], outputs=[o])
# model.compile(optimizer='adam', loss=[softmax_dice_loss], metrics=[dice_coef_rounded_ch1, dice_coef_rounded_ch2, dice_coef_rounded_ch0, metrics.categorical_crossentropy])
model.compile(optimizer='adam', loss=[softmax_dice_loss], metrics=[dice_coef2, binary_crossentropy, dice_coef_rounded_ch0, dice_coef_rounded_ch1])

model.summary()
logger.warn('preparing for training')

earlystopper = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
checkpointer = ModelCheckpoint(''.join(['./models/',outfname,'.h5']), verbose=1, save_best_only=True)
tbCallBack = TensorBoard(log_dir=logfolder,histogram_freq=0,write_graph=False, write_images=False)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience=4, verbose=1, min_lr=0.0000001)
training_generator = batch_generator(X_train, Y_train, 1)
validation_generator = batch_generator(X_val, Y_val, 1)

model.fit_generator(generator=training_generator,
                    epochs=40,
                    steps_per_epoch=len(X_train) // 1,
                    validation_data=validation_generator,
                    validation_steps=len(X_val) // 1,
                    callbacks=[earlystopper,checkpointer,tbCallBack, reduce_lr],
                    verbose=0)
