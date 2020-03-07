# -*- coding: utf-8 -*-
"""
Created on Thu May  2 05:31:02 2019

@author: elzbieta
"""

from __future__ import division
from __future__ import print_function
import numpy as np
from keras.models import Model, load_model
from keras.losses import binary_crossentropy
from keras import backend as K
import tensorflow as tf
import cv2
from skimage.measure import regionprops
import openslide
import scipy.misc
import os
import time
from tqdm import tqdm
import logging
logger = logging.getLogger()
console = logging.StreamHandler()
logger.addHandler(console)

##############
##custom functions & CNN models
#############
def dice_coef2(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) +1)
#
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


modfname = './nuclei_segm_active_contour_20x_annotations_revised.h5'
model = load_model(modfname, custom_objects={'dice_coef_rounded_ch1' : dice_coef_rounded_ch1, 'dice_coef_rounded_ch0' : dice_coef_rounded_ch0, 'dice_coef2': dice_coef2, 'tf': tf, 'softmax_dice_loss' : softmax_dice_loss})

classname = './lympho_other_supervised_classifier_v3.h5'
classifier = load_model(classname)


def sliding_window(image, stepSize, windowSize, width=None, height=None, whole_slide=False):
    if not whole_slide:        
        for y_im in xrange(0, image.shape[0], stepSize):
            for x_im in xrange(0, image.shape[1], stepSize):
                yield (x_im, y_im, image[y_im:y_im + windowSize[1], x_im:x_im + windowSize[0]])
    else:
        for xm in tqdm(range(0, width, stepSize)):
            for ym in range(0, height, stepSize):
                if xm + windowSize[1] > width:
                    pw_x = width - xm
                else:
                    pw_x = windowSize[1]
                if ym + windowSize[0] > height:
                    pw_y = height - ym
                else:
                    pw_y = windowSize[0]
                patch = image.read_region((xm, ym), 0, (pw_x, pw_y))
                im1 = np.asarray(patch, dtype=np.uint8)[:, :, :3]
                im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)
                yield(xm, ym, im1)

if not os.path.exists('./mormin_predictions/'):
    os.mkdir('./mormin_predictions/')
    logger.warn('created directory for wholeslide predictions')

def get_lymphocyte_predictions(slidename, wsize, intersection_stepsize):
    prediction_start_time = time.time()
    oslide = openslide.open_slide(slidename)
    width = oslide.dimensions[0]
    height = oslide.dimensions[1]
    stem, ext = os.path.splitext(slidename)
    csvoutput = '{}{}_lymphocytes.csv'.format('./mormin_predictions/', stem.split('/')[-1])
    res = open(csvoutput, 'w')
    labelrow = "Obj No,Centroid_coord_x,Centroid_coord_y"
    print(labelrow, file=res)
    ID = 0
    tilID = 0
    for (x, y, tile) in sliding_window(oslide, stepSize=wsize, windowSize=(wsize, wsize), width=width, height=height, whole_slide=True):

        d0 = 256 - (tile.shape[0] % 256)
        d1 = 256 - (tile.shape[1] % 256)
        im2 = np.zeros(shape=(tile.shape[0] + d0, tile.shape[1] + d1, 3), dtype=np.float)
        im2[:tile.shape[0], :tile.shape[1], :] = tile
        im3 = np.zeros(shape=(im2.shape[0], im2.shape[1], 3), dtype=np.int)
        for (x_p, y_p, patch) in sliding_window(im2, stepSize=intersection_stepsize, windowSize=(256, 256)):
            if patch.shape[0] == 256 and patch.shape[1] == 256:
                if np.std(patch) > 2:
                    patch = patch.astype(np.float) / 255.
                    w = model.predict(patch.reshape([1, 256, 256, 3]), batch_size=1, verbose=0)
                    a = w[0, :, :, 0]
                    b = w[0, :, :, 1]
                    c = np.zeros((w.shape[0], w.shape[1]), dtype=np.uint8)

                    b[b > 0.5] = 1
                    ax = a - b

                    ax[ax <= 0.5] = 0
                    ax[ax > 0.5] = 255

                    b[b <= 0.5] = 0
                    b[b > 0.5] = 255
                    b = b.astype(np.uint8)
                    ax = ax.astype(np.uint8)
                else:
                    ax = 0
                    b = 0
                    c = 0
                im3[y_p:y_p + 256, x_p:x_p + 256, 0] += ax
                im3[y_p:y_p + 256, x_p:x_p + 256, 1] += b
                im3[y_p:y_p + 256, x_p:x_p + 256, 2] += c

        im3 = im3[:-d0, :-d1, :]

        im3[im3 > 255] = 255
        im3 = im3.astype(np.uint8)
        labeled_array, _ = scipy.ndimage.label(im3[:, :, 0])
        objs = scipy.ndimage.find_objects(labeled_array)
        nucleus_array = []
        coord_array = []
        for n, ob in enumerate(objs):
            if n > 0:
                m = im3[:, :, 0][ob]
                x1 = x + ob[1].start + int((ob[1].stop - ob[1].start) / 2)
                y1 = y + ob[0].start + int((ob[0].stop - ob[0].start) / 2)

                m2 = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
                img2 = tile[ob]

                maskout = cv2.subtract(m2, img2)
                maskout = cv2.subtract(m2, maskout)
                maskout[np.where((maskout == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

                desired_size = 32
                old_size = maskout.shape[:2]

                delta_w = desired_size - old_size[1]
                delta_h = desired_size - old_size[0]
                if delta_h >= 0 and delta_w >= 0:
                    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                    left, right = delta_w // 2, delta_w - (delta_w // 2)

                    color = [255, 255, 255]
                    new_im = cv2.copyMakeBorder(maskout, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
                    new_im = new_im / 255.
                    nucleus_array.append(new_im)
                    coord_array.append([x1, y1])

                else:
                    try:
                        coef = desired_size / np.amax(old_size)
                        resized_w = int(old_size[1] * coef)
                        resized_h = int(old_size[0] * coef)
                        resized_img = cv2.resize(maskout, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
                        delta_2w = desired_size - resized_w
                        delta_2h = desired_size - resized_h
                        top, bottom = delta_2h // 2, delta_2h - (delta_2h // 2)
                        left, right = delta_2w // 2, delta_2w - (delta_2w // 2)
                        color = [255, 255, 255]
                        new_im = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
                        new_im = new_im / 255.
                        nucleus_array.append(new_im)
                        coord_array.append([x1, y1])
                    except cv2.error:
                        """in rare cases, resized_* dimensions are rounded to integer 0, hence cv2 error """
                        continue
        if len(nucleus_array) >= 1:
            nucleus_array = np.asarray(nucleus_array)
            predictions = classifier.predict_classes(nucleus_array, batch_size=256)
            for nuc_id, prediction in enumerate(predictions):
                if prediction == 0:
                    row = ",".join((str(ID), str(int(coord_array[nuc_id][0])), str(int(coord_array[nuc_id][1]))))
                    print(row, file=res)
                    ID += 1
        tilID += 1

    res.close()
    prediction_end_time = time.time()
    execution_time = round((prediction_end_time - prediction_start_time) / 3600, 2)
    logger.warn('***Done with slide {}, execution time: {} hrs***'.format(slidename, execution_time))


processed_slides = [slide.replace('_lymphocytes.csv', '') for slide in sorted(os.listdir('./mormin_predictions/')) if os.path.isfile('./mormin_predictions/' + slide)]
slidepath = '/home/elzbieta/Documents/Lymphocytes_v2/MORMIN/mormin/'
slidenames = [name for name in sorted(os.listdir(slidepath)) if os.path.isfile(slidepath + name) and name.replace('.svs', '') not in processed_slides]
logger.warn('total slides to predict: {}'.format(len(slidenames)))
logger.warn('***starting to predict***')
for slide in slidenames:
    slidename = slidepath + slide
    get_lymphocyte_predictions(slidename, 512, intersection_stepsize=128)
