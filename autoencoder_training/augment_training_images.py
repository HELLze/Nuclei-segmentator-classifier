import h5py
import numpy as np
import cv2
import os
import random
import math
from skimage.color import hed2rgb, rgb2hed
import logging
logger = logging.getLogger()
console = logging.StreamHandler()
logger.addHandler(console)

def shift_scale(img, angle=10, scale=1, dx=0.99, dy=0.99):
    height, width = img.shape[:2]

    cc = math.cos(angle/180*math.pi) * scale
    ss = math.sin(angle/180*math.pi) * scale
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
    box1 = box0 - np.array([width/2, height/2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width/2 + dx*width, height/2+dy*height])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)
    img2 = cv2.warpPerspective(img, mat, (width, height),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    return img2

def clahe(img, clipLimit = 2.0, tileGridSize=(8, 8)):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_LAB2BGR)
    return img_output


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img2 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img2


def rgb_aug(img):
    adj_range = 0.1
    adj_add = 5
    rgb_mean = np.mean(img, axis=(0, 1), keepdims=True).astype(np.float32)
    adj_magn = np.random.uniform(1 - adj_range, 1 + adj_range, (1, 1, 3)).astype(np.float32)
    img_col = np.clip((img - rgb_mean)*adj_magn + rgb_mean + np.random.uniform(-1.0, 1.0, (1, 1, 3)) * adj_add, 0.0, 255.0)
    # img_col = (img - rgb_mean) * adj_magn + rgb_mean + np.random.uniform(-1.0, 1.0, (1, 1, 3)) * adj_add
    return img_col.astype(np.uint8)

def color_aug(img):
    adj_add = np.array([[[0.02, 0.001, 0.15]]], dtype = np.float32)
    img2 = np.clip(hed2rgb(rgb2hed(img.transpose((1, 0, 2)) / 255.0) + np.random.uniform(-1.0, 1.0, (1, 1, 3))*adj_add).transpose((1, 0, 2))*255.0, 0.0, 255.0)
    return img2.astype(np.uint8)

def vertical_flip(img):
    return cv2.flip(img, 0)

def horizontal_filp(img):
    return cv2.flip(img, 1)

def rotate(img):
    return np.rot90(img)

def transpose(img):
    return img.transpose(1, 0, 2)

image_path = './ANNOTATIONS/train/ori/'
mask_path = './ANNOTATIONS/train/mask/'

def shuffle_dataset(dataset):
    np.random.seed(40)
    np.take(dataset, np.random.rand(dataset.shape[0]).argsort(), axis=0, out=dataset)
    return dataset

def augment_images(image_path):
    logger.warn('starting augmenting images')
    augmented_image = []
    fls = sorted(os.listdir(image_path))
    for f in fls:
        img = cv2.imread(image_path + f)
        augmented_image.append(img)
        augmented_image.append(rotate(img))
        augmented_image.append(transpose(img))
        augmented_image.append(vertical_flip(img))
        np.random.seed(40)
        if np.random.rand(1)[0] > 0.5:
            clah0 = clahe(img)
            augmented_image.append(shift_scale(clah0))
        else:
            augmented_image.append(shift_scale(img))
        clah = clahe(img)
        augmented_image.append(clah)
        augmented_image.append(horizontal_filp(clah))
        augmented_image.append(rotate(clah))
        augmented_image.append(transpose(clah))
        idx = random.randint(25, 45)
        bright = increase_brightness(img, value=idx)
        augmented_image.append(bright)
        augmented_image.append(vertical_flip(bright))
        augmented_image.append(transpose(bright))
        augmented_image.append(horizontal_filp(shift_scale(bright)))
        augmented_image.append(rotate(bright))
        augmented_image.append(clahe(bright))
        he = color_aug(img)
        augmented_image.append(he)
        np.random.seed(40)
        if np.random.rand(1)[0] > 0.5:
            rgb = rgb_aug(img)
            augmented_image.append(rotate(rgb))
            augmented_image.append(transpose(rgb))
            augmented_image.append(increase_brightness(rgb))
        else:
            he2 = color_aug(img)
            augmented_image.append(rotate(he2))
            augmented_image.append(transpose(he2))
            augmented_image.append(clahe(he2))
    logger.warn('DONE with augmenting images')
    return np.asarray(augmented_image)

def augment_masks(mask_path):
    logger.warn('augmenting masks')
    augmented_masks = []
    fls = sorted(os.listdir(mask_path))
    for f in fls:
        img = cv2.imread(mask_path + f)
        augmented_masks.append(img)
        augmented_masks.append(rotate(img))
        augmented_masks.append(transpose(img))
        augmented_masks.append(vertical_flip(img))
        augmented_masks.append(shift_scale(img))
        augmented_masks.append(img)
        augmented_masks.append(horizontal_filp(img))
        augmented_masks.append(rotate(img))
        augmented_masks.append(transpose(img))
        augmented_masks.append(img)
        augmented_masks.append(vertical_flip(img))
        augmented_masks.append(transpose(img))
        augmented_masks.append(horizontal_filp(shift_scale(img)))
        augmented_masks.append(rotate(img))
        augmented_masks.append(img)
        augmented_masks.append(img)
        augmented_masks.append(rotate(img))
        augmented_masks.append(transpose(img))
        augmented_masks.append(img)
    logger.warn('DONE with mask augmentation')
    return np.asarray(augmented_masks)



augmented_images = augment_images(image_path)
augmented_masks = augment_masks(mask_path)

np.random.seed(40)
np.take(augmented_images, np.random.rand(augmented_images.shape[0]).argsort(), axis=0, out=augmented_images)

np.random.seed(40)
np.take(augmented_masks, np.random.rand(augmented_masks.shape[0]).argsort(), axis=0, out=augmented_masks)

dset = h5py.File('./training_images_augmented.h5', 'w')
dset.create_dataset('orig', data=augmented_images)
dset.close()

dset = h5py.File('./training_masks_augmented.h5', 'w')
dset.create_dataset('msk', data=augmented_masks)
dset.close()
