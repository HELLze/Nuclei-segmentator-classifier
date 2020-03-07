from __future__ import division
import os
import sys
import random
import h5py
import numpy as np
from keras.models import Model, load_model, Sequential
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

modfname = './model/lympho_other_supervised_classifier_v3.h5'
model = load_model(modfname)
model.summary()

dset = h5py.File('./nuclei_images/TCGA_test.h5', 'r')
x_test = dset['nucs'][:]
y_test = dset['labs'][:]
dset.close()

y_pred = model.predict_classes(x_test / 255.)

dict_characters = {0: 'Lymphocytes', 1: 'Other'}
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.get_cmap('Blues')):
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], fontsize=14,
                 horizontalalignment="center",
                 color="white" if cm [i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_confusion_matrix2(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.get_cmap('Blues')):
    fig, ax=plt.subplots()
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks = np.arange(cm.shape[1]),
           yticks = np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, title=title, ylabel='True label', xlabel='Predicted label')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], fontsize=16,
                 horizontalalignment="center",
                 color="white" if cm [i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

confusion_mtx = confusion_matrix(y_test, y_pred)

plot_confusion_matrix2(confusion_mtx, classes=list(dict_characters.values()))

FN = []
FP = []

for l in range(len(y_test)):
    if y_test[l] == 0:
        if y_pred[l] == 1:
            falsen = x_test[l]
            FN.append(falsen)
    if y_test[l] == 1:
        if y_pred[l] == 0:
            falsep = x_test[l]
            FP.append(falsep)

# #%%
# import imutils
# from imutils import build_montages
# import cv2
# #%%
# montage1 = build_montages(FP, (32, 32), (5, 6))
# #%%
# for mont in montage1:
#     plt.imshow(cv2.cvtColor(mont, cv2.COLOR_BGR2RGB))
#     plt.yticks([])
#     plt.xticks([])
#     plt.title('TCGA testing False Positives')
#     plt.show()
#
# #%%
from sklearn.metrics import roc_curve, auc

def get_roc_curve():
    y_predprob = model.predict(x_test / 255.)
    probs = [1-i[0] for i in y_predprob]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' %roc_auc)
    plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc="lower right")
    plt.title('ROC curve')
    plt.plot()
