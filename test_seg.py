# -*- coding: utf8 -*-

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'

import sys
import numpy as np
import math
import cv2
from timeit import default_timer as timer
import mxnet as mx

import matplotlib.pyplot as plt
import codecs
import shutil

from collections import namedtuple

Batch = namedtuple('Batch', ['data'])

seg_data_shape = 512

cls_mean_val = np.array([[[107]],[[107]],[[107]]])
cls_std_scale = 1.0

ctx = mx.gpu(1)

def get_segmentation_mod():

    sym, arg_params, aux_params = mx.model.load_checkpoint('./segnet_bb', 0)
    mod = mx.mod.Module(symbol=sym, context=ctx, data_names=['data'], label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,512,512))], label_shapes=None)
    mod.set_params(arg_params=arg_params, aux_params=aux_params)
    return mod

def seg_img(img, mod):
    raw_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(raw_img2, (2,0,1))
    img = img[np.newaxis, :]
    img = cls_std_scale * (img.astype(np.float32) - cls_mean_val)

    mod.forward(Batch([mx.nd.array(img)]))
    pred = mod.get_outputs()[0].asnumpy()
    pred = np.argmax(pred, axis=1)[0]

    return pred

if __name__ == "__main__":
    testdir = r'/mnt/15F1B72E1A7798FD/DK2/bbanno/train/image'
    savedir = r'/mnt/15F1B72E1A7798FD/DK2/bbanno/res'

    imgfiles = [i for i in os.listdir(testdir) if i.endswith('.png')]

    seg_mod = get_segmentation_mod()

    for i,fn in enumerate(imgfiles):
        fn_path = testdir+'/'+fn
        raw_img = cv2.imread(fn_path)
        pred = seg_img(raw_img, seg_mod)

        plt.subplot(121)
        plt.imshow(raw_img)
        plt.subplot(122)
        plt.imshow(pred)
        plt.savefig(savedir +'/%d.png' % i)
        #plt.waitforbuttonpress()