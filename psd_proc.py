import os
import numpy as np
import cv2
import pytoshop
import matplotlib.pyplot as plt

def parse_psd(fn):
    with open(fn, 'rb') as fd:
        im = pytoshop.read(fd)
        imdata = im.image_data
        layers = im.layer_and_mask_info.layer_info.layer_records
        bglayer = layers[0]
        bgimg = bglayer.channels[0].image
        fglayer = layers[1]
        print(fglayer._left, fglayer._top, fglayer._bottom, fglayer._right)
        
        fgroi = fglayer.channels[-1].image

        fgimg = np.zeros(shape=bgimg.shape)
        fgimg[fglayer._top : fglayer._bottom, fglayer._left: fglayer._right] = fgroi

        return bgimg, fgimg


psddir = '/home/kk/data/bbanno'
psdfiles = [_  for _ in os.listdir(psddir) if _.endswith('.psd')]
savedir = '/home/kk/data/bbanno/train'

save_image_dir = savedir + '/image'
save_label_dir = savedir + '/label'

os.mkdir(save_image_dir)
os.mkdir(save_label_dir)

def flipaug(img):
    a0 = cv2.flip(img, 0)
    a1 = cv2.flip(img, 1)
    a2 = cv2.flip(img, -1)
    a3 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    a4 = cv2.flip(a3, 0)
    a5 = cv2.flip(a3, 1)
    a6 = cv2.flip(a3, -1)
    return [a0, a1, a2, a3, a4, a5, a6]

for i, fn in enumerate(psdfiles):
    bgimg, fgimg = parse_psd(psddir + '/' + fn)
    cv2.imwrite('%s/%d_0.png' % (save_image_dir, i), bgimg)

    bgaugs = flipaug(bgimg)
    for j, au in enumerate(bgaugs):
        cv2.imwrite('%s/%d_%d.png' % (save_image_dir, i, j+1), au)

    cv2.imwrite('%s/%d_0.png' % (save_label_dir, i), fgimg)

    fgaugs = flipaug(fgimg)
    for j, au in enumerate(fgaugs):
        cv2.imwrite('%s/%d_%d.png' % (save_label_dir, i, j+1), au)