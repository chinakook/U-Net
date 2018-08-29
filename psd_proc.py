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


psddir = '/mnt/15F1B72E1A7798FD/DK2/bbanno'
psdfiles = [_  for _ in os.listdir(psddir) if _.endswith('.psd')]
savedir = '/mnt/15F1B72E1A7798FD/DK2/bbanno/train'

save_image_dir = savedir + '/image'
save_label_dir = savedir + '/label'

os.mkdir(save_image_dir)
os.mkdir(save_label_dir)

for i, fn in enumerate(psdfiles):
    bgimg, fgimg = parse_psd(psddir + '/' + fn)
    cv2.imwrite('%s/%d.png' % (save_image_dir, i), bgimg)
    cv2.imwrite('%s/%d.png' % (save_label_dir, i), fgimg)