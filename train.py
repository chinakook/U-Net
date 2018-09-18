# coding: utf-8
import os

import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag

from mxnet.gluon.data import Dataset, DataLoader
from mxnet import image

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import cv2

import time

from model_unet import UNet
from model_dilated_unet import DilatedUNet
        

class MyDataSet(Dataset):
    def __init__(self, root, split, transform=None, use_mask=False):
        self.root = os.path.join(root, split)
        self.transform = transform
        
        self.img_paths = []
        
        self._img = os.path.join(root, split, 'image', '{}.png')
        self._use_mask = use_mask
        if self._use_mask:
            self._mask = os.path.join(root, split, 'mask', '{}.png')
        self._lbl = os.path.join(root, split, 'label', '{}.png')
        
        for fn in os.listdir(os.path.join(root, split, 'label')):
            if len(fn) > 3 and fn[-4:] == '.png':
                self.img_paths.append(fn[:-4])
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self._img.format(self.img_paths[idx])
        if self._use_mask:
            mask_path = self._mask.format(self.img_paths[idx])
        lbl_path = self._lbl.format(self.img_paths[idx])

        img = cv2.imread(img_path)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)

        all_count = np.prod(lbl.shape)
        fg_count = np.count_nonzero(lbl)
        bg_count = all_count - fg_count
        alpha = 1. / fg_count
        beta = 1. / bg_count
        
        alpha = alpha / (alpha + beta)
        beta = beta / (alpha + beta)
        
        if self._use_mask:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = np.bitwise_not(mask)
            lbl = np.bitwise_or(mask, lbl/255)
        else:
            lbl = lbl / 255

        if self.transform is not None:
            img, lbl = self.transform(img, lbl)


        weight = lbl * alpha + (1 - lbl) * beta
        # plt.subplot(121)
        # plt.imshow(img[2].asnumpy())
        # plt.subplot(122)
        # plt.imshow(lbl.asnumpy().astype(np.float32))
        # plt.waitforbuttonpress()
        
        return img, lbl, weight


class ToNDArray():
    def __call__(self, img, lbl):
        img = mx.nd.array(img)
        lbl = mx.nd.array(lbl) #, dtype=np.int32)
        
        return img, lbl

class Normalize:
    def __init__(self, mean, std):
        self.mean = mx.nd.array(mean)
        self.std = mx.nd.array(std)

    def __call__(self, img, lbl):
        img = mx.image.color_normalize(img, self.mean, self.std)
        img = mx.nd.transpose(img, (2, 0, 1))

        return img, lbl

class Compose:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, img, lbl):
        for t in self.trans:
            img, lbl = t(img, lbl)
        return img, lbl

class Resize:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        
    def __call__(self, img, lbl):
        img = cv2.resize(img, (self.w, self.h), 0, 0, cv2.INTER_LINEAR)
        lbl = cv2.resize(lbl, (self.w, self.h), 0, 0, cv2.INTER_NEAREST)
        
        return img, lbl

class RandomCrop:
    def __init__(self, crop_size=None, scale=None):
        # assert min_scale <= max_scale
        self.crop_size = crop_size
        self.scale = scale
        # self.min_scale = min_scale
        # self.max_scale = max_scale

    def __call__(self, img, lbl):
        if self.crop_size:
            crop = self.crop_size
        else:
            crop = min(img.shape[0], img.shape[1])
        
        if crop > min(img.shape[0], img.shape[1]):
            crop = min(img.shape[0], img.shape[1])
        print(crop, img.shape[0], img.shape[1])  
        if self.scale:
            factor = random.uniform(self.scale, 1.0)
            crop = int(round(crop * factor))

        x = random.randint(0, img.shape[1] - crop)
        y = random.randint(0, img.shape[0] - crop)

        img = img[y:y+crop, x:x+crop,:]
        lbl = lbl[y:y+crop, x:x+crop,:]
        return img, lbl

class RandomAffine:
    def __init__(self):
        pass
    
    def __call__(self, img, lbl):
        #scale = random.uniform(1, 1)
        theta = random.uniform(-np.pi, np.pi)
        flipx = random.choice([-1,1])
        flipy = random.choice([-1,1])
        imgh = img.shape[0]
        imgw = img.shape[1]
        T0 = np.array([[1,0,-imgw/2.],[0,1,-imgh/2.],[0,0,1]])
        S = np.array([[flipx,0,0],[0, flipy,0],[0,0,1]])
        R = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0],[0,0,1]])
        T1 = np.array([[1,0,imgw/2.],[0,1,imgh/2.],[0,0,1]])
        M = np.dot(S, T0)
        M = np.dot(R, M)
        M = np.dot(T1, M)
        M = M[0:2,:]
        
        img = cv2.warpAffine(img, M, dsize=(imgw, imgh), flags=cv2.INTER_LINEAR)
        lbl = cv2.warpAffine(lbl, M, dsize=(imgw, imgh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        
        return img, lbl


my_train_aug = Compose([
    #RandomCrop(crop_size=5000),
    #Resize(500,500),
    #RandomAffine(),
    ToNDArray(),
    Normalize(nd.array([107]), nd.array([1]))
])


my_train = MyDataSet('/mnt/15F1B72E1A7798FD/DK2/bbanno', 'train', my_train_aug)


# my_val_aug = Compose([
#     ToNDArray(),
#     Normalize(nd.array([107]), nd.array([1]))
# ])


# my_val = MyDataSet('/home/kk/data/ema', 'train', my_val_aug)

train_loader = DataLoader(my_train, batch_size=8, shuffle=True, last_batch='rollover')

ctx = [mx.gpu(1), mx.gpu(2)]

net = DilatedUNet()
net.hybridize()

net.collect_params().initialize(ctx=ctx)



# x = mx.sym.var('data')
# y = net(x)

# mx.viz.plot_network(y,shape={'data':(8,3,500,500)}, node_attrs={'shape':'oval','fixedsize':'fasl==false'}).view()
# exit(0)

num_epochs = 180
num_steps = len(my_train) // 8

trainer = gluon.Trainer(net.collect_params(), 'sgd', {
    'learning_rate': 0.1,
    'wd': 0.0005,
    'momentum': 0.9,
    'lr_scheduler': mx.lr_scheduler.PolyScheduler(num_steps * num_epochs, 0.1,  2, 0.0001)
})

criterion = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)

class SegMetric(mx.metric.EvalMetric):
    """CalculSegMetricate metrics for Seg training """
    def __init__(self, eps=1e-8, use_mask=False):
        super(SegMetric, self).__init__('Seg')
        self.eps = eps
        self.num = 2
        self.ac = 0
        self.ce = 0
        self.name = ['Accuracy_background','Accuracy_foreground']

        self.use_mask = use_mask

        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        """
        # get generated multi label from network
        
        for l ,p in zip(labels, preds):
            l = l.asnumpy().astype(np.int32)
            p = p.asnumpy()
            
            pl = np.argmax(p, axis=1)


            if self.use_mask:
                m = l != 255
                m255 = 255 - m*255
                pl = np.bitwise_or(pl, m255)
            
            bg_gt = l==0
            fg_gt = l==1
            
            bg = bg_gt * (pl == 0) #np.bitwise_and(bg_gt, pl==0)
            fg = fg_gt * (pl == 1) #np.bitwise_and(fg_gt, pl==1)
            
            # plt.subplot(121)
            # plt.imshow(fg[0])
            # plt.subplot(122)
            # plt.imshow(fg_gt[0])
            # plt.show()
            
            self.sum_metric[0] += bg.sum()
            self.sum_metric[1] += fg.sum()
            # print(fg.sum())
            
            self.num_inst[0] += bg_gt.sum()
            self.num_inst[1] += fg_gt.sum()

        
    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / float(self.num_inst))
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan')                 for x, y in zip(self.sum_metric, self.num_inst)]            
            return (names, values)

metrics = [SegMetric(use_mask=False)]

for epoch in range(num_epochs):
    t0 = time.time()
    total_loss = 0
    for m in metrics:
        m.reset()
    for data, label, weight in train_loader:
        batch_size = data.shape[0]
        dlist = gluon.utils.split_and_load(data, ctx)
        llist = gluon.utils.split_and_load(label, ctx)
        wlist = gluon.utils.split_and_load(weight, ctx)
        #mlist = [y!=255 for y in llist]
        with ag.record():
            #losses = [criterion(net(X), y, m) for X, y in zip(dlist, llist, mlist)]
            preds = [net(X) for X in dlist]
            losses = []
            for i in range(len(preds)):
                l = criterion(preds[i], llist[i], wlist[i]) # , mlist[i])
                losses.append(l)
            ag.backward(losses)
        total_loss += sum([l.sum().asscalar() for l in losses])
        trainer.step(batch_size)
        #print(label.shape, preds.shape)
        for m in metrics:
            m.update(labels=llist, preds=preds)
    
    for m in metrics:
        name, value = m.get()

    t1 = time.time()
    print(epoch, t1-t0, total_loss, name, value)


net.export('segnet_bb')

