import os
import cv2
import numpy as np
import logging
from unet_symbol import *
logging.basicConfig(level=logging.INFO)

train_path = './data/train'
gt_path = './data/train_label'

proc_src = lambda fn: cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2RGB)[np.newaxis,:]
proc_gt = lambda fn: cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2GRAY)[np.newaxis,:]
imgs = np.array([proc_src(train_path+'/'+i) for i in os.listdir(train_path) if i.endswith('.png')],dtype=np.float32)
gts = np.array([proc_gt(gt_path+'/'+i) for i in os.listdir(gt_path) if i.endswith('.png')],dtype=np.float32)

# convert the ground truth label to one hot label
gts_one_hot = np.stack((1-gts[:,0,:,:], gts[:,0,:,:]), axis=1)

nd_iter = mx.io.NDArrayIter(data={'data':imgs}, label={'y': gts_one_hot}, batch_size=16)

sym = unet_weighted_softmax(class_weights=[0.063872,0.936128]) # background, foreground class weights
mod = mx.mod.Module(sym, data_names=['data'], label_names=['y'], context=[mx.gpu(0),mx.gpu(1),mx.gpu(2),mx.gpu(3)])
mod.bind(data_shapes=nd_iter.provide_data, label_shapes=nd_iter.provide_label)
mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
mod.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate':0.01, 'momentum': 0.9})

# segmentation accuracy, simply count the correct predictions
def seg_acc(label, pred):
    c = (label[:,1,:,:].astype(np.int64) == np.argmax(pred, axis=1)).sum()
    return c*2, label.size

# foreground IOU score
def fg_acc(label, pred):
    l = label[:,1,:,:].astype(np.int64)
    p = (pred[:,1,:,:] > 0.5)
    a = np.bitwise_and(p, l)
    b = np.bitwise_or(p, l)
    return a.sum(), b.sum()+1

seg_acc_metric = mx.metric.CustomMetric(seg_acc, output_names=['pred_output'])
fg_acc_metric = mx.metric.CustomMetric(fg_acc, output_names=['pred_output'])
metric = mx.metric.create([seg_acc_metric, fg_acc_metric])

# training...
for epoch in range(90):
    nd_iter.reset()
    metric.reset()
    for batch in nd_iter:
        mod.forward(batch, is_train=True)       # compute predictions
        mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
        mod.backward()                          # compute gradients
        mod.update()                            # update parameters
    print('Epoch %d, Training %s' % (epoch, metric.get()))
    
mod.save_checkpoint('u_net', 89)
