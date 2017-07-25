import os
import cv2
import numpy as np
import logging
from unet_symbol import *
logging.basicConfig(level=logging.INFO)

train_path = './data/train'
gt_path = './data/train_label'

proc_src = lambda fn: np.transpose(cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2RGB), (2, 0, 1))
proc_gt = lambda fn: cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2GRAY)[np.newaxis,:]
imgs = np.array([proc_src(train_path+'/'+i) for i in os.listdir(train_path) if i.endswith('.png')],dtype=np.float32)
gts = np.array([proc_gt(gt_path+'/'+i) for i in os.listdir(gt_path) if i.endswith('.png')],dtype=np.float32)

nd_iter = mx.io.NDArrayIter(data={'data':imgs}, label={'y': gts}, batch_size=16)

sym = unet_standard()
mod = mx.mod.Module(sym, data_names=['data'], label_names=['y'], context=[mx.gpu(0),mx.gpu(1),mx.gpu(2),mx.gpu(3)])
mod.bind(data_shapes=nd_iter.provide_data, label_shapes=nd_iter.provide_label)
mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
mod.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate':0.01, 'momentum': 0.9})

# lr_sch = mx.lr_scheduler.MultiFactorScheduler([60 * nd_iter.num_data // nd_iter.batch_size,80 * nd_iter.num_data // nd_iter.batch_size], 0.2)
# optimizer =mx.optimizer.SGD(momentum = 0.9, learning_rate=0.01, lr_scheduler=lr_sch, rescale_grad=1.0/nd_iter.batch_size)
# mod.init_optimizer(optimizer=optimizer)

# training...
mod.fit(nd_iter, num_epoch=60,eval_metric= 'acc')

mod.save_checkpoint('u_net', 59)
