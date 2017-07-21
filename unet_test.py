import os
import sys
import mxnet as mx

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

sym, arg_params, aux_params = mx.model.load_checkpoint('u_net', 59)

context=mx.gpu()
mod = mx.mod.Module(symbol=all_layers['pred_output'], context=context, data_names=['data'], label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 500, 500))], label_shapes=None, force_rebind=False)
mod.set_params(arg_params=arg_params, aux_params=aux_params, force_init=False)

im_raw = Image.open('test.png')
im = np.array(im_raw, dtype=np.float32)
im = im[np.newaxis,np.newaxis,:,:]

# reshape to process variant image size
mod.reshape(data_shapes=[('data', (1, 1, im.shape[2], im.shape[3]))])
mod.forward(Batch([mx.nd.array(im)]), is_train=False)

f = mod.get_outputs()[0].asnumpy()
fore_ground = f[0,1,:,:]
pred_label = (f1>0.5)*255
pred_label = res.astype(np.uint8)
res = np.hstack((np.array(im_raw), pred_label))
plt.imshow(f1>0.5)
plt.show()