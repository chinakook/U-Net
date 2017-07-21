import os
import sys
import mxnet as mx

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

sym, arg_params, aux_params = mx.model.load_checkpoint('u_net', 59)
all_layers = sym.get_internals()

context=mx.gpu()
mod = mx.mod.Module(symbol=all_layers['conv_res_output'], context=context, data_names=['data'], label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))], label_shapes=None, force_rebind=False)
mod.set_params(arg_params=arg_params, aux_params=aux_params, force_init=False, allow_missing=True, allow_extra=True)

im_raw = Image.open('test.png')
im = np.array(im_raw, dtype=np.float32)
im = np.transpose(im, (2,0,1))
im = im[np.newaxis,:,:]

# reshape to process variant image size
mod.reshape(data_shapes=[('data', (1, 1, im.shape[2], im.shape[3]))])
mod.forward(Batch([mx.nd.array(im)]), is_train=False)

f = mx.nd.softmax(mod.get_outputs()[0], axis=1).asnumpy()
fore_ground = f[0,1,:,:]
pred_label = (fore_ground>0.5)*255
pred_label = pred_label.astype(np.uint8)

plt.subplot(121)
plt.imshow(pred_label>0.5)
plt.subplot(122)
plt.imshow(im_raw)
plt.show()
