import os
import sys
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, autograd

# For softmax class weights initialization
@mx.init.register
class MyConstant(mx.init.Initializer):
    def __init__(self, value):
        super(MyConstant, self).__init__(value=value)
        self.value = value

    def _init_weight(self, _, arr):
        arr[:] = mx.nd.array(self.value)
        
# Note: This version only support binary segmentation
def unet_base():
    def Conv3(x, cn):
        x = nn.Conv2D(cn, 3, padding=1)(x)
        x = mx.sym.BatchNorm(x, eps=0.0001, fix_gamma=False, use_global_stats=True)
        x = nn.Activation('relu')(x)
        return x
    
    def Pool(x):
        # Trick to guarantee the up_block output is bigger than corresponding down_block.
        return nn.MaxPool2D(2,2,ceil_mode=True)(x)
    
    def down_block(x, cn):
        x = Conv3(x, cn)
        x = Conv3(x, cn)
        return x
    
    def up_block(x, s, cn): # s as sibling (i.e. down_block output)
        # x = nn.Conv2DTranspose(cn, 2, 2)(x)
        # x = nn.Activation('relu')(x)
        
        x = mx.sym.UpSampling(x, scale=2, sample_type='nearest')
        x = nn.Conv2D(cn, 1)(x)
        x = mx.sym.BatchNorm(x, eps=0.0001, fix_gamma=False, use_global_stats=True)
        x = nn.Activation('relu')(x)
        
        # As the corresponding down_block output s is smaller, then crop x to the same size with s.
        # This will make the final output shape of unet be same to the data shape.
        x = mx.sym.Crop(*[x, s], center_crop=True)
        x = mx.sym.concat(s, x, dim=1)
        x = Conv3(x, cn)
        x = Conv3(x, cn)
        return x
    
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('y')

    d0 = down_block(data, 64)
    d1 = down_block(Pool(d0), 128)
    d2 = down_block(Pool(d1), 256)
    d3 = down_block(Pool(d2), 512)
    d4 = down_block(Pool(d3), 1024)
    u3 = up_block(d4, d3, 512)
    u2 = up_block(u3, d2, 256)
    u1 = up_block(u2, d1, 128)
    u0 = up_block(u1, d0, 64)
    conv_res = nn.Conv2D(2, 1, name='conv_res')(u0)

    return conv_res, label

# Standard unet with ground truth label image
def unet_standard():
    conv_res, label = unet_base()
    return mx.sym.SoftmaxOutput(conv_res, label, multi_output=True, name='pred')

# Unet with one-hot label image and class weighted softmax
def unet_weighted_softmax(class_weights=[]):
    conv_res, label = unet_base()
    
    conv_res = mx.sym.softmax(conv_res, axis=1)
    
    assert(len(class_weights)==2)
    w = mx.sym.Variable('w',shape=(1,2), init=MyConstant([class_weights]))
    w = mx.sym.BlockGrad(w)
    # keep more stable log operation to avoid -inf result
    conv_res = mx.sym.maximum(conv_res, 0.000001)
    ce = -mx.sym.mean(label*mx.sym.log(conv_res), axis=(2,3))
    ce = mx.sym.broadcast_mul(w, ce) # weighted cross entropy by channels multiplied with weights
    ce = mx.sym.MakeLoss(ce, name='loss')
    sym = mx.sym.Group([ce, mx.sym.BlockGrad(conv_res, name='pred')])
    return sym 
