import mxnet as mx
import mxnet.gluon.nn as nn


def ConvBlock(channels, kernel_size):
    out = nn.HybridSequential()
    #with out.name_scope():
    out.add(
        nn.Conv2D(channels, kernel_size, padding=kernel_size // 2, use_bias=False),
        nn.BatchNorm(),
        nn.Activation('relu')
    )
    return out

def down_block(channels):
    out = nn.HybridSequential()
    #with out.name_scope():
    out.add(
        ConvBlock(channels, 3),
        ConvBlock(channels, 3)
    )
    return out


class up_block(nn.HybridBlock):
    def __init__(self, channels, shrink=True, **kwargs):
        super(up_block, self).__init__(**kwargs)
        #with self.name_scope():
        self.upsampler = nn.Conv2DTranspose(channels=channels, kernel_size=4, strides=2, 
                                            padding=1, use_bias=False, groups=channels, weight_initializer=mx.init.Bilinear())
        self.upsampler.collect_params().setattr('gred_req', 'null')

        self.conv1 = ConvBlock(channels, 1)
        self.conv3_0 = ConvBlock(channels, 3)
        if shrink:
            self.conv3_1 = ConvBlock(channels // 2, 3)
        else:
            self.conv3_1 = ConvBlock(channels, 3)
    def hybrid_forward(self, F, x, s):
        x = self.upsampler(x)
        x = self.conv1(x)
        x = F.relu(x)
        
        x = F.Crop(*[x,s], center_crop=True)
        x = F.concat(s,x, dim=1)
        #x = s + x
        x = self.conv3_0(x)
        x = self.conv3_1(x)
        return x

class UNet(nn.HybridBlock):
    def __init__(self, first_channels=64, **kwargs):
        super(UNet, self).__init__(**kwargs)
        with self.name_scope():
            self.d0 = down_block(first_channels)
            
            self.d1 = nn.HybridSequential()
            self.d1.add(nn.MaxPool2D(2,2,ceil_mode=True), down_block(first_channels*2))
            
            self.d2 = nn.HybridSequential()
            self.d2.add(nn.MaxPool2D(2,2,ceil_mode=True), down_block(first_channels*2**2))
            
            self.d3 = nn.HybridSequential()
            self.d3.add(nn.MaxPool2D(2,2,ceil_mode=True), down_block(first_channels*2**3))
            
            self.d4 = nn.HybridSequential()
            self.d4.add(nn.MaxPool2D(2,2,ceil_mode=True), down_block(first_channels*2**4))
            
            self.u3 = up_block(first_channels*2**3, shrink=True)
            self.u2 = up_block(first_channels*2**2, shrink=True)
            self.u1 = up_block(first_channels*2, shrink=True)
            self.u0 = up_block(first_channels, shrink=False)
            
            self.conv = nn.Conv2D(2,1)
    def hybrid_forward(self, F, x):
        x0 = self.d0(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)

        y3 = self.u3(x4,x3)
        y2 = self.u2(y3,x2)
        y1 = self.u1(y2,x1)
        y0 = self.u0(y1,x0)
        
        out = self.conv(y0)
        
        return out