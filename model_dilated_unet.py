import mxnet as mx
from mxnet import gluon
import mxnet.gluon.nn as nn
from model_unet import ConvBlock, down_block, up_block

class BottleNectPath(nn.HybridBlock):
    def __init__(self, filters, bottleneck_depth, **kwargs):
        super(BottleNectPath, self).__init__(**kwargs)
        with self.name_scope():
            
            self.cascade = nn.HybridSequential()
            for i in range(bottleneck_depth):
                self.cascade.add(
                    nn.Conv2D(filters, 3, padding=2**i, dilation=2**i, activation='relu')
                )
    def hybrid_forward(self, F, x):
        dilated_layers = []
        for c in self.cascade:
            x = c(x)
            dilated_layers.append(x)
        return F.add_n(*dilated_layers)
        

class DilatedUNet(nn.HybridBlock):
    def __init__(self, first_channels=44, **kwargs):
        super(DilatedUNet, self).__init__(**kwargs)
        with self.name_scope():
            self.d0 = down_block(first_channels)
            
            self.d1 = nn.HybridSequential()
            self.d1.add(nn.MaxPool2D(2,2,ceil_mode=True), down_block(first_channels*2))
            
            self.d2 = nn.HybridSequential()
            self.d2.add(nn.MaxPool2D(2,2,ceil_mode=True), down_block(first_channels*2**2))
            
            self.bottlenect = nn.HybridSequential()
            self.bottlenect.add(
                nn.MaxPool2D(2,2,ceil_mode=True)
                , BottleNectPath(filters=first_channels*2**3, bottleneck_depth=6)
                )
            
            self.u2 = up_block(first_channels*2**2, shrink=True)
            self.u1 = up_block(first_channels*2, shrink=True)
            self.u0 = up_block(first_channels, shrink=False)
            
            self.conv = nn.Conv2D(2,1)
    def hybrid_forward(self, F, x):
        # x => 512

        # x0 => 512
        x0 = self.d0(x)

        # x1 => 256
        x1 = self.d1(x0)

        # x2 => 128
        x2 = self.d2(x1)

        # b => 64
        b = self.bottlenect(x2)


        y2 = self.u2(b,x2)
        y1 = self.u1(y2,x1)
        y0 = self.u0(y1,x0)
        
        out = self.conv(y0)
        
        return out

if __name__ == '__main__':
    net = DilatedUNet()
    x = mx.nd.random.uniform(shape=(2,3,512,512), ctx=mx.cpu())
    net.initialize()
    #net.summary(x)
    print(net(x))