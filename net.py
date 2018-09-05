# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 22:10:32 2018

@author: Ashley
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 20:08:07 2018

@author: ashli
"""
import chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L

class Gan_gen(Chain):
    def __init__(self, z_dim):
        super(Gan_gen, self).__init__(
            l1=L.Linear(z_dim, 3*3*512),
            
            dcn1=L.Deconvolution2D(512,256, 2, stride=2, pad=1),
            dcn2=L.Deconvolution2D(256, 128, 2, stride=2, pad=1),
            dcn3=L.Deconvolution2D(128, 64, 2, stride=2, pad=1),
            dcn4=L.Deconvolution2D(64, 1, 3, stride=3, pad=1),
            
            bn1 = L.BatchNormalization(512),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(128),
            bn4 = L.BatchNormalization(64),
        )
        self.z_dim = z_dim
        
    def __call__(self, z):
        #損失関数
        h = self.l1(z)
        h = F.reshape(h, (z.data.shape[0], 512, 3, 3))
        
        h = F.relu(self.bn1(h))
        h = F.relu(self.bn2(self.dcn1(h)))
        h = F.relu(self.bn3(self.dcn2(h)))
        h = F.relu(self.bn4(self.dcn3(h)))
        x = F.tanh(self.dcn4(h))
        return x
    
class Gan_dis(Chain):
    def __init__(self):
        super(Gan_dis, self).__init__(
            cn1 = L.Convolution2D(1, 64, 3, stride=3, pad=1),
            cn2 = L.Convolution2D(64, 128, 2, stride=2, pad=1),
            cn3 = L.Convolution2D(128, 256, 2, stride=2, pad=1),
            cn4 = L.Convolution2D(256, 512, 2, stride=2, pad=1),
            
            l1 = L.Linear(3*3*512, 2),
            
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x):
        #損失関数
        h = F.leaky_relu(self.cn1(x))
        h = F.leaky_relu(self.cn2(h))
        h = F.leaky_relu(self.cn3(h))
        h = F.leaky_relu(self.cn4(h))
        y = self.l1(h)
        return y





















    
    
    
    
    
    
    