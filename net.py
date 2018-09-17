# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 23:11:14 2018

@author: ashli
"""


from chainer import Chain
import chainer.functions as F
import chainer.links as L
import chainer

class srlc(Chain):
    def __init__(self):
        initializer = chainer.initializers.HeNormal()
        super(srlc, self).__init__(
            #パラメータを含む関数の宣言
            cn1 = L.Convolution2D(1, 4, 2, initialW=initializer),#(45,45)→(4,44,44)
            cn2 = L.Convolution2D(4, 16, 1, initialW=initializer),#(4,22,22)→(4,22,22)
            bnorm1 = L.BatchNormalization(4),
            bnorm2 = L.BatchNormalization(16),
            )
    def fwd1(self, x):
        h1 = self.cn1(x)
        #h1 = F.tanh(h1)
        h1 = F.sigmoid(h1)
        #h1 = self.bnorm1(h1)
        h1 = F.max_pooling_2d(h1, 2)#(4,44,44)→(4,22,22)
        return h1
    def fwd2(self, x):
        h2 = self.cn1(x)
        #h2 = F.tanh(h2)
        h2 = F.sigmoid(h2)
        #h2 = self.bnorm1(h2)
        h2 = F.max_pooling_2d(h2, 2)#(4,44,44)→(4,22,22)
        h2 = self.cn2(h2)
        #h2 = F.tanh(h2)
        h2 = F.sigmoid(h2)
        #h2 = self.bnorm2(h2)
        h2 = F.max_pooling_2d(h2, 2)#(16,22,22)→(16,11,11)
        return h2
