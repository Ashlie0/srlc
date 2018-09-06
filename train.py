# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:07:32 2018

@author: ashli
"""


import chainer.functions as F
import numpy as np
from chainer import cuda, optimizers
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, srlc):
        self.srlc=srlc
    def fit(self, X, epochs=100, plotting=True):
        self.X=X
        self.epochs=epochs
        self.plotting=plotting
        
        o_srlc=optimizers.Adam(alpha=1e-5, beta1=0.1)
        o_srlc.setup(self.srlc)
        self.loss=[]
        for epoch in range(1,epochs+1):
            sum_loss_1=np.float32(0)
            sum_loss_2=np.float32(0)
            sum_loss_sum=np.float32(0)
            
            for i in range(120):
                x1=self.srlc.fwd1(X)
                x2=self.srlc.fwd2(X)
                y1=np.ones((120,4,22,22), dtype=np.float32)
                y2=np.ones((120,8,11,11), dtype=np.float32)
                y1=cuda.to_gpu(y1,0)
                y2=cuda.to_gpu(y2,0)
                loss_1 = F.mean_absolute_error(x1, y1)
                loss_2 = F.mean_absolute_error(x2, y2)
                loss_sum=loss_1+loss_2
                
                self.srlc.cleargrads()
                if loss_1.data>loss_2.data:
                    loss_1.backward()
                else:
                    loss_2.backward()
                o_srlc.update()
                
                sum_loss_1 += loss_1.data
                sum_loss_2 += loss_2.data
                sum_loss_sum += loss_sum.data
            
            print('epoch:', epoch, 'sum_loss_1:', sum_loss_1,\
                  'sum_loss_2:', sum_loss_2)
                
            self.loss.append(sum_loss_sum)
            
            if self.plotting and epoch%(self.epochs//10)==0:
                z=self.srlc.fwd1(self.X)
                z=z.data
                z=cuda.to_cpu(z)
                re=z[0][0]
                plt.imshow(re)
                plt.gray()
                plt.show()
                plt.close()
        #print(self.loss)