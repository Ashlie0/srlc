# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 22:51:51 2018

@author: Ashley
"""

import chainer.functions as F
import numpy as np
from chainer import cuda, Variable, optimizers

import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, gen, dis):
        self.gen=gen
        self.dis=dis
        self.z_dim=gen.z_dim
    def fit(self, X, epochs=100, batch_size=1000, plotting=True):
        """
        param X: 正しい画像
        epochs:訓練回数
        batch_size:バッチ数
        plotting:Trueのときプロットする
        """
        
        self.X=X
        self.epochs=epochs
        self.batchsize=batch_size
        self.plotting=plotting
        
        n_train=X.shape[0]
        o_gen=optimizers.Adam(alpha=1e-5, beta1=0.1)
        o_dis=optimizers.Adam(alpha=1e-5, beta1=0.1)
        
        o_gen.setup(self.gen)
        o_dis.setup(self.dis)
        
        self.loss=[]
        for epoch in range(1,epochs+1):
            perm = np.random.permutation(n_train)
            sum_loss_of_dis=np.float32(0)
            sum_loss_of_gen=np.float32(0)
            
            for i in range(int(n_train / batch_size)):
                z=np.random.uniform(-1, 1, (batch_size, self.z_dim))
                z=z.astype(dtype=np.float32)
                z=cuda.to_gpu(z, device=0)
                z=Variable(z)
                
                x=self.gen(z)
                y1=self.dis(x)
                
                #0と判別させたい(gen)
                #→0のみの配列とのcross_entropyを取る
                loss_gen = F.softmax_cross_entropy(
                        y1, Variable(cuda.to_gpu(
                                        np.zeros(batch_size, dtype=np.int32),
                                        device=0)))
                #1と判別したい(dis)
                loss_dis = F.softmax_cross_entropy(
                        y1, Variable(cuda.to_gpu(
                                        np.ones(batch_size, dtype=np.int32),
                                        device=0)))
                
                #本物の画像
                idx = perm[i * batch_size:(i+1)*batch_size]
                x_data=self.X[idx]
                x_data=cuda.to_gpu(x_data, device=0)
                x_data=Variable(x_data)
                
                y2=self.dis(x_data)
                
                #今度は0と判別したい(dis)
                loss_dis += F.softmax_cross_entropy(
                        y2, Variable(cuda.to_gpu(
                                np.ones(batch_size, dtype=np.int32),
                                device=0)))
                self.gen.cleargrads()
                loss_gen.backward()
                o_gen.update()
                
                self.dis.cleargrads()
                loss_dis.backward()
                o_dis.update()
                
                sum_loss_of_dis += loss_dis.data
                sum_loss_of_gen += loss_gen.data
                
            print('epoch:', epoch, 'sum_loss_of_dis:', sum_loss_of_dis,\
                  'sum_loss_of_gen:', sum_loss_of_gen)
                
            self.loss.append([sum_loss_of_gen, sum_loss_of_dis])
            
            if epoch in [1,10,50,100,200,300,400,500,600,700,800,900,1000]:
                if plotting:
                    plt.figure(figsize=(12, 12))
                    n_row=5
                    s=n_row**2
                    z=Variable(cuda.to_gpu(np.random.uniform(-1, 1, 100*s).reshape(-1, 100)\
                                           .astype(np.float32), device=0))
                    x=self.gen(z)
                    y=self.dis(x)
                    y=F.softmax(y).data
                    x=x.data.reshape(-1, 28, 28)
                    x=cuda.to_cpu(x)
                    for i, xx in enumerate(x):
                        plt.subplot(n_row, n_row, i+1)
                        plt.imshow(xx, interpolation='nearest', cmap='gray')
                        plt.axis('off')
                        plt.title('true Prob')
                    plt.tight_layout()
                    st='epoch'+str(epoch)+'.png'
                    plt.savefig(st, dip=100)
                    plt.close('all')
        print(self.loss)

            
            
            
            
            
            
            
            
            
            
        
        
        






