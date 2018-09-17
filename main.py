# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 23:11:35 2018

@author: ashli
"""

from load import loada
from chainer import cuda
from net import srlc
import numpy as np
from train import Trainer
import pandas as pd
import chainer
import matplotlib.pyplot as plt

model=srlc()
gpu_device=0
cuda.get_device(gpu_device).use()
model.to_gpu(gpu_device)

x=loada()
x=np.array(x, dtype=np.float32)
x/=255
print(x.shape)
X=cuda.to_gpu(x, device=0)


#trainer = Trainer(model)
#trainer.fit(X,epochs=1000)

#df_loss=pd.DataFrame(trainer.loss)
#df_loss.to_csv('loss.csv')

#chainer.serializers.save_npz("srlc100000_sig.npz", model)
#chainer.serializers.save_npz("srlc100000_tanh.npz", model)
chainer.serializers.load_npz("srlc100000_sig.npz", model)
#chainer.serializers.load_npz("srlc100000_tanh.npz", model)

res1=[]
res2=[]

for i in range(120*2):
    if i==0:
        z1=model.fwd1(X[i].reshape(1,1,45,45))
        z2=model.fwd2(X[i].reshape(1,1,45,45))
        z1=z1.data
        z2=z2.data
        z1=cuda.to_cpu(z1)
        z2=cuda.to_cpu(z2)
        z1-=np.ones((1,4,22,22))/2
        z2-=np.ones((1,16,11,11))/2
        t1=np.sum(z1)
        t2=np.sum(z2)
        #zt1=z1*0.02
        #zt2=z2*0.02
        #zt1=np.ones((1,4,22,22))*t1/4/22/22
        #zt2=np.ones((1,16,11,1))*t2/16/11/11
    else:
        i%=120
        z1=model.fwd1(X[i].reshape(1,1,45,45))
        z2=model.fwd2(X[i].reshape(1,1,45,45))
        z1=z1.data
        z2=z2.data
        z1=cuda.to_cpu(z1)
        z2=cuda.to_cpu(z2)
        z1-=np.ones((1,4,22,22))/2
        z2-=np.ones((1,16,11,11))/2
        #if i%20==0:
            #z1-=zt1
            #z2-=zt2
            #zt1-=zt1
            #zt2-=zt2
        t1=np.sum(z1)
        t2=np.sum(z2)
        res1.append(t1)
        res2.append(t2)
        #zt1+=z1*0.02
        #zt2+=z2*0.02
        #zt1=np.ones((1,4,22,22))*t1/4/22/22
        #zt2=np.ones((1,16,11,1))*t2/16/11/11

x=np.arange(1,120*2,1)
res1=np.array(res1)
res2=np.array(res2)
plt.plot(x,res1)
plt.show()
plt.close()
plt.plot(x,res2)
plt.show()
plt.close()












