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

model=srlc()
gpu_device=0
cuda.get_device(gpu_device).use()
model.to_gpu(gpu_device)

x=loada()
x=np.array(x, dtype=np.float32)
x/=255
X=cuda.to_gpu(x, device=0)


trainer = Trainer(model)

trainer.fit(X,epochs=1000)

df_loss=pd.DataFrame(trainer.loss)
df_loss.to_csv('loss.csv')











