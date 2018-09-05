# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 22:09:22 2018

@author: Ashley
"""


import numpy as np
import chainer
from chainer import cuda, Function, report, training, utils, Variable
from chainer import iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from sklearn import datasets
import matplotlib.pyplot as plt
from PIL import Image
from net import Gan_gen, Gan_dis
from trainer import Trainer
import pandas as pd

gen=Gan_gen(100)
dis=Gan_dis()

gpu_device=0
cuda.get_device(gpu_device).use()

gen.to_gpu(gpu_device)
dis.to_gpu(gpu_device)

data = datasets.fetch_mldata('MNIST original', data_home='.')
xc=data['data']
n_train=xc.shape[0]
xc=np.array(xc, dtype=np.float32)
xc/=255
xc=xc.reshape(n_train, 1, 28, 28)
X = cuda.to_gpu(xc, device=0)

trainer = Trainer(gen, dis)

trainer.fit(X, batch_size=1000, epochs=1000)

df_loss = pd.DataFrame(trainer.loss)
df_loss.to_csv('loss.csv')


