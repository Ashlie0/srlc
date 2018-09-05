from load import loada,loadb
import chainer
import chainer.functions as F
import chainer.links as L
x = loada(1)
y = loadb(1)
train = chainer.datasets.tuple_dataset.TupleDataset(x)


















