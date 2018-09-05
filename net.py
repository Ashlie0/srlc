from chainer import Chain
import chainer.functions as F
import chainer.links as L

class srlc(Chain):
    def __init__(self):
        super(srlc, self).__init__(
            #パラメータを含む関数の宣言
            cn1 = L.Convolution2D(1, 4, 2),#(45,45)→(4,44,44)
            cn2 = L.Convolution2D(4, 8, 1),#(4,22,22)→(4,22,22)
            )
    def __call__(self, x, t):
        #損失関数
        return F.mean_squared_error(self.fwd(x),t)
    def fwd(self, x):
        h = F.relu(self.cn1(x))
        h = F.max_pooling_2d(h, 2)#(4,44,44)→(4,22,22)
        h = F.relu(self.cn2(h))
        h = F.max_pooling_2d(h, 2)#(16,22,22)→(16,11,11)
        return h
