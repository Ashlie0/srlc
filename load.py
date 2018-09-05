import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def loada(a=0):
    img = os.listdir("./dataa/")
    img_in = []
    #開いたフォルダの中にある画像の名前をイテレーションで全て抽出する
    for i in img:
        #listdirで開くと画像の名前とは関係のないThumbs.dbが抽出されるので無視する
        if i == "Thumbs.db":
            continue
        #imagesの中にある全ての画像の配列を開いていく
        imagea = np.array(Image.open("./dataa/"+i))
        #reshapeを使って開かれた配列を1次元配列に変換する
        imagea_resize = imagea.reshape(imagea.size)
        #バッチリストに追加していく
        img_in.append(imagea_resize)


    x = np.array(img_in).reshape(120, 45, 45).astype(np.float32)
    if a != 0:
        plt.imshow(x[a])
        plt.gray()
        plt.show()

    return x
def loadb(b=0):
    img = os.listdir("./datab/")
    img_in = []
    #開いたフォルダの中にある画像の名前をイテレーションで全て抽出する
    for i in img:
        #listdirで開くと画像の名前とは関係のないThumbs.dbが抽出されるので無視する
        if i == "Thumbs.db":
            continue
        #imagesの中にある全ての画像の配列を開いていく
        imagea = np.array(Image.open("./datab/"+i))
        #reshapeを使って開かれた配列を1次元配列に変換する
        imagea_resize = imagea.reshape(imagea.size)
        #バッチリストに追加していく
        img_in.append(imagea_resize)


    x = np.array(img_in).reshape(120, 45, 45).astype(np.float32)
    if a != 0:
        plt.imshow(x[a])
        plt.gray()
        plt.show()

    return x
