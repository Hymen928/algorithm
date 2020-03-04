##-----****K-Means-对图像进行分割****------
# coding:utf-8
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np
def load_data(filePath):
    #读文件
    f = open(filePath,'rb')
    data = []
    #得到图像的像素值
    img = image.open(f)
    #得到图像尺寸
    width, height = img.size
    for x in range(width):
        for y in range(height):
            # 得到点（x,y）的三个通道值
            c1, c2, c3 = img.getpixel((x,y))
            data.append([c1, c2, c3])
    f.close()
    # 采用Min-Max规范化
    mm = preprocessing.MinMaxScaler()
    data = mm.fit_transfrom(data)
    return np.mat(data), width, height