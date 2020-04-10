# -*- coding: utf-8 -*-
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

#加载数据，避免中文乱码问题
data_ori = pd.read_csv('./heros7.csv', encoding='bb18030')
features = [u'最大生命',u'生命成长',u'初始生命',u'最大法力', u'法力成长',u'初始法力',u''最高物攻',u'物攻成长',u'初始物攻',u'最大物防',u'物防成长',u'初始物防', u'最大每5秒回血', u'每5秒回血成长', u'初始每5秒回血', u'最大每5秒回蓝', u'每5秒回蓝成长', u'初始每5秒回蓝', u'最大攻速', u'攻击范围']
data = data_ori[feature_remian]
data[u'最大攻速'] = data['最大攻速'].apply(lambda x:float(x.strip('%'))/100)
data[u'攻击范围'] = data['攻击范围'].map({'远程':1,'近战':0})
#采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
ss = StandardScaler()
data = ss.fit_transform(data)
# 构造GMM聚类
gmm = GaussianMinture(n_components=30,covariance_type='full')
gmm.fit(data)
#训练数据
prediction = gmm.predict(data)
print(prediction)
#将分类结果输出到csv文件中
data_ori.insert(0, '分组', prediction)
data_ori.to_csv('./hero_out.csv', index=False, sep=',')

#聚类和分类不一样，聚类是无监督的学习方式，也就是我们没有实际的结果可以进行比对，
#所以聚类的结果评估不像分类准确率一样直观，
#可以采用 Calinski-Harabaz 指标
from sklearn.metrics import calinski_harabaz_score
print(calinski_harabaz_score(data, prediction))