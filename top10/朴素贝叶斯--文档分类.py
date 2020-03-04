##-----*****朴素贝叶斯--文档分类****------
#!/usr/bin/env python
# _*_ coding:utf8 _*_

import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer #计算单词TF-IDF向量的值，底数是e不是10
from sklearn.naive_bayes import MultinomialNB #多项式
from sklearn import metrics # 准确率

LABEL_MAP = {‘体育’:0, '女性':1, '文学':2, '校园':3} #？？？
# 加载停用词
with open('./..../stopword.txt','rb') as f:
    STOP_WORDS = [line.strip() for line in f.readlines()]
    
def load_data(base_path):
    documents = []
    labels = []
    
    for root, dirs, files in os.walk(base_path):#循环所有文件并进行分词打标
        for file in files:
            label = root.split('\\')[-1] #因为windows上路径符号自动转成\了，所以要转义下
            labels.append(label)
            filename = os.path.join(root, file)
            with open(filename, 'rb') as f: # 因为字符集问题直接用二进制方式读取
                content = f.read()
                word_list = list(jieba.cut(content))
                words = [wl for wl in word_list]
                documents.append('.join(words)')
    return documents, labels
    
def train_fun(td, tl, testd, testl):
    """
    构造模型并计算测试集准确率，字数限制变量名简写
    :param td:训练集数据
    :param tl:训练集标签
    :param testd:测试集数据
    :param testl:测试集标签
    :return:测试集准确率
    """
    
    #计算矩阵
    tt = TfidfVectorizer(stop_words=STOP_WORDS, max_df=0.5)
    tf = tt.fit_transform(td) #使用 fit_transform 方法进行拟合，得到 TF-IDF 特征空间 features
    #训练模型
    clf = MultinomialNB(alpha=0.001).fit(tf, tl)  
    #当 alpha=1 时，使用的是 Laplace 平滑，加 1；当 0<alpha<1,使用的是 Lidstone 平滑。对于 Lidstone 平滑来说，alpha 越小，迭代次数越多，精度越高
    #模型预测
    test_tf = TfidfVectorizer(stop_words=STOP_WORDS, max_df=0.5, vocabulary=tt.vocabulary_)
    test_features = test.tf.fit_transform(testd)
    predicted_labels = clf.perdict(test_features)
    #获取结果
    x = metrics.accuracy+score(testl, predicted_labels)
    return x

# text classification与代码同目录下
train_documents, train_labels = load_data('./text classification/train')
test_documents, test_labels = load_data('./text classification/test')
x = train_fun(train_documents, train_labels, test_documents, test_labels)
print(x)