##-----****KNN-对手写数字进行识别****------

#_*_ coding:utf-8 _*_
#引用KNN分类器
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="")
#引用KNN分类器
#from sklearn.neighbors import KNeighborsRegressor

#加载数据
digits = load_digits()
data = digits.data
#数据探索
print(data.shape)
#查看第一幅图像
print(digits.images[0])
#查看第一幅图像代表的数字含义
print(digits.target[0])
#将第一幅图像显示出来
plt.gray()
plt.imshow(digits.images[0])
plt.show()

#分割数据
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)
#采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

#在train的时候用到了：train_ss_x = ss.fit_transform(train_x)
#实际上：fit_transform是fit和transform两个函数都执行一次。所以ss是进行了fit拟合的。只有在fit拟合之后，才能进行transform
#在进行test的时候，我们已经在train的时候fit过了，所以直接transform即可

#创建KNN分类器
knn = KNeighborsClassifier()
knn.fit(train_ss_x, train_y)
predict_y = knn.predict(test_ss_x)
print("KNN准确率： %.4lf" % accuracy_score(test_y, predict_y))

#创建SVM分类器
svm = SVC(kernel='rbf', C=1E6, gamma='auto')
svm.fit(train_ss_x, train_y)
predict_y=svm.predict(test_ss_x)
print('SVM准确率: %0.4lf' % accuracy_score(test_y, predict_y))

#采用Min-Max规范化
mm = preprocessing.MinMaxScaler()
train_mm_x = mm.fit_transform(train_x)
test_mm_x = mm.transform(test_x)

#创建Naive Bayes分类器
mnb = MultinomialNB()
mnb.fit(train_mm_x, train_y)
predict_y=mnb.predict(test_mm_x)
print('多项式朴素贝叶斯准确率: %0.4lf' % accuracy_score(test_y, predict_y))

#这里需要注意的是，我们在做多项式朴素贝叶斯分类的时候，传入的数据不能有负数。因为 Z-Score 会将数值规范化为一个标准的正态分布，即均值为 0，方差为 1，
#数值会包含负数。因此我们需要采用 Min-Max 规范化，将数据规范化到 [0,1] 范围内
#创建Naive Bayes分类器

#创建CART决策树分类器
dtc = DecisionTreeClassifier()
dtc.fit(train_x, train_y)
predict_y=mnb.predict(test_x)
print('CART决策树准确率: %0.4lf' % accuracy_score(test_y, predict_y))

#如果数据量很大，比如 MNIST 数据集中的 6 万个训练数据和 1 万个测试数据，那么采用深度学习 +GPU 运算的方式会更适合。
#因为深度学习的特点就是需要大量并行的重复计算，GPU 最擅长的就是做大量的并行计算