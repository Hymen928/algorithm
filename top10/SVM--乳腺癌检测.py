##-----*****SVM--乳腺癌检测****------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm  #SVM 既可以做回归（SVR），也可以做分类器(SVC)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#加载数据并对数据做部分探索
data = pd.read_csv("./data.csv")
pd.set_option('display.max_columns', None)
print(data.columns)
print(data.head(5))
print(data.describe())

#将特征字段分为3组
features_mean= list(data.columns[2:12])
features_se= list(data.columns[12:22])
features_eorst= list(data.columns[22:32])
#数据清洗
#ID列没有用，删除
data.drop("id",axis=1,implace=True)
#将B良性替换为0，M恶性替换为1
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

#特征字段的筛选
#将肿瘤诊断结果可视化
sns.countplot(data['diagnosis'],label="count")
plt.show()
#用热力图呈现features_mean字段之间的相关性
corr = data[features_mean].corr()
plt.figure(figsize=(14,14))
#annot=True显示每个方格的数据
sns.heatmap(corr, annot=True)
plt.show()

#特征选择
features_remain = ['redius_mean','texture_mean','smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean']

#抽取30%的数据作为测试机，其余作为训练集
train, test = train_test_split(data, test_size=0.33)
#抽取特征选择的数值作为训练和测试数据
train_X = train[features_remain]
train_Y = train['diagnosis']
test_X = test[features_remain]
test_Y = test['diagnosis']
#数据规范化
#采用Z-Score规范化数据，保证每个特征为度的数据均值为0，方差为1
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)

#创建SVM分类器
model =svm.SVC()
#用训练集做训练
model.fit(train_X, train_Y)
#用测试集做预测
prediction=model.predict(test_X)
print('准确率： ',metrics.accuracy_score(prediction,test_Y))