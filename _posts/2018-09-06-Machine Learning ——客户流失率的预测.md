---
layout:     post
title:      Machine Learning ——客户流失率的预测
subtitle:   Machine Learning ——客户流失率的预测
date:       2018-09-06
author:     sunlianglong
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - 机器学习
    - python
---

　　最近在自己本地的jupyter notebook中看到一个不知出处的笔记，很久没有学习算法和写代码了，所以回顾一下，发现这篇笔记虽然例子很简单，但是内容还算深刻，就整理了一下，虽然不知出处，但是在笔记开头作者有致谢，那我也在这里致谢一下，以示尊敬：Credits: Forked from [growth-workshop](https://github.com/aprial/growth-workshop) by [aprial](https://github.com/aprial), as featured on the [yhat blog](http://blog.yhathq.com/posts/predicting-customer-churn-with-sklearn.html)

　　补充：已找到笔记的出处：[github：deep-learning-with-python-notebooks](https://github.com/fchollet/deep-learning-with-python-notebooks "github：deep-learning-with-python-notebooks")

　　“流失率”是一个商业术语，用来描述客户离开或停止支付产品或服务的费率。这是许多商业的关键因素，因为通常情况下，获得新客户比保留现有客户要昂贵得多（在某些情况下，成本高出5到20倍）。

　　因此，理解是什么让顾客能够持续参与其中是很有价值的，因为这是制定`保留策略`的合理基础。许多公司越来越关注开发更好的`churn-detection techniques`，这也导致了许多人希望通过`data mining`和`machine learning`来寻找新的创造性方法。

　　对于具有订购模式（如手机，有线电视或商家信用卡）的企业来说，预测客户流失尤为重要。但建模流失在许多领域都有广泛的应用。例如，赌场已经使用预测模型来预测理想的房间条件，以便将顾客留在二十一点牌桌上以及何时奖励前排座位的不幸赌徒。同样，航空公司可能会向抱怨客户提供一流的升级服务。

　　那么公司采用什么样的策略来防止流失呢？一般来说防止顾客的流失需要一些重要的资源，比如一些专业的运营团队会采取为顾客提供更多更好的服务，来呼吁这些风险客户不要流失。

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/machine-learning/machine-learning-9.png" width = "600" height = "600"/>
</center>
　　本文的其余部分将探讨一个简单的案例研究，以展示如何使用Python及其科学库来预测客户的流失。

### The Dataset

　　我们使用的数据集是一个长期的电信客户数据集。数据很简单。每行代表客户。每列包含客户属性，例如电话号码，在一天中的不同时间的呼叫分钟，服务产生的费用，终身帐户的持续时间以及客户是否流失。

　　首先导入需要的各种python库，导入数据并简单展示。


```python
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
%matplotlib inline

```

```python
churn_df = pd.read_csv('../data/churn.csv')
col_names = churn_df.columns.tolist()

print "Column names:"
print col_names

to_show = col_names[:6] + col_names[-6:]

print "\nSample data:"
churn_df[to_show].head(5)
      
```
```ptyhon
    Column names:
    ['State', 'Account Length', 'Area Code', 'Phone', "Int'l Plan", 'VMail Plan', 'VMail Message', 'Day Mins', 'Day Calls', 'Day Charge', 'Eve Mins', 'Eve Calls', 'Eve Charge', 'Night Mins', 'Night Calls', 'Night Charge', 'Intl Mins', 'Intl Calls', 'Intl Charge', 'CustServ Calls', 'Churn?']
```
    Sample data:

<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Account Length</th>
      <th>Area Code</th>
      <th>Phone</th>
      <th>Int'l Plan</th>
      <th>VMail Plan</th>
      <th>Night Charge</th>
      <th>Intl Mins</th>
      <th>Intl Calls</th>
      <th>Intl Charge</th>
      <th>CustServ Calls</th>
      <th>Churn?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>128</td>
      <td>415</td>
      <td>382-4657</td>
      <td>no</td>
      <td>yes</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OH</td>
      <td>107</td>
      <td>415</td>
      <td>371-7191</td>
      <td>no</td>
      <td>yes</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NJ</td>
      <td>137</td>
      <td>415</td>
      <td>358-1921</td>
      <td>no</td>
      <td>no</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>84</td>
      <td>408</td>
      <td>375-9999</td>
      <td>yes</td>
      <td>no</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OK</td>
      <td>75</td>
      <td>415</td>
      <td>330-6626</td>
      <td>yes</td>
      <td>no</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>False.</td>
    </tr>
  </tbody>
</table>
</div>


　　将相关字符串转化成布尔值，删掉没用的属性

```python
# Isolate target data
churn_result = churn_df['Churn?']
y = np.where(churn_result == 'True.',1,0)
y
```

    array([0, 0, 0, ..., 0, 0, 0])

```python
# We don't need these columns
to_drop = ['State','Area Code','Phone','Churn?']
churn_feat_space = churn_df.drop(to_drop,axis=1)
```

```python
# 'yes'/'no' has to be converted to boolean values
# NumPy converts these from boolean to 1. and 0. later
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'
```

```python
# Pull out features for future use
features = churn_feat_space.columns
print features
```
```python
    Index([u'Account Length', u'Int'l Plan', u'VMail Plan', u'VMail Message',
           u'Day Mins', u'Day Calls', u'Day Charge', u'Eve Mins', u'Eve Calls',
           u'Eve Charge', u'Night Mins', u'Night Calls', u'Night Charge',
           u'Intl Mins', u'Intl Calls', u'Intl Charge', u'CustServ Calls'],
          dtype='object')
```
```python
X = churn_feat_space.as_matrix().astype(np.float)
X
```
```python
    array([[ 128.  ,    0.  ,    1.  , ...,    3.  ,    2.7 ,    1.  ],
           [ 107.  ,    0.  ,    1.  , ...,    3.  ,    3.7 ,    1.  ],
           [ 137.  ,    0.  ,    0.  , ...,    5.  ,    3.29,    0.  ],
           ..., 
           [  28.  ,    0.  ,    0.  , ...,    6.  ,    3.81,    2.  ],
           [ 184.  ,    1.  ,    0.  , ...,   10.  ,    1.35,    2.  ],
           [  74.  ,    0.  ,    1.  , ...,    4.  ,    3.7 ,    0.  ]])
```
　　StandardScaler通过将每个特征标准化为大约1.0到-1.0的范围来实现归一化。

```python
X = churn_feat_space.as_matrix().astype(np.float)

# This is important
scaler = StandardScaler()
X = scaler.fit_transform(X)
print "Feature space holds %d observations and %d features" % X.shape
print "Unique target labels:", np.unique(y)
```

    Feature space holds 3333 observations and 17 features
    Unique target labels: [0 1]

### 模型的好坏判断
　　使用交叉验证法来避免过拟合，同时为每个观测数据集生成预测。

```python
from sklearn.cross_validation import KFold

def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=3,shuffle=True) #生成交叉数据集
    y_pred = y.copy() 
    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred
```

　　让我们比较三个相当独特的算法：支持向量机，随机森林和K近邻算法。

```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import average_precision_score

def accuracy(y_true,y_pred):
    # NumPy interpretes True and False as 1. and 0.
    return np.mean(y_true == y_pred)

print "Logistic Regression:"
print "%.3f" % accuracy(y, run_cv(X,y,LR))
print "Gradient Boosting Classifier"
print "%.3f" % accuracy(y, run_cv(X,y,GBC))
print "Support vector machines:"
print "%.3f" % accuracy(y, run_cv(X,y,SVC))
print "Random forest:"
print "%.3f" % accuracy(y, run_cv(X,y,RF))
print "K-nearest-neighbors:"
print "%.3f" % accuracy(y, run_cv(X,y,KNN))
```
```python
    Logistic Regression:
    0.861
    Gradient Boosting Classifier
    0.951
    Support vector machines:
    0.920
    Random forest:
    0.944
    K-nearest-neighbors:
    0.892
```
### 查准率和查全率

　　查准率亦称“准确率”(Precision)，查全率亦称“召回率”(Recall)。具体关于机器学习性能度量的相关方法详解，可以参见《西瓜书》P29。

#### 混淆矩阵

　　我们使用内置的scikit-learn函数来构建混淆矩阵。混淆矩阵是一种可视化分类器预测的方式，用一个表格来显示特定类的预测分布。 x轴表示每个观察的真实类别，而y轴对应于模型预测的类别。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/machine-learning/machine-learning-10.png" width = "350" height = "350"/>
</center>

　　查准率P与查全率R分别定义为：
<center><font color=grey>$$P = \frac{ TP}{TP + FP}$$</font></center>
<center><font color=grey>$$R = \frac{ TP}{TP + FN}$$</font></center>

```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def draw_confusion_matrices(confusion_matricies,class_names):
    class_names = class_names.tolist()
    for cm in confusion_matrices:
        classifier, cm = cm[0], cm[1]
        print cm

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix for %s' % classifier)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

y = np.array(y)
class_names = np.unique(y)

confusion_matrices = [
    ( "Support Vector Machines", confusion_matrix(y,run_cv(X,y,SVC)) ),
    ( "Random Forest", confusion_matrix(y,run_cv(X,y,RF)) ),
    ( "K-Nearest-Neighbors", confusion_matrix(y,run_cv(X,y,KNN)) ),
    ( "Gradient Boosting Classifier", confusion_matrix(y,run_cv(X,y,GBC)) ),
    ( "Logisitic Regression", confusion_matrix(y,run_cv(X,y,LR)) )
]

# Pyplot code not included to reduce clutter
# from churn_display import draw_confusion_matrices
%matplotlib inline

draw_confusion_matrices(confusion_matrices,class_names)
```

    [[2815   35]
     [ 249  234]]

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/machine-learning/machine-learning-11.png" width = "300" height = "300"/>
</center>

    [[2824   26]
     [ 159  324]]

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/machine-learning/machine-learning-12.png" width = "300" height = "300"/>
</center>

    [[2797   53]
     [ 315  168]]

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/machine-learning/machine-learning-13.png" width = "300" height = "300"/>
</center>

    [[2815   35]
     [ 130  353]]

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/machine-learning/machine-learning-14.png" width = "300" height = "300"/>
</center>

    [[2767   83]
     [ 387   96]]

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/machine-learning/machine-learning-15.png" width = "300" height = "300"/>
</center>

　　通过上图可以看出对于查准率查全率，随机森林算法都是其中最好的算法，我们也可以根据自己对于问题的需要选择P、R不同的算法，根据需求进行取舍。

### ROC 绘图 & AUC

　　ROC是另外一中性能度量的方法，具体也可以参加西瓜书，对于原理方面不想多说，因为书中已经解释得非常清楚。

```python
from sklearn.metrics import roc_curve, auc
from scipy import interp

def plot_roc(X, y, clf_class, **kwargs):
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y),2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for i, (train_index, test_index) in enumerate(kf):
        #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)
        #组合为一个索引序列，同时列出数据和数据下标
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes,输出实例在每个类中的概率
        y_prob[test_index] = clf.predict_proba(X_test)
        # 按照步骤生成绘图所需的点
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])

        mean_tpr += interp(mean_fpr, fpr, tpr) # 一维线性插值函数

        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)# 计算曲线下面积
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= len(kf)#上述计算是取了5次迭代每条曲线的平均值的和，需/5 
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

print "Support vector machines:"
plot_roc(X,y,SVC,probability=True)

print "Random forests:"
plot_roc(X,y,RF,n_estimators=18)

print "K-nearest-neighbors:"
plot_roc(X,y,KNN)

print "Gradient Boosting Classifier:"
plot_roc(X,y,GBC)
```

    Support vector machines:

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/machine-learning/machine-learning-16.png" width = "300" height = "300"/>
</center>

    Random forests:

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/machine-learning/machine-learning-17.png" width = "300" height = "300"/>
</center>

    K-nearest-neighbors:

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/machine-learning/machine-learning-18.png" width = "300" height = "300"/>
</center>

    Gradient Boosting Classifier:

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/machine-learning/machine-learning-19.png" width = "300" height = "300"/>
</center>

### 特征重要性

　　每个特征的重要性，也是帮助决策者决策的重要因素，在这里，我们将其按照重要性排序，重要性的赋值准测归功于决策树，前面的博客中已经提到，这里不再赘述。

```python
train_index,test_index = train_test_split(churn_df.index)

forest = RF()
forest_fit = forest.fit(X[train_index], y[train_index])
forest_predictions = forest_fit.predict(X[test_index])

importances = forest_fit.feature_importances_[:10]

std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0) #计算标准偏差
indices = np.argsort(importances)[::-1]#排序，返回坐标


# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))

# Plot the feature importances of the forest
#import pylab as pl
plt.figure()
plt.title("Feature importances")
plt.bar(range(10), importances[indices], yerr=std[indices], color="r", align="center")
plt.xticks(range(10), indices)
plt.xlim([-1, 10])
plt.show()
```
```pythpn
    [4 6 7 1 9 3 5 0 8 2]
    Feature ranking:
    1. Account Length (0.157386)
    2. Int'l Plan (0.104888)
    3. VMail Plan (0.079780)
    4. VMail Message (0.075888)
    5. Day Mins (0.054804)
    6. Day Calls (0.044179)
    7. Day Charge (0.033902)
    8. Eve Mins (0.031440)
    9. Eve Calls (0.031259)
    10. Eve Charge (0.025442)
```
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/machine-learning/machine-learning-20.png" width = "350" height = "350"/>
</center>

### 输出概率
　　一般来说，输出概率要比输出决策结果更有说服力，在sklearn中，我们只需要更改一丢丢代码，就可以实现概率输出：`predict_proba`函数。

```python
def run_prob_cv(X, y, clf_class, roc=False, **kwargs):
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y),2))
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
    return y_prob
```

### 概率统计
　　我们可以看到，随机森林算法预测89个人的流失概率为0.9，实际上该群体的流速为0.97。

```python
import warnings
warnings.filterwarnings('ignore')

# Use 10 estimators so predictions are all multiples of 0.1
pred_prob = run_prob_cv(X, y, RF, n_estimators=10)
pred_churn = pred_prob[:,1]
is_churn = y == 1

# Number of times a predicted probability is assigned to an observation
counts = pd.value_counts(pred_churn)
counts[:]
```
```python
    0.0    1798
    0.1     668
    0.2     272
    0.3     112
    0.9      85
    0.7      72
    0.6      71
    0.5      71
    0.4      67
    0.8      62
    1.0      55
    dtype: int64
```
```python
from collections import defaultdict
true_prob = defaultdict(float)

# calculate true probabilities
for prob in counts.index:
    true_prob[prob] = np.mean(is_churn[pred_churn == prob])
true_prob = pd.Series(true_prob)

# pandas-fu
counts = pd.concat([counts,true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']
counts
```

<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred_prob</th>
      <th>count</th>
      <th>true_prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0 </th>
      <td> 0.0</td>
      <td> 1796</td>
      <td> 0.027840</td>
    </tr>
    <tr>
      <th>1 </th>
      <td> 0.1</td>
      <td>  681</td>
      <td> 0.020558</td>
    </tr>
    <tr>
      <th>2 </th>
      <td> 0.2</td>
      <td>  258</td>
      <td> 0.081395</td>
    </tr>
    <tr>
      <th>3 </th>
      <td> 0.3</td>
      <td>  112</td>
      <td> 0.089286</td>
    </tr>
    <tr>
      <th>4 </th>
      <td> 0.4</td>
      <td>   74</td>
      <td> 0.270270</td>
    </tr>
    <tr>
      <th>5 </th>
      <td> 0.5</td>
      <td>   53</td>
      <td> 0.622642</td>
    </tr>
    <tr>
      <th>6 </th>
      <td> 0.6</td>
      <td>   50</td>
      <td> 0.800000</td>
    </tr>
    <tr>
      <th>7 </th>
      <td> 0.7</td>
      <td>   79</td>
      <td> 0.898734</td>
    </tr>
    <tr>
      <th>8 </th>
      <td> 0.8</td>
      <td>   94</td>
      <td> 0.957447</td>
    </tr>
    <tr>
      <th>9 </th>
      <td> 0.9</td>
      <td>   89</td>
      <td> 0.977528</td>
    </tr>
    <tr>
      <th>10</th>
      <td> 1.0</td>
      <td>   47</td>
      <td> 1.000000</td>
    </tr>
  </tbody>
</table>
</div>



　　We can see that random forests predicted that 75 individuals would have a 0.9 proability of churn and in actuality that group had a ~0.97 rate.

### Calibration and Descrimination

　　使用上面的DataFrame，我们可以绘制一个非常简单的图形来帮助可视化概率测量。x轴表示随机森林分配给一组个体的流失概率。 y轴是该组内的实际流失率。距离红线越远，预测的越不精准。

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/machine-learning/machine-learning-22.png" width = "500" height = "500"/>
</center>



```python
from churn_measurements import calibration, discrimination
from sklearn.metrics import roc_curve, auc
from scipy import interp
from __future__ import division 
from operator import idiv

def print_measurements(pred_prob):
    churn_prob, is_churn = pred_prob[:,1], y == 1
    print "  %-20s %.4f" % ("Calibration Error", calibration(churn_prob, is_churn))
    print "  %-20s %.4f" % ("Discrimination", discrimination(churn_prob,is_churn))

    print "Note -- Lower calibration is better, higher discrimination is better"

print "Support vector machines:"
print_measurements(run_prob_cv(X,y,SVC,probability=True))

print "Random forests:"
print_measurements(run_prob_cv(X,y,RF,n_estimators=18))

print "K-nearest-neighbors:"
print_measurements(run_prob_cv(X,y,KNN))

print "Gradient Boosting Classifier:"
print_measurements(run_prob_cv(X,y,GBC))

print "Random Forest:"
print_measurements(run_prob_cv(X,y,RF))
```
```python
    Support vector machines:
      Calibration Error    0.0016
      Discrimination       0.0678
    Note -- Lower calibration is better, higher discrimination is better
    Random forests:
      Calibration Error    0.0072
      Discrimination       0.0845
    Note -- Lower calibration is better, higher discrimination is better
    K-nearest-neighbors:
      Calibration Error    0.0023
      Discrimination       0.0443
    Note -- Lower calibration is better, higher discrimination is better
    Gradient Boosting Classifier:
      Calibration Error    0.0017
      Discrimination       0.0859
    Note -- Lower calibration is better, higher discrimination is better
    Random Forest:
      Calibration Error    0.0062
      Discrimination       0.0782
    Note -- Lower calibration is better, higher discrimination is better
```
　　与之前的分类比较不同，随机森林在这里并不是明显的领先者。虽然它擅长区分高概率和低概率流失事件，但它难以为这些事件分配准确的概率估计。例如，随机森林预测具有30％流失率的群体实际上具有14％的真实流失率。

