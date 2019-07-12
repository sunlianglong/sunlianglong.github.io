---
layout:     post
title:      Matplotlib使用与矢量图生成
subtitle:   Matplotlib使用与矢量图生成
date:       2018-04-26
author:     sunlianglong
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - Matplotlib
    - python
    -矢量图 
---

　　Matplotlib 是一个非常强大的 Python 画图工具，它可能是 Python 2D-绘图领域使用最广泛的套件。它能让使用者很轻松地将数据图形化，并且提供多样化的输出格式，Matplotlib 提供了一整套和Matlab相似的API，对于习惯使用Python的人来说非常便利，[相关API见此](https://matplotlib.org/ "相关API见此")，个人感觉这个官方网址里面的内容非常的全面，学习matplotlib完全足够。

### 实验结果折线图
#### 相关包

　　Matplotlib的pyplot子库是命令行式函数的集合，是一个很常用的子库。每一个函数都可以图像进行修改，比如创建图形，在图像上创建画图区域，在画图区域上画线，给图像添加标签等。首先导入相关的包。
```python
import matplotlib.pyplot as plt
import numpy as np
```
#### 相关数据
```python
labels = np.loadtxt(open(URL, "rb"), delimiter=",", skiprows=0)
x = np.linspace(1, 10, 10)
k_mnist = labels[:, 1]
k_office = labels[:, 2]
```
　　pycharm下，可以CTRL+鼠标左键查看该函数的详细信息，也可以去查看函数的API文档，来了解和使用该函数的详细参数。我导入本地的数据如下，`labels[0]`是横坐标，`labels[1]`是y1的数据，`labels[2]`是y2的数据：
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/Python/para_1.png" width = "300" height = "300"/>
</center>

#### 绘制函数调用
```python
plt.ylim(50, 100) # 设置纵坐标范围
plt.xlabel('K value ', fontsize=20) # 坐标标签和字体设置
plt.ylabel("Accuracy(%)", fontsize=20)
# plt.title("Average Accuracy with The iteration Number", size='large') # 图像标题设置
group_labels = ['e-15', 'e-13', 'e-11', 'e-9', 'e-7', 'e-5', 'e-3', 'e-1', 'e+1', 'e+3'] # 横坐标数值映射

plt.xticks(x, group_labels, rotation=0, fontsize=14) # 坐标映射和字体等设置
plt.yticks(fontsize=14)
plt.grid() # 显示网格

plt.plot(x, k_mnist*100, '-g', marker='o', label="Mnist Datasets", linewidth=4, markersize=9)  # dashdot
plt.plot(x, k_office*100, '--c', marker='^', label="Office Datasets", linewidth=4, markersize=10)  # dotted
plt.legend()
plt.show()
```
### 效果
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/Python/Figure_1.png" width = "500" height = "400"/>
</center>
　　考虑到要插入图片到Paper，个人感觉不能使用默认的字体等设置，比如线段宽度以及marker的大小，label的大小都要放大，坐标轴的label的大小也要放大，这样在阅读Paper的时候才会更加清楚一目了然，而不用放到很大才能看清楚我们想要表达的是什么。

### 改进

　　感觉默认的网格有点丑，那再创建绘图时使用经典的Matplotlib风格，个人感觉会好看一些。

```python
plt.style.use('classic')
```
　　参考其他Paper中的插图，图像中横坐标都是从第二个刻度开始，倒数第二个刻度结束，也就是两端空出来让折现居中显示，那么这该怎么实现呢？google无果，我自己研究出了一种方法。那就是为横坐标设置大于实际区间两个值的坐标区间后，再设置坐标显示的字符串，对应着实际区间。代码如下：

```python
plt.xlim(0, 11)   # 0-11一共12个刻度
x = np.linspace(1, 10, 10)  # 1-10一共10个刻度
group_labels = ['$1^{-15}$', '$1^{-13}$', '$1^{-11}$', '$1^{-9}$', '$1^{-7}$', '$1^{-5}$', '$1^{-3}$', '$1^{-1}$', '$10$','$10^{3}$'] # 横坐标数值映射
plt.xticks(x, group_labels, rotation=0, fontsize=16) # 将10个刻度的区间对应映射成需要显示的字符串（比如1，2，3，4，5，6，7，8，9，10）
```
### 效果
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/Python/Figure_5.png" width = "500" height = "400"/>
</center>

### 保存为矢量图
　　还有一个比较关键的问题是，在插入图片到Paper中时，要做到PDF放大不失真。我去了解了下做到不失真的方法：

- 一个是利用工具或者编程实现增加图片的分辨率，将图片的像素点数量放大到几倍。缺点很明显，图片规格变大。虽然我们不会采取这种方式，但是还是可以了解一下：Photo Zoom Pro 6 安装包[在此](https://pan.baidu.com/s/1xUernropPPTP3Thl45gJzw "在此"),提取密码: ce34。
- 二是利用工具或编程将图片转化为矢量图，刚接触到矢量图的效果后，感觉很完美。这个工具感觉也很好用：[Vector Magic](https://vectormagic.com/ "Vector Magic")，而python有自带可以输出任意格式的函数，实现如下：

```python
figure_fig = plt.gcf()  # 'get current figure'
figure_fig.savefig('figure.eps', format='eps', dpi=1000)
```
#### 效果
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/Python/Figure_3.png" width = "500" height = "500"/>
</center>