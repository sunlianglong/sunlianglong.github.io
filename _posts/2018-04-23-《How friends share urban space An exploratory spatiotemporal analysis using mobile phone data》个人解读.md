---
layout:     post
title:      《How friends share urban space An exploratory spatiotemporal analysis using mobile phone data》 个人解读
subtitle:   Paper 解读
date:       2018-04-23
author:     sunlianglong
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - Paper
---

## 城市空间分享的探索性时空分析
　　前段时间有意识的去搜集关于学术可视化方面的知识，注意到了[麻省理工学院可感知城市实验室](http://senseable.mit.edu/ "麻省理工学院可感知城市实验室")的官网，引起了我很大的兴趣，随着一段时间的了解和关注，它成了我个人在可视化领域最喜欢的一个实验室，这个实验室通过可视化设计和科学分析实现城市的想象力和社会的变革创新。实验室主页上展示了从2004——2018年的实验室项目，几乎每一项都有实际城市数据的验证，或者是基于某个发达城市的真实数据集的研究。我们可以在每个项目中找到其研究报告和新闻稿，有的是一些已经发表的论文。这其中有很多最新的研究成果和创新知识，值得我们去学习，去探讨。

　　在这个实验室的开放资源中，我研读的第一篇论文是关于人们如何分享城市空间的探索性时空分析：《How friends share urban space: An exploratory spatiotemporal analysis using mobile phone data》，我尝试理解其新颖的算法，并提出自己的思考。
### 介绍
　　随着全球城市化加速，城市对人与人之间的社交互动变得越来越重要。文章介绍了如何建立一个城市空间角色的架构，根据新加坡的一份手机呼叫详细记录数据集来探索城市布局跟人类社交互动之间的关系，了解社交网络如何与地理空间中的人类活动相联，对城市设计和经济具有重大意义发展和社会福祉。文中提出了两个比较新颖且很有说服力的指标，那就是“bonding” capabilities 和 “bridging” capabilities。

　　作者先去了解了大量的相关论文，得出以下结论，或许会在之后的工作中用到：

- 两个人仍然存在友谊的概率和跟两人的地理距离之间遵循衰减函数。地理位置在社交网络结构方面发挥着重要作用。
- 比利时城市间电信网的强度可以用一个标度指数α= 2的重力模型来模拟。标度指数的变化表明物理空间对社交网络结构的影响是相互交织的，关于社会关系如何与人类流动模式联系起来的问题需要得到更好的解决。
- 在社交空间中彼此接近的人更有可能在地理空间中具有协同定位模式。社会接近度和流动相似度之间具有相关性。

　　Friendly Cities is a project that demonstrates how social network structure and mobility patterns extracted from mobile phone data can help us better understand the social roles of urban spaces.

### 贡献
　　定量研究了城市空间的角色和特征对于促进社会参与的作用。

### 工作
　　引入了一个空间联合定位度量方法，来量化给定的一对手机用户在同一时间出现在同一位置的可能性。通过从手机数据集中提取城市规模的社会网络，对网络中的**好友对**和**随机用户对**进行了应用。从而定义两个指标：朋友间的“Bonding capability”，陌生人间的“Bridging capability”。
<center>
<img src=" http://myblog-1253290602.file.myqcloud.com/Paper/paper05.png" width = "600" height = "270"/>
</center>
　　这个最终的指标是怎么定义的呢？首先将新加坡分成若干个规划区域，计算出每个规划区域中的手机用户总数与普查记录的人口分布之间的皮尔逊相关系数为0.98，说明可以用这儿数据来参与建模。

　　文中对新加坡：每隔500m建立一个网络单元格，每个网格单元作为一个手机塔的覆盖范围集合。这种方法的一个优点是, 当测量两个手机用户的共用位置模式时, 如果他们的手机信号出现在同一手机塔, 那么意味着他们将被分配到相同的网格单元格。

　　如何在给定的一对手机用户之间测量共同定位模式？用户的位置仅在他们进行电话通话/短信通信时可用, 不适合直接测量两个个体相互交流的次数或持续时间。因此，文中采用了**空间共位率**的概念，描述了一定的时间内两个个体在空间上共同定位的概率。
<center>
<img src=" http://myblog-1253290602.file.myqcloud.com/Paper/paper04.png" width = "600" height = "400"/>
</center>
　　通过定义Bonding capability 和 Bridging capability，将两者进行对比分析。应用层次聚类算法对某个地域的Bonding capability的时间特征进行了研究。将聚类结果与各种类型的兴趣点 (POIs) 相关联, 以揭示这些地方的语义与它们的Bonding capability之间的关系。
### 可视化方面的展示
<center>
<img src=" http://myblog-1253290602.file.myqcloud.com/Paper/paper06.png" width = "500" height = "400"/>
</center>
### 总结
　　文中提到了将模型结果与兴趣点（POIs）相关联，进行对比分析，让我想到了语义在现在的研究中的一种上涨趋势，各个领域结合语义的研究，逐渐成为了更多研究者想去攻克的方向，这也让我想到了李飞飞教授最近的一篇论文，有时间要去好好研读一下，了解下当前人工智能和学术研究的趋势。









