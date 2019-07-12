---
layout:     post
title:      物理CPU，CPU核数，逻辑CPU
subtitle:   物理CPU，CPU核数，逻辑CPU
date:       2019-01-25
author:     sunlianglong
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - CPU
    - 操作系统
---

### CPU概念
#### 物理CPU
> 在Info中由physical id识别

物理CPU指的是实际主板上插槽上的CPU个数。physical id 就是每个物理CPU的ID，不重复的 physical id 有几个，物理cpu数量就是几个。

#### CPU核数（物理核）  
> 在Info中由 cpu cores 识别

CPU核数是指单块CPU上面能处理数据的芯片组的数量，如双核、四核等。通常每个CPU下的核数都是固定的，比如你的计算机有两个物理CPU，每个CPU是双核，那么计算机就是四核的。

#### 逻辑CPU（逻辑核）
> 在Info中由 processor 识别

逻辑CPU是指用Intel的超线程技术(HT)将物理核虚拟而成的逻辑处理单元。

在windows系统下面，我们看到有4个cpu记录，其实这是双核CPU使用HT技术虚拟出来的4个逻辑CPU。

![image](http://myblog-1253290602.file.myqcloud.com/YouDaoNote/Others/CPU/CPU%5B1%5D.png)

任务管理器中也能查看自己PC中基本的物理CPU CPU核数以及逻辑CPU 。

![image](http://myblog-1253290602.file.myqcloud.com/YouDaoNote/Others/CPU/CPU%5B2%5D.png)

在linux系统下面的/proc/cpuinfo文件的条目中siblings记录了对应的物理CPU（以该条目中的physical id标识）有多少个逻辑核：

![image](http://myblog-1253290602.file.myqcloud.com/YouDaoNote/Others/CPU/CPU%5B3%5D.png)

总核数 = 物理CPU个数 X 每颗物理CPU的核数 

总逻辑CPU数 = 物理CPU个数 X 每颗物理CPU的核数 X 超线程数

### 查看cpu信息概要
> 摘自 http://www.cnblogs.com/mafeng/p/6558941.html

```
#lscpu
Architecture:          i686                            #架构686
CPU(s):                  2                                #逻辑cpu颗数是2
Thread(s) per core:    1                           #每个核心线程数是1                 
Core(s) per socket:    2                           #每个cpu插槽核数/每颗物理cpu核数是2
CPU socket(s):         1                            #cpu插槽数是1
Vendor ID:             GenuineIntel           #cpu厂商ID是GenuineIntel
CPU family:            6                              #cpu系列是6
Model:                 23                                #型号23
Stepping:              10                              #步进是10
CPU MHz:               800.000                 #cpu主频是800MHz
Virtualization:        VT-x                         #cpu支持的虚拟化技术VT-x
L1d cache:             32K                         #一级缓存32K（google了下，这具体表示表示cpu的L1数据缓存为32k）
L1i cache:             32K                          #一级缓存32K（具体为L1指令缓存为32K）
L2 cache:              3072K                      #二级缓存3072K
```




### 查看cpu信息命令

```sh
# 查看物理CPU个数
cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l
# 查看每个物理CPU中core的个数(即核数)
cat /proc/cpuinfo| grep "cpu cores"| uniq
# 查看逻辑CPU的个数
cat /proc/cpuinfo| grep "processor"| wc -l
```



### CPU部分信息解释

```
vendor id     如果处理器为英特尔处理器，则字符串是 GenuineIntel。
processor     包括这一逻辑处理器的唯一标识符。
physical id   包括每个物理封装的唯一标识符。
core id       保存每个内核的唯一标识符。
siblings      列出了位于相同物理封装中的逻辑处理器的数量。
cpu cores     包含位于相同物理封装中的内核数量。
```

1. 拥有相同 physical id 的所有逻辑处理器共享同一个物理插座，每个 physical id 代表一个唯一的物理封装。

2. Siblings 表示位于这一物理封装上的逻辑处理器的数量，它们可能支持也可能不支持超线程（HT）技术。

3. 每个 core id 均代表一个唯一的处理器内核，所有带有相同 core id 的逻辑处理器均位于同一个处理器内核上。
简单的说：“siblings”指的是一个物理CPU有几个逻辑CPU，”cpu cores“指的是一个物理CPU有几个核。

4. 如果有一个以上逻辑处理器拥有相同的 core id 和 physical id，则说明系统支持超线程（HT）技术。

5. 如果有两个或两个以上的逻辑处理器拥有相同的 physical id，但是 core id不同，则说明这是一个多内核处理器。cpu cores条目也可以表示是否支持多内核。

6. “所有带有相同 core id 的逻辑处理器均位于同一个处理器内核上” 这样的描述感觉是有问题的。我测试了一下，ubuntu的双CPU的系统，每个CPU可以分别命名CPU核0、CPU核1...然后就有四个core id均为0的编号，但是他们分别属于不同的处理器内核。

![image](http://myblog-1253290602.file.myqcloud.com/YouDaoNote/Others/CPU/CPU%5B4%5D.png)

![image](http://myblog-1253290602.file.myqcloud.com/YouDaoNote/Others/CPU/CPU%5B5%5D.png)

### 参考
https://blog.csdn.net/kobejayandy/article/details/24875881
https://www.jianshu.com/p/6903604cd1d4
https://blog.csdn.net/u012062455/article/details/78358113
https://blog.csdn.net/kobejayandy/article/details/24875881#commentBox

