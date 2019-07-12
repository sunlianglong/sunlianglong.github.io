---
layout:     post
title:      《云计算与云服务》课程大作业——大数据平台的使用(Hadoop 生态圈、CDH)
subtitle:   《云计算与云服务》课程大作业——大数据平台的使用(Hadoop 生态圈、CDH)
date:       2018-06-22
author:     sunlianglong
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - 云计算
    - 大数据平台
    - Hadoop
    - CDH
---

# 目录
-   [**一、shell自行搭建Hadoop集群（2节点以上）**](#一shell自行搭建hadoop集群2节点以上)
    -   [1.1 系统准备](#系统准备)
    -   [1.2 系统基础配置](#系统基础配置)
    -   [1.3 组件安装与配置](#组件安装与配置)
        -   [1.3.1 Hadoop](#hadoop)
        -   [1.3.2 Hive](#hive)
        -   [1.3.3 Hbase](#hbase)
        -   [1.3.4 Spark](#spark)
-   [**二、Cloudera CDH安装Hadoop平台**](#二cloudera-cdh安装hadoop平台)
    -   [2.1 Cloudera quickstart 安装](#cloudera-quickstart-安装)
    -   [2.2 CDH 中HQL数据操作](#cdh-中hql数据操作)
-   [**三、集群中的HQL数据操作**](#三集群中的hql数据操作)
    -   [3.1 创建表](#创建表)
    -   [3.2 创建分区](#创建分区)
    -   [3.3 Hive数据导入](#hive数据导入)
    -   [3.4 Hive操作](#hive操作)
    -   [3.5 hdfs目录分区](#hdfs目录分区)
-   [**四、Spark程序分析HDFS或Hive数据**](#四spark程序分析hdfs或hive数据)
    -   [4.1版本以及流程测试](#版本以及流程测试)
        -   [4.1.1 准备maven集成环境](#准备maven集成环境)
        -   [4.1.2 生成jar包上传集群](#生成jar包上传集群)
        -   [4.1.3 Spark UI界面](#spark-ui界面)
-   [**五、使用oozie调度spark job**](#五使用oozie调度spark-job)
    -   [5.1 oozie的编译、安装与配置](#oozie的编译安装与配置)
        -   [5.1.1 pig 安装与配置](#pig-安装与配置)
        -   [5.1.2 更改maven中央仓库](#更改maven中央仓库)
        -   [5.1.3 下载oozie并编译](#下载oozie并编译)
        -   [5.1.4 报错以及解决](#报错以及解决)
        -   [5.1.5 验证oozie是否安装成功](#验证oozie是否安装成功)
    -   [5.2 CDH中使用Oozie调度Spark](#cdh中使用oozie调度spark)
-   [**六、spark mllib算法使用与数据分析**](#六spark-mllib算法使用与数据分析)
    -   [6.1 20 news_groups数据与Mllib调用](#news_groups数据与mllib调用)
        -   [6.1.1 读取数据](#读取数据)
        -   [6.1.2 基本分词和词组过滤](#基本分词和词组过滤)
        -   [6.1.3 训练TF-IDF模型](#训练tf-idf模型)
        -   [6.1.4 朴素贝叶斯算法](#朴素贝叶斯算法)
    -   [6.2 Job运行分析](#job运行分析)


一、shell自行搭建Hadoop集群（2节点以上）
========================================

**1.1 系统准备**
----------------
　　准备centos6.5系统，命名为master，按照虚拟机安装镜像的步骤安装之后，克隆一份到slave，也就是**准备一个mater节点，一个slave节点的系统开发环境**。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_1.png" width = "300" height = "300"/>
</center>
**1.2 系统基础配置**
--------------------

　　**关闭防火墙，设置静态IP，设置SSH免密登陆。**

　　关闭防火墙的目的是，我们需要配置的组件应用程序需要使用一些端口，这些端口容易被防火墙封掉，那么这时候有两种选择：要么关闭防火墙，要么将某些应用程序需要的端口在防火墙的规则中进行配置。在实验环境下，暂且不考虑黑客攻击的情况，所以就直接关掉防火墙，以方便下面的配置。

　　设置静态IP的原因是，每次系统重新启动之后，有些系统默认设置下，IP都会随机改变，我们需要在host文件中修改主机名列表，便于master和slave IP的识别和后续操作，所以需要设置静态IP。

　　SSH免密登陆出现在scp远程复制、Hadoop启动、Spark启动等操作中，如果不设置免密登陆每次都要输入很多次密码。

**1.3 组件安装与配置**
----------------------

　　Hadoop生态圈含有很多组件，而且呈现出一个堆积金字塔状态，上面的组件是基于下面组建的成功运行来搭建的，所以我们再手动搭建组件的时候，也要遵循先后的原则，在这次大作业中，可能只需要用到最基础的组件和spark，以及oozie，在这里我只介绍最基础的组件，oozie第五部分会详细讲。

### 1.3.1 Hadoop

\(1) hadoop安装与配置

　　我们可选择用yum命令进行安装，一行命令解决所有；但是在这里采取手动安装的方式，自行配置环境变量等，以便于自己更好的理解。首先下载`hadoop-2.5.2.tar.gz`，下载的方式有很多，**wget+下载网址**命令很方便，但是有的较老版本的组件安装包的链接已经失效，所以我们可以前往apache官网根据自己的需要进行下载，随后解压，解压目录和文件列表如下。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_2.png" />
</center>

　　解压完成后的下一步就是配置环境变量运行脚本，进入修改`hadoop-2.5.2/etc/hadoop/hadoop-env.sh`，将之前安装好的java路径写入。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_3.png" />
</center>

　　以同样的方法配置环境变量`yarn-env.sh`。然后，配置核心组件`core-site.xml`、文件系统`hdfs-site.xml`、调度系统`yarn-site.xml`、计算框架`mapred-site.xml`。配置完成后，用`scp`将整个hadoop文件远程复制到slave节点的`\~`节点下。

\(2) hadoop启动

　　在master节点下输入`jps`会发现有**4个进程**，在slave节点下输入`jps`会发现有**3个进程**，说明配置成功。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_4.png" width = "600" height = "600" />
</center>

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_5.png" width = "600" height = "600" />
</center>

　　为了进一步核实与验证，我们打开浏览器，输 `http://master:18088` ,显示出如下配置。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_6.png" width = "600" height = "600" />
</center>

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_7.png" width = "600" height = "600" />
</center>

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_8.png" width = "600" height = "600" />
</center>

\(3) HDFS文件目录

　　`hadoop fs -ls / `查看hdfs文件目录。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_9.png" width = "600" height = "600" />
</center>

### 1.3.2 Hive

　　配置hive的方法同理，在这里遇到一个问题，那就是启动hive时报错。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_10.png" width = "600" height = "600" />
</center>

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_11.png" width = "600" height = "600" />
</center>

　　根据报错，排查原因发现是mysql没有开启。解决办法：**重启mysql服务**，注意要在root下执行命令才会成功。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_12.png" width = "600" height = "600" />
</center>

　　重新启动hive成功。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_13.png" width = "600" height = "600" />
</center>

### 1.3.3 Hbase

　　进入到hbase目录，运行`start-hbase.sh`，启动Hbase。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_14.png" width = "600" height = "600" />
</center>

　　浏览器输入：`http://master:60010`, 打开控制面板，启动成功。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_15.png" width = "600" height = "600" />
</center>

### 1.3.4 Spark

　　安装并配置spark，进入到spark目录下并启动spark。jps查看进程，如下图所示，**master节点下增加了两个进程，slave节点下增加了一个进程。其中Worker进程运行具体逻辑的进程。**
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_16.png" width = "600" height = "600" />
</center>

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_17.png" />
</center>

　　执行`./spark-shell `得到命令窗口，spark默认的是scala语言。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_18.png" width = "600" height = "600" />
</center>

　　浏览器中输入:`http://master:4040`，得到spark**临时**控制面板，这不是主要的界面，所以我称之为临时控制面板，具体内容第四部分会详细讲。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_19.png" width = "600" height = "600" />
</center>

二、Cloudera CDH安装Hadoop平台
==============================

**2.1 Cloudera quickstart 安装**
--------------------------------

　　在Cloudrea官网上下载quick-start vm镜像，在虚拟机中分配8G内存启动。在虚拟机中安装中文支持包，并更改系统语言为中文，这时候将cloudera manager控制面板的语言更改为中文，才会成功.安装好的CDH如下：
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_20.png" width = "600" height = "600" />
</center>

**2.2 CDH 中HQL数据操作**
-------------------------

　　在HUE页面下，建表gps，并将`gps.csv`数据传入到hive数据库中，实施查询功能。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_21.png" width = "600" height = "600" />
</center>

三、集群中的HQL数据操作
=======================

　　进入到hive目录下并启动hive。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_22.jpg" width = "600" height = "600" />
</center>

**3.1 创建表**
--------------

　　创建表gps，以便后面导入`2016-06-01`重庆部分地区的gps数据，根据时间时间建立分区，数据之间以逗号间隔。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_23.png" width = "600" height = "600" />
</center>

**3.2 创建分区**
----------------

　　为表gps增加两个分区，一个是`20160601`，代表的是2016-06-01这一天，另外一个是`20160602`。`desc gps `查看表gps的结构。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_24.png" width = "600" height = "600" />
</center>

**3.3 Hive数据导入**
--------------------

　　Hive数据的导入可以分为三种，**一是本地文件的导入，二是hdfs文件的导入，三是`insert`或者创建表时直接导入**。我首先尝试本地文件的导入，将gps数据上传到系统中，输入以下命令将其导入到gps表，并声明放入的分区。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_25.png" width = "600" height = "600" />
</center>

**3.4 Hive操作**
----------------

　　select基础查询导入的数据。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_26.png" width = "600" height = "600" />
</center>

**3.5 hdfs目录分区**
--------------------

　　导入hive的数据都默认放在了hdfs下的hive仓库中，具体路径如下。而且分区的建立，使得数据的存放分隔而独立。分别在gps下面建立了`date=20160601`和`date=20160602`文件夹。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_27.png" width = "600" height = "600" />
</center>

四、Spark程序分析HDFS或Hive数据
===============================

**4.1版本以及流程测试**
-----------------------

### 4.1.1 准备maven集成环境

　　关于第四部分用spark程序分析HDFS或者Hive数据，我首先做一个简单的词频统计来实现读取HDFS数据的任务，目的在于测试一下自己win10系统下scala-spark的版本与我的集群的兼容情况。因为scala与spark的版本要兼容，运行的程序才能最大限度不报错。

　　我win10下的scala是2.10.4版本，spark版本为1.6，可以兼容运行。然而我的集群下的spark版本为1.2，最终要把代码打包到集群上运行，所以测试一下是有必要的。我的任务流程为win10下面打jar包，上传到master集群，然后采用spark集群的方式运行。首先将win10下IDEA中的依赖包替换成集群中的版本：`spark-assembly-1.2.0-hadoop2.4.0`。

　　其次，用IDEA+Maven+Scala创建wordcount,并打包jar，项目的目录结构如下：
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_28.png" width = "600" height = "600" />
</center>

　　注意要导入scala-sdk，之后编辑代码，`setMaster("local")`代表以本地模式运行，导入文件路径，运行WordCount，输入控制台如下所示，只是完成了map的功能。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_29.png" width = "600" height = "600" />
</center>

### 4.1.2 生成jar包上传集群

　　之后准备打jar包，并上传到集群运行，可以在代码中更改运行方式，更改为`setMaster("spark://master:7077")`,意为在集群上运行，`spark://master:7077`是集群运行的默认端口。也可以不更改，那么代码就只会在master节点上单机运行，在这里先不做更改，第六部分学习Mllib的时候再更改为集群的方式运行。其次，要修改导入的文件，将第八行改为`sc.textfile(args(0))`，`args(0)`是传入的第一个参数。

　　更改完毕后，准备打jar包，参考网上的快捷打包方式很简单的就在out文件夹里生成了打好的jar包，在这里要注意把多余的依赖全都删掉，只留下下图中的那一个。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_30.png" width = "600" height = "600" />
</center>

　　提交到集群，`spark-submit`命令运行，运行的结果跟IDEA里面运行的结果一模一样，如下。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_31.png" width = "600" height = "600" />
</center>

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_32.png" width = "600" height = "600" />
</center>

### 4.1.3 Spark UI界面

　　在运行途中，打开:`http://master:4040 `是可以看到此时job的运行情况的，但是一旦`sc.stop`，也就是job任务完毕，SparkContext终止，那么网址便不可访问。因为每一个SparkContext会发布一个web界面，默认端口是4040，它显示了应用程序的有用信息。如果某台机器上运行多个SparkContext,它的web端口会自动连续加一，比如4041,4042,4043等等。就像第一部分中启动`./spark-shell`，其实启动过后，就是启动了一个SparkContext，终止命令窗口时，就是执行了`sc.stop`命令。

　　但是只要hadoop启动成功，spark集群启动成功，那么spark集群的web端口：8080便会一直可用，`http://master:8080` 可以一直访问。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_33.png" width = "600" height = "600" />
</center>

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_34.png" width = "600" height = "600" />
</center>

　　Spark 程序运行完(成功/失败)后，我们将无法查看Application的历史记录，但是按照经验来说这不应该啊，hadoop生态圈中对于运行日志的保存至关重要，怎么会出现这样的情况呢？google一下，果然有解决办法，那就是`Spark history Server`。

　　`Spark history Server`就是为了应对这种情况而产生的，通过配置可以在Application执行的过程中记录下了日志事件信息，那么在Application执行结束后，WEBUI就能重新渲染生成UI界面展现出该Application在执行过程中的运行时信息。具体的配置很简单，在这里就不在阐述。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_35.png" width = "600" height = "600" />
</center>

　　经过测试，版本应该没有太大的问题，那么可以安心的学习Mllib，不再为版本和集群环境而困扰。

五、使用oozie调度spark job
==========================

　　Apache Oozie 是一个用于管理 Apache Hadoop 作业的工作流调度程序。Oozie 非常灵活。人们可以很容易启动，停止，暂停和重新运行作业。Oozie 可以很容易地重新运行失败的工作流。可以很容易重做因宕机或故障错过或失败的作业。甚至有可能跳过一个特定故障节点。oozie本身 apache 只提供源码，需要自己编译，在节点上自行安装并编译oozie稍微有点复杂，在CDH上面就可以很简单的使用，但是也不妨尝试一下，所以在本节中，我在我的master上自行搭建oozie的同时，也在CDH上面学会使用。

**5.1 oozie的编译、安装与配置**
-------------------------------

　　首先根据网上的教程，额外安装Maven 3.0.1+、Pig 0.7+，趁此机会也学习一下Pig。

### 5.1.1 pig 安装与配置

　　Pig 有两种运行模式： Local 模式和 MapReduce 模式。当 Pig 在 Local 模式运行的时候， Pig 将只访问本地一台主机；当 Pig 在 MapReduce 模式运行的时候， Pig 将访问一个 Hadoop 集群和 HDFS 的安装位置。这时， Pig 将自动地对这个集群进行分配和回收。因为 Pig 系统可以自动地对 MapReduce 程序进行优化，所以当用户使用 `Pig Latin` 语言进行编程的时候，不必关心程序运行的效率， Pig 系统将会自动地对程序进行优化。这样能够大量节省用户编程的时间。

　　Pig 的 Local 模式和 MapReduce 模式都有三种运行方式，分别为： `Grunt Shell` 方式、脚本文件方式和嵌入式程序方式。

　　下载pig-0.7版本，配置好`profile`下的环境变量，输入`pig -help`验证成功。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_36.png" width = "600" height = "600" />
</center>

运行`pig -x local` 以`Grunt Shell `方式执行。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_37.png" width = "600" height = "600" />
</center>

### 5.1.2 更改maven中央仓库

　　国外镜像下载很慢，所以将Maven的中央仓库修改为阿里云的仓库地址，但是修改后会出现依赖包不全的问题，下面会解决。

```html
<mirror>
    <id>nexus-aliyun</id>
    <mirrorOf>*</mirrorOf>
    <name>Nexus aliyun</name>
    <url>http://maven.aliyun.com/nexus/content/groups/public</url>
</mirror>

```

### 5.1.3 下载oozie并编译

　　去Apache官网下载`oozie-4.3.0.tar.gz`（不是最新版本），解压到`\~`目录。进入解压后的目录
`oozie-4.3.0`，执行`mvn clean package assembly:single -DskipTests` 编译命令，在阿里云镜像库中编译oozie。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_38.png" width = "600" height = "600" />
</center>

　　第一次编译过程中，报出找不到依赖包的错误：`Could not find artifact org.pentaho:pentaho-aggdesigner-algorithm:jar:5.1.5-jhyde in Central`。下载不了该jar包我就去手工下载，先找到下载地址，就会发现该jar包来源根本不是maven的central仓库，而是spring。下载地址为：`http://repo.spring.io/plugins-release/org/pentaho/pentaho-aggdesigner-algorithm/5.1.5-jhyde/pentaho-aggdesigner-algorithm-5.1.5-jhyde.jar`。

　　进入系统的`\~/.m2/repository/org/pentaho/pentaho-aggdesigner-algorithm/5.1.5-jhyde/`目录下，`rm -rf \* `清空目录，手工上传下载好的jar包。

　　第二次、第三次编译过程中，仍然报出找不到依赖包的错误。这时，我就依照报错的提示找到相应的依赖包，手动下载并上传到对应的依赖目录。第四次编译的时候，成功，输入以下内容，最后一行显示build
success。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_39.png" width = "600" height = "600" />
</center>

### 5.1.4 报错以及解决

　　生成oozie数据库脚本文件，初始化mysql时报错：`ERROR 1045 (28000): Access denied for user \'oozie\'@\'localhost\' (using password: YES)`
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_40.png" width = "600" height = "600" />
</center>

　　解决办法，为oozie用户授权。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_41.png" width = "600" height = "600" />
</center>

　　再次启动oozie，运行`bin/oozied.sh start`，得到以下显示，代表成功。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_42.png" width = "600" height = "600" />
</center>

### 5.1.5 验证oozie是否安装成功

　　运行` bin/oozie admin -oozie http://localhost:11000/oozie -status`，输出`System mode: NORMAL`即表示配置成功。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_43.png" width = "600" height = "600" />
</center>

　　在浏览器中打开`http://localhost:11000/oozie/`，显示如图所示。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_44.png" width = "600" height = "600" />
</center>

**5.2 CDH中使用Oozie调度Spark**
-------------------------------

　　进入到HUE界面，打开hdfs，传入依赖包`SparkTest.jar` 和` word_txt`文件。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_45.png" width = "600" height = "600" />
</center>

　　点击Query，新建一个spark任务，依照下图传入相应的依赖包和设置相应的参数。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_46.png" width = "600" height = "600" />
</center>

　　然后点击左下角的执行按钮，在CDH里面执行简单的spark操作，结果如下图。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_47.png" width = "600" height = "600" />
</center>

　　创建Oozie工作流。注意，只有新建了spark任务之后，创建Oozie工作流时才可以识别到传入的spark任务。在下图中，我们看到有hive任务、pig任务、spark-shel脚本、spark等等，这些都是Oozie支持的作业。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_48.png" width = "600" height = "600" />
</center>

　　设置完毕后，点击执行按钮开始执行Oozie调度。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_49.png" width = "600" height = "600" />
</center>

　　在执行之前，有个选项：`Do a dryrun before submitting`。也就是可以选择先执行一个演练任务，具体可以在Oozie UI界面中看到。

　　如下图在Oozie UI界面中我们可以看到此次调度的情况。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_50.png" width = "600" height = "600" />
</center>

　　电脑配置问题，12G的内存，分配8G+2核CPU给CDH都会很卡，勉强赶上进度，运行一个很简单词频统计都要卡半天，所以有一些细小的问题不在改动，因为太浪费时间了，等更换了电脑之后，按照CDH的官网参看文档，一点点再来学习，对我来说还是蛮有趣的，所以大作业中关于CDH的部分到此为止。

六、spark mllib算法使用与数据分析
=================================

**6.1 20 news\_groups数据与Mllib调用**
--------------------------------------

　　因为20 news_groups数据量还是蛮大的（相对于的电脑的配置来说），所以我并没有全部使用。用spark的一个好处也是我们可以本地运行小批量数据集，准确无误后再在集群上运行整个大数据集，节约时间。数据一共大约20000条，我从中均匀选择了200条(每类10条)作为训练集，测试集只选了10条左右，只是为了在本地测试代码，所以效果没那么好。Maven项目的Pom.xml配置如下：
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_51.png" width = "600" height = "600" />
</center>

### 6.1.1 读取数据

　　参考《Spark机器学习》和scala API文档，我对数据的分析处理，代码如下。

```scala
val sparkConf = new SparkConf().setAppName("SparkNLP").setMaster("spark://master:7077")
val data = args(0)
val sc = new SparkContext(sparkConf)
val rdd = sc.wholeTextFiles(data) //加载数据
val text = rdd.map { case (file, text) => text }
text.cache()
println(text.count)//200
val news_groups = rdd.map { case (file, text) => file.split("/").takeRight(2).head }
val countByGroup = news_groups.map { n => (n, 1) }.reduceByKey(_ + _).collect()
.sortBy(-_._2).mkString("\n")
println(countByGroup)

```
　　项目的运行环境设置为`spark://master:7077`，意为在集群上运行。读取第一个参数，也就是训练集的hdfs路径。之后加载数据，对数据进行一些转换操作，注意在这里将相应的RDD进行`cache`或者`persist`操作，是为了将其缓存下来，因为RDD懒惰执行的特性，每一次的Action操作都会将无环图所涉及的RDD执行一遍，所以将常用的RDD缓存下来可以节约时间。

　　补充一点，`reduceByKey`等函数，以及一些标识符的使用在这里可能会报错，不被识别，原因是在spark1.2.0以及之前的版本中，RDD没有`reduceByKey`这些方法。将其隐式转换成`PairRDDFunctions`才能访问，在前面导入语句：`import org.apache.spark.SparkContext._ `。

　　转换操作中，`sortBy(-_._2)`的意思是将新闻主题按主题统计个数并按从大到小排序。这段代码最终得到的是按照数量进行排序的新闻主题。

### 6.1.2 基本分词和词组过滤

```scala
val whiteSpaceSplit = text.flatMap(t => t.split(" ").map(_.toLowerCase))
val nonWordSplit = text.flatMap(t => t.split("""\W+""").map(_.toLowerCase)) //把不是单词的字符过滤掉
val regex = """[^0-9]*""".r
val filterNumbers = nonWordSplit.filter(token => regex.pattern.matcher(token).matches)
val tokenCounts = filterNumbers.map(t => (t, 1)).reduceByKey(_ + _)
val oreringDesc = Ordering.by[(String, Int), Int](_._2)
println(tokenCounts.top(20)(oreringDesc).mkString("\n"))
val stopwords = Set(
"the","a","an","of","or","in","for","by","on","but","is","not","with","as","was","if","they","are","this","and","it","have","from","at","my","be","that","to"
)//停用词表
//过滤掉停用词
val tokenCountsFilteredStopWords = tokenCounts.filter{  case (k,v) => ! stopwords.contains(k) }
//过滤掉仅仅含有一个字符的单词
val tokenCountsFilteredSize = tokenCountsFilteredStopWords.filter{case (k,v) => k.size >= 2}
//基于频率去除单词：去掉在整个文本库中出现频率很低的单词
val oreringAsc = Ordering.by[(String, Int), Int](- _._2)//新建一个排序器，将单词，次数对按照次数从小到大排序
//将所有出现次数小于2的单词集合拿到
val rareTokens = tokenCounts.filter{ case (k, v) => v < 2 }.map{ case (k, v) => k }.collect().toSet
//过滤掉所有出现次数小于2的单词
val tokenCountsFilteredAll = tokenCountsFilteredSize.filter{ case (k, v) => !rareTokens.contains(k) }
println(tokenCountsFilteredAll.top(20)(oreringAsc).mkString(","))
println(tokenCountsFilteredAll.count)//打印不同的单词

```

　　首先执行基本的分词，然后把不是单词的字符过滤掉，我们可以使用正则表达式切分原始文档来移除这些非单词字符；然后手动去掉停用词，过滤掉停用词，过滤掉仅含有一个字符的单词，去掉整个文本库中出现频率很低的单词,过滤掉所有出现次数小于2的单词。

　　为什么知道该这么操作呢？每一个转换操作之后，我们都可以打印一下当前数据的形式，查看需要对现在的RDD执行什么过滤操作。我将程序设置为本地模式运行后，打印出过滤过程的一系列结果。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_52.png" width = "600" height = "600" />
</center>

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_53.png" width = "600" height = "600" />
</center>

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_54.png" width = "600" height = "600" />
</center>

<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_55.png" width = "600" height = "600" />
</center>

### 6.1.3 训练TF-IDF模型

```scala
val dim = math.pow(2, 18).toInt
val hashingTF = new HashingTF(dim)
//HashingTF的transform函数把每个输入文档（即词项的序列）映射到一个MLlib的Vector对象。
val tf = hashingTF.transform(tokens)
tf.cache()//把数据保持在内存中加速之后的操作
val v = tf.first().asInstanceOf[SV]
println("tf的第一个向量大小:" + v.size)//262144
println("非0项个数：" + v.values.size)//706
println("前10列的下标：" + v.values.take(10).toSeq)
//WrappedArray(1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0)
println("前10列的词频：" + v.indices.take(10).toSeq)
//WrappedArray(313, 713, 871, 1202, 1203, 1209, 1795, 1862, 3115, 3166)

```

　　HashingTF使用特征哈希把每个输入文本的词映射为一个词频向量的下标，每个词频向量的下标是一个哈希值（依次映射到特征向量的某个维度），词项的值是本身的TF-IDF权重。可以看到每一个词频的稀疏向量的维度是262144（2\^18）.然而向量中的非0项只有706个。

　　下面分析TF-IDF权重，并计算整个文档的TF-IDF最小和最大权值。

```scala
val idf = new IDF().fit(tf)
//transform将词频向量转为TF_IDF向量
val tfidf = idf.transform(tf)
val v2 = tfidf.first().asInstanceOf[SV]
val minMaxVals = tfidf.map{ v => val sv = v.asInstanceOf[SV]
  (sv.values.min, sv.values.max)
}
val globalMinMax = minMaxVals.reduce{ case ( (min1, max1), (min2, max2)) =>
  (math.min(min1, min2), math.max(max1, max2))
}

```

### 6.1.4 朴素贝叶斯算法

```scala
val newsgroupMap = news_groups.distinct().collect().zipWithIndex.toMap
val zipped = news_groups.zip(tfidf)
val train = zipped.map{ case (topic, vector) =>
  LabeledPoint(newsgroupMap(topic), vector)
}
```

　　从新闻组RDD开始，其中每个元素是一个话题，使用`zipWithIndex`,给每个类赋予一个数字下标.使用zip函数把话题和由TF-IDF向量组成的tiidf RDD组合，其中每个label是一个类下标，特征就是IF-IDF向量。

```scala
//加载测试集数据
val testRDD = sc.wholeTextFiles(args(1))
val testLabes = testRDD.map{  case (file, text) =>
  val topic = file.split("/").takeRight(2).head
  newsgroupMap(topic) }//根据类别得到此类别的标号
val testTf = testRDD.map{ case (file, text) => hashingTF.transform(tokenize(text)) }
val testTfIdf = idf.transform(testTf)//将MLlib的Vector对象转化为tf-idf向量
val zippedTest = testLabes.zip(testTfIdf)//将类型和向量打包
//将类型和向量打包转换为LabeledPoint对象
val test = zippedTest.map{  case (topic, vector) => LabeledPoint(topic, vector) }
val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
//使用模型预测
val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
val metrics = new MulticlassMetrics(predictionAndLabels = predictionAndLabel)
//计算加权F指标，是一个综合了准确率和召回率的指标（这里类似于ROC曲线下的面积，当接近1时有较好的表现），并通过类之间加权平均整合
println("准确率=" + accuracy)
```


　　将每个文件的内容进行分词处理，`HashingT`F的`transform`函数把每个输入文档（即词项的序列）映射到一个MLlib的Vector对象。得到结果如下所示：
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_56.png" width = "600" height = "600" />
</center>

　　因数据集太少(训练集200，测试集6)，只是为了节约时间而抛出结果，所以结果不理想。

**6.2 Job运行分析**
-------------------

　　提交任务到集群：
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_57.png" width = "600" height = "600" />
</center>

　　打开`http://master:4040`查看Job运行状态如下。4040界面详细给出了每一个Job的运行情况，正在运行的以及运行完成，这些运行在不同集群的Job组成Spark任务。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_58.png" width = "600" height = "600" />
</center>

　　打开`http://master:8080`查看spark的WEB集成界面，在这里显示了Spark任务的情况，正在运行的任务，已经完成的任务，以及worker的ID和资源状态一目了然。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_59.png" width = "600" height = "600" />
</center>

　　下面是Spark Job的状态、存储等界面，也是详细给出了各部分的运行时状态。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_60.png" width = "600" height = "600" />
</center>
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_61.png" width = "600" height = "600" />
</center>

　　等到任务运行完毕后，4040端口随着SparkContext的停止而停止，任务的状态也由Runing转化成为了Completed。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_62.png" width = "600" height = "600" />
</center>

　　任务最终完成的结果。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/bigdata/bigdata_63.png" width = "600" height = "600" />
</center>

## 七、部分博客参考

- [Maven将中央仓库修改为阿里云的仓库地址](https://blog.csdn.net/a639735331/article/details/79161798 "Maven将中央仓库修改为阿里云的仓库地址")
- [Pig系列之二：Pig的安装和配置](https://blog.csdn.net/shenlan211314/article/details/6326599 "Pig系列之二：Pig的安装和配置")
- [oozie搭建及examples使用教程](http://trumandu.github.io/2017/06/01/oozie%E6%90%AD%E5%BB%BA%E5%8F%8Aexamples%E4%BD%BF%E7%94%A8%E6%95%99%E7%A8%8B/ "oozie搭建及examples使用教程")
- [Oozie编译-安装-配置](https://blog.csdn.net/Shannon_ying/article/details/51025214 "Oozie编译-安装-配置")
- [使用Oozie在YARN上编写使用PySpark或SparkR编写的Spark作业](https://developer.ibm.com/hadoop/2017/06/30/scheduling-spark-job-written-pyspark-sparkr-yarn-oozie/ "使用Oozie在YARN上编写使用PySpark或SparkR编写的Spark作业")
- [在 YARN 上使用 Oozie 调度用 Java 或 Scala 编写的 SparkSQL 或 SparkML 作业](https://developer.ibm.com/cn/blog/2017/scheduling-sparksql-sparkml-job-written-java-scala-yarn-oozie/ "在 YARN 上使用 Oozie 调度用 Java 或 Scala 编写的 SparkSQL 或 SparkML 作业")
- [错误1045（28000）：拒绝访问用户'root @ localhost'（使用密码：否）](https://askubuntu.com/questions/401449/error-104528000-access-denied-for-user-rootlocalhost-using-password-no "错误1045（28000）：拒绝访问用户'root @ localhost'（使用密码：否）")
- [oozie4.3.0的安装与配置 + hadoop2.7.3](https://www.cnblogs.com/30go/p/8335523.html "oozie4.3.0的安装与配置 + hadoop2.7.3")
- [oozie 安装过程总结](https://segmentfault.com/a/1190000002738484 "oozie 安装过程总结")
- [《Spark机器学习》笔记——Spark高级文本处理技术（NLP、特征哈希、TF-IDF、朴素贝叶斯多分类、Word2Vec）](https://blog.csdn.net/csj941227/article/details/79028661 "《Spark机器学习》笔记——Spark高级文本处理技术（NLP、特征哈希、TF-IDF、朴素贝叶斯多分类、Word2Vec）")
- [spark web UI端口 4040,18080打不开](https://blog.csdn.net/Heitao5200/article/details/79674684 "spark web UI端口 4040,18080打不开")


























