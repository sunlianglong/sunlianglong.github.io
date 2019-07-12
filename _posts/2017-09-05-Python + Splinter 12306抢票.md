---
layout:     post
title:      Python + Splinter 12306抢票
subtitle:   Python + Splinter 
date:       2017-09-05
author:     sunlianglong
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - Python
    - Splinter 
---


　　临近春节和期末放假，很多小伙伴也开始关注起来了12306的放票与抢票，学习python之余，敲一个小的demo，如果能帮到人成功抢到票，那便是最好不过的了。这个小的脚本完成过程中，做了很多的调试，力求满足所有人的条件。但是没有在关键抢票环时间点拿出来使用过，所以如果有人能试用下，记得留言给我反馈，感谢！

　　其实这个脚本就是模拟的鼠标点击与键盘输入而已，原理很简单，但是却从输入到点击都提升了很多档次，尤其是输入。还有就是在抢票开始前几分钟就可以打开脚本，他会一直检测你需要的车次与刷新界面，直到官方开放抢票接口的那一刻，都在不停检测。

### 实现步骤

1. 配置python + splinter环境，本人本次使用的是python2.7 + splinter0.7.7，splinter直接pip install即可。
2. 浏览器 +WebDriver。**建议Chrome + ChromeDriver**。由于我本机Chrome位置移动最后未能识别到，索性就使用了splinter默认的浏览器Firefox + FirefoxDriver 。
3. Chrome浏览器可自行下载，附上[浏览器驱动ChromeDriver]。(http://npm.taobao.org/mirrors/chromedriver/ "浏览器驱动ChromeDriver")链接。
4. 浏览器驱动下载下来是一个.exe执行文件，将其放在你本机python环境的目录下，也就是**跟python.exe同一目录**。要是找不到你的python环境，可以去环境变量中寻找。

### 注意点

1. Chrome 与驱动ChromeDriver 应该兼容，与splinter也应该兼容。我这里**均使用的最新版本**，打开上面的ChromeDriver网址，找到LATEST_RELEASE，下载并打开，会显示最新的ChromeDriver是哪个版本，然后下载最新版中的[chromedriver_win32.zip](http://npm.taobao.org/mirrors/chromedriver/2.34/ "chromedriver_win32.zip")，Mac跟Linux当然对应其他的驱动.zip包）。
2. 如果安装完毕后会报错，可能的原因有：版本不对不兼容；python环境不对，anaconda多环境的要注意下；驱动安装位置不对；浏览器的问题，试试重装最新版等。

### 半自动抢票代码实现

　　具体的实现逻辑比较简单，是splinter的基础练手内容，毕竟是初学者，在这里我就给大家讲一下我实现的抢票代码的使用方式，运行过后大家也可以试着改成自己想要的样子。
1. 上述环境等操作完成。
2. 根据自己的情况修改代码中的账号密码等数值，（虽然是python的初学者，但是没有将其写成一个接口，深感羞愧）。
3. 运行代码，唯一需要做的就是选择登录程序中的验证码，点击登录。如果自身浏览器比较慢的话，可能在个人界面卡一下，我并没有在代码上优化这一点，所以有的时候可能需要你手动点击一下*车票预订*（在后期可以试着写一下对验证码的识别，不过应该是比较难的，毕竟360在这上面花了那么多功夫，到时候再看）。
4. 实现的功能有：查询，预定，是否学生票，为朋友预定，他们是否学生票。最后自行选座位之后，点击最后一个提交按钮。
5.为什么是说半自动抢票呢？不是在于最后一步的自己选座位（我给选也不太好），而是在于要自己去找对应车站的cookie信息，比如重庆——潍坊，重庆的cookie是 *%u91CD%u5E86%2CCQW* ，潍坊的cookie是 *%u6F4D%u574A%2CWFK* 。怎么找呢？步骤如下：

- Chrome打开12306官网的[查询余票界面](https://kyfw.12306.cn/otn/leftTicket/init "查询余票界面")
- 选中出发点，或者目的地，按下F12，找到Application，点击Cookies看到的name/value就是对应的站点cookie信息。如图，框后对应的value就是对应的cookie。
<center>
<img src="http://myblog-1253290602.file.myqcloud.com/longlong-blog/cookie.png" width = "450" height = "330"/>
</center>

### 详细代码
```python
# -*- coding: utf-8 -*-
from splinter.browser import Browser
from time import sleep

# 以下是你需要根据自己情况修改的数值
# 可以提前试一下，免得到时候因为自己的输入或者网络太慢/浏览器太慢出现问题
# 抢到抢不到很大程度上依赖于机器和网络的好坏
my_name = "孙良龙(学生)"               # 你要抢票的人的名字，注意格式。当然名字要提前在12306上存档的。
myfriends_name1 = "XXX(学生)"          # 如果同时也给朋友抢的话，可以修改这一栏，没有的话不要修改，修改会报错
myfriends_name2 = "XXX(学生)"          # 如果同时也给朋友抢的话，可以修改这一栏，没有的话不要修改，修改会报错
myfriends_name3 = "XXX(学生)"          # 如果同时也给朋友抢的话，可以修改这一栏，没有的话不要修改，修改会报错
ticket_type = "普通票"                 # 你是要买学生票还是普通票
friend_ticket_type1 = "学生票"                 # 你的朋友1是要买学生票还是普通票
friend_ticket_type2 = "学生票"                 # 你的朋友2是要买学生票还是普通票
friend_ticket_type3 = "学生票"                 # 你的朋友3是要买学生票还是普通票
my_username = "1294054316@qq.com"  # 换成你的用户名
my_password = "sun651066702"  # 换成你的密码
my_time = "2018-02-06"  # 选择要抢票的日期，就算12306上面不可选也可以填 注意格式一定要与示例相同
train_num = 5        # 选择车次，按照在12306上面的次序，比如上海—太原的Z196，train_num为6，起始为0
train_from = "%u7AE0%u4E18%2CZTK"      # 选择你的始发站,这是示例重庆
train_to = "%u6F4D%u574A%2CWFK"        # 选择你的终点站，这是示例潍坊
# 起始地址的cookies值要自己去找，寻找你的始终站的方法：http://www.sunlianglong.cn/index.php/2018/01/07/python01/




# 12306
login_url = "https://kyfw.12306.cn/otn/login/init"
my_url = "https://kyfw.12306.cn/otn/index/initMy12306"
ticket_url = "https://kyfw.12306.cn/otn/leftTicket/init"
buy = "https://kyfw.12306.cn/otn/confirmPassenger/initDc"
driver = Browser()
driver.visit(login_url)
# username password
driver.fill("loginUserDTO.user_name", my_username)
driver.fill("userDTO.password", my_password)
print(u"等待验证码，自行输入...")

while True:
    if driver.url != my_url:
        sleep(1)
    else:
        break

driver.find_by_text(u"车票预订").first.click()

while True:
    if driver.url != ticket_url:
        sleep(1)
    else:
        break

# To determine if a single interface has been entered
flag = 0
while flag == 0:
    # 填写出发点目的地
    driver.cookies.add({u"_jc_save_fromStation": train_from})
    driver.cookies.add({"_jc_save_fromDate": my_time})
    driver.cookies.add({u"_jc_save_toStation": train_to})
    # loading
    driver.reload()
    order = 1
    if order != 0:
        while True:
            driver.find_by_text(u"查询").click()
            try:
                noDate2 = driver.find_by_id("qd_closeDefaultWarningWindowDialog_id")
                if noDate2 == []:
                    wait = driver.find_by_text(u"预订")[train_num]
                    if wait == []:
                        break
                    else:
                        wait.click()
                    # To determine whether a successful reservation
                    if driver.url != buy:
                        order = order-1
                else:
                    break
            except Exception as e:
                print u"还没开始预订"
                continue
            if order == 0:
                flag = 1
                break

print flag
if True:
    # Choose yourself
    driver.find_by_text(my_name).click()
    stu_toast = driver.find_by_id("dialog_xsertcj_ok")
    if stu_toast != [] :
        if ticket_type == "学生票":
            stu_toast.click()
        else:
            driver.find_by_id("dialog_xsertcj_cancel").click()
    # Choose friend 1
    if myfriends_name1 != "XXX(学生)":
        driver.find_by_text(myfriends_name1).click()
        stu_toast1 = driver.find_by_id("dialog_xsertcj_ok")
        if stu_toast1 != []:
            if friend_ticket_type1 == "学生票":
                stu_toast1.click()
            else:
                driver.find_by_id("dialog_xsertcj_cancel").click()

    # Choose friend 2
    if myfriends_name2 != "XXX(学生)":
        driver.find_by_text(myfriends_name2).click()
        stu_toast2 = driver.find_by_id("dialog_xsertcj_ok")
        if stu_toast2 != []:
            if friend_ticket_type2 == "学生票":
                stu_toast2.click()
            else:
                driver.find_by_id("dialog_xsertcj_cancel").click()

    # Choose friend 3
    if myfriends_name3 != "XXX(学生)":
        driver.find_by_text(myfriends_name3).click()
        stu_toast3 = driver.find_by_id("dialog_xsertcj_ok")
        if stu_toast3 != []:
            if friend_ticket_type3 == "学生票":
                stu_toast3.click()
            else:
                driver.find_by_id("dialog_xsertcj_cancel").click()

while True:
    driver.find_by_text(u"提交订单").click()
    if driver.url != buy:
        break

print "success!"


```







