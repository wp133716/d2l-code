'''
在matplotlib中画图有两种显示模式：

（1）阻塞模式，即必须利用plt.show()显示图片，且图片关闭之前代码将阻塞在该行。

（2）交互模式，即plt.plot()后立马显示图片，且不阻塞代码的继续运行。

Matplotlib中默认是使用阻塞模式。看一下这里用到的matplotlib中的几个函数：

plt.ion()：打开交互模式
plt.ioff()：关闭交互模式
plt.clf()：清除当前的Figure对象
plt.cla()：清除当前的Axes对象
plt.pause()：暂停功能 # 测试只用plt.pause()就可以实现动态图，原因未知
'''

import matplotlib.pyplot as plt
x = list(range(1, 21))  # epoch array
loss = [2 / (i**2) for i in x]  # loss values array
plt.ion()
for i in range(1, len(x)):
    print("###",i)
    ix = x[:i]
    iy = loss[:i]
    plt.cla()
    plt.title("loss")
    plt.plot(ix, iy)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.pause(0.5)
plt.ioff()
plt.show()
