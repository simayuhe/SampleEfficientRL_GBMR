import numpy as np
import matplotlib.pyplot as plt
path = "/home/kpl/SEGBMRresults/"
algorithm ='GQv2'
riqi = "0410"
envname = "Hero"

T=800
a=50

def average(M,a,T):
    t = M[0:T]
    for i in range(a):
        t=t+M[i:T+i]
    return t/a



# tmux a -t 0
name1=path+algorithm+riqi+envname+'-v4aver.npy'
NEC1=np.load(name1)
nec1 = average(NEC1,a,T)
l1, = plt.plot(range(T),nec1,label=name1,color="r")
plt.savefig(path+"Hero"+".png")
plt.close()
# # tmux a -t 0
# name=path+algorithm+riqi+envname
# name1="NEC0128-1bMsPacmanNoFrameskip-v4aver"
# NEC1=np.load(name1+".npy")
# nec1 = average(NEC1,a,T)

# # tmux a  -t 2
# name31="GQ0128-2-20bMsPacmanNoFrameskip-v4aver"
# GQ2201=np.load(name31+".npy")
# gq_2201 = average(GQ2201,a,T)

# # tmux a -t 3
# name34="GQ0128-2-50bMsPacmanNoFrameskip-v4aver"
# GQ4201=np.load(name34+".npy")
# gq_4201 = average(GQ4201,a,T)

# #tmux a -t 6
# name5="GBIL0128-2-20-3bMsPacmanNoFrameskip-v4aver"
# GBIL2203=np.load(name5+".npy")
# gbil2203 = average(GBIL2203,a,T)

# # tmux a -t 7
# name6="GBIL0128-2-20-5bMsPacmanNoFrameskip-v4aver"
# GBIL2205=np.load(name6+".npy")
# gbil2205 = average(GBIL2205,a,T)

# #tmux a -t 10
# name7="GBIL_2_0128-2-20bMsPacmanNoFrameskip-v4aver"
# GBIL2403=np.load(name7+".npy")
# gbil2403 = average(GBIL2403,a,T)

# # name8="GBIL0122-2-40-52MsPacmanNoFrameskip-v4aver"
# # GBIL2405=np.load(name8+".npy")
# # gbil2405 = average(GBIL2405,a,T)


# l1, = plt.plot(range(T),nec1,label=name1,color="r")
# l31, = plt.plot(range(T),gq_2201,label=name31,color=[0.7,0.4,0.6])
# l34, = plt.plot(range(T),gq_4201,label=name34,color=[0.7,0.6,0.1])
# l5, = plt.plot(range(T),gbil2203,label=name5,color="m")
# l6, = plt.plot(range(T),gbil2205,label=name6,color="y")
# l7, = plt.plot(range(T),gbil2403,label=name7,color="k")

# #l8, = plt.plot(range(T),gbil2405,label=name8,color=[0.2,0.3,0.4])
# plt.legend([l1,l31,l34,l5,l6,l7],[name1,name31,name34,name5,name6,name7],loc = 'lower right',fontsize ="xx-small")
# plt.savefig("compareMsPacmanNoFrameskipb"+".png")
# plt.close()


# 标记符    颜色
# r          红
# g          绿
# b          蓝
# c          蓝绿
# m          紫红
# y          黄
# k          黑
# w          白
