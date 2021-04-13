import numpy as np
import matplotlib.pyplot as plt
path = "/home/kpl/SEGBMRresults/"
algorithm1 ='GQv2'
algorithm2 = 'GBMRv2'
algorithm3 ='NEC'
algorithm4 = "GBIL_3_"
algorithm5 = "GBMRv1"
riqi1 = "0410"
riqi2 = "0410_1"
# riqi3 = "0227-1c"
# riqi3 = "0326-1gbil"
# riqi3 = "0410"
riqi3= "0128-1b"
riqi4 = "0227-2-50c"
riqi5 = "_0410_3"

envname = "Alien"
# envname = "MsPacman"
# envname = "Hero"
# envname = "BankHeist"
# envname = "Amidar"
# envname = "Bowling"
# envname = "Frostbite"
# envname = "Pong"
# envname = "MontezumaRevenge"
#Alien-v4, MsPacman-v4, Hero-v4, BankHeist-v4,Amidar-v4, Bowling-v4, Frostbite-v4, Pong-v4,MontezumaRevenge-v4，
T=800
a=50
def average(M,a,T):
    t = M[0:T]
    for i in range(a):
        t=t+M[i:T+i]
    return t/a

# tmux a -t 0
name1=path+algorithm1+riqi1+envname+'-v4aver.npy'
GQ1=np.load(name1)
gq1 = average(GQ1,a,T)
l1, = plt.plot(range(T),gq1,label=name1,color="r")

name2 = path+algorithm2+riqi2+envname+'-v4aver.npy'
GBMR1 = np.load(name2)
gbmr1 = average(GBMR1,a,T)
l2, = plt.plot(range(T),gbmr1,label=name2,color="m")

name3 = path+algorithm3+riqi3+envname+'-v4aver.npy'
NEC1 = np.load(name3)
nec1 = average(NEC1,a,T)
l3, = plt.plot(range(T),nec1,label=name3,color="g")

name4 = path+algorithm4+riqi4+envname+'-v4aver.npy'
GBIL1 = np.load(name4)
gbil1 = average(GBIL1,a,T)
l4, = plt.plot(range(T),gbil1,label=name4,color="b")

name5 = path+algorithm5+riqi5+envname+'-v4aver.npy'

GBMRv1 = np.load(name5)
print(GBMRv1[0:40])
gbmrv1 = average(GBMRv1,a,T)
l5, = plt.plot(range(T),gbmrv1,label=name5,color="c")

plt.legend([l1,l2,l3,l4,l5],[algorithm1,algorithm2,algorithm3,algorithm4,algorithm5],loc = 'lower right',fontsize ="xx-small")
plt.savefig(path+envname+".png")
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

