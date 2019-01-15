import numpy as np
import math as M
import matplotlib.pyplot as plt

filename = 'glacierdata4.txt'
Equ = np.loadtxt(filename, delimiter="\t")
#print(Equ[-2])
#print(np.mean([3230,3430,3370,4430,3140,3080,2720,306,3470,2440,2600]))
#print((1100**2+45000**2)**(0.5))

Equ1=np.zeros(len(Equ))
Equ2=np.zeros(len(Equ))

for i in range(0,len(Equ)):
    Equ1[i]=Equ[i][0]
    Equ2[i]=Equ[i][1]

begin=np.mean(Equ2[-10:])
print(begin)
jaar = 2200
slope = 50*0.04
for i in range(1,2100-int(Equ1[-1])):
    Equ1=np.append(Equ1,2016+i)
    Equ2=np.append(Equ2,begin+slope*(i))

for i in range(1,jaar-2100):
    Equ1=np.append(Equ1,2100+i)
    Equ2=np.append(Equ2,Equ2[-1])

plt.plot(Equ1[-300:],Equ2[-300:])
print(Equ1[-301])
