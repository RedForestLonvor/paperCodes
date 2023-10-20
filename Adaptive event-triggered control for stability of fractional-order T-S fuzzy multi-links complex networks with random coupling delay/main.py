import math
import random
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

Tspan = 15
s = 0.01
step = int(Tspan/s)
t = np.linspace(0,Tspan,step)
m = 10

A = 0.01*np.mat([[0, 1, 0, 0, 0, 3, 0, 0, 0, 2],
                [1, 0, 1, 0, 0, 2, 3, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 2, 3, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 2, 3, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 2, 3],
                [3, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 3, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 3, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 3, 2, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 3, 0, 0, 0, 0, 0]])

AAA = np.diag(np.array(np.sum(A.T,axis=0))[0])-A
tmpAAA = np.array(AAA)
tmpA = np.array(A)
AAA = np.zeros(((m+1),(m+1)),dtype='float')
A = np.zeros(((m+1),(m+1)),dtype='float')
for i in range(1,(m+1)):
    for j in range(1,(m+1)):
        AAA[i][j]=tmpAAA[i-1][j-1]
        A[i][j]=tmpA[i-1][j-1]

y0 = [0.0,-0.4,-0.3,0.4,-0.5,-0.6,0.3,-0.3,-0.2,0.5,-0.35]
x0 = [0.0,-0.3,0.3,0.6,-0.4,0.2,0.6,-0.3,0.3,0.2,0.25]
yita0 = [0.1]*(m+1)
yCtrl0 = [0.0,-0.4,-0.3,0.4,-0.5,-0.6,0.3,-0.3,-0.2,0.5,-0.35]
xCtrl0 = [0.0,-0.3,0.3,0.6,-0.4,0.2,0.6,-0.3,0.3,0.2,0.25]

a = [1.0]*(m+1)
rou = [0.0,0.76,0.98,0.62,0.76,0.98,0.62,0.76,0.98,0.62,0.76]
T = [0.0]*101
L1 = 0
for i in range(1,100+1):
    if(T[i]==Tspan):
        break
    T[i+1]=i*3.5
    if(T[i]<Tspan and T[i+1]>Tspan):
        T[i+1]=Tspan
        break

L1=len(T)
S = [0.0]*101
for i in range(1,L1):
    S[i]=3.5*(i-1)+2.5 + random.random()*0.1

cntT=2
cntS=1
isctrl = np.zeros(step,dtype='float')
isc = 1
isctrl[1] = 1
isctrl[0] = 1
for i in range(1,step):
    if t[i] >= T[cntT] and t[i-1] <= T[cntT]:
        cntT+=1
        isc=1
    if t[i] >= S[cntS] and t[i - 1] <= S[cntS]:
        cntS+=1
        isc=0
    isctrl[i] = isc

isctrl[step-1]=isctrl[step-2]
cntT=1
ksi = [5]*(m+1)
alpha = [1.0]*(m+1)
ts1=np.zeros((m+1,step))
flag1 = [0.0]*(m+1)
x = np.zeros((step,(m+1)),dtype='float')
x[0]=x0
y = np.zeros((step,(m+1)),dtype='float')
y[0]=y0
ui=1
ux = np.zeros((step,(m+1)),dtype='float')
uy = np.zeros((step,(m+1)),dtype='float')
L = np.zeros((step,(m+1)),dtype='float')
V = np.zeros((step,(m+1)),dtype='float')
yita = np.zeros((step,(m+1)),dtype='float')
xCtrl = np.zeros((step,(m+1)),dtype='float')
xCtrl[0]=xCtrl0
yCtrl = np.zeros((step,(m+1)),dtype='float')
yCtrl[0]=yCtrl0
aa = [21]*(m+1)
kk = np.zeros(((step,(m+1))),dtype='float')
zx = np.zeros((m+1),dtype='float')
zy = np.zeros((m+1),dtype='float')
zxt = np.zeros((m+1),dtype='float')
zyt = np.zeros((m+1),dtype='float')
k0 = [0.0,1.3,1.2,1.4,1.5,1.6,1.7,1.8,1.9,1.25,1.65]
kk[0] = k0
kk[1] = k0
kConst = [0.0,120,90,150,120,90,150,120,90,150,120]
delta = [1.2]*(m+1)
gamma = [0.0,0.2,0.3,0.1,0.4,0.2,0.1,0.3,0.2,0.3,0.1]
sigma = [0.0,0.396,0.306,0.486,0.396,0.306,0.486,0.396,0.306,0.486,0.396]

b = [0.02]*(m+1)
phi = [0.8]*(m+1)
tau = [2.2]*(m+1)
omega1 = [1.9]*(m+1)
omega = np.zeros((step,m+1),dtype='float')
Q = np.zeros((m+1),dtype='float')
E = np.zeros((m+1),dtype='float')
omega2 = [0.9]*(m+1)
thate = [700]*(m+1)
cntts2 = 0
Q1 = [1.03]*(m+1)
Q2 = [1.15]*(m+1)
E1 = [1.25]*(m+1)
E2 = [1.05]*(m+1)
t1 = [0]*int(Tspan/s)
ts2 = [0]*int(Tspan/s)
cntts1 = [0]*(m+1)
c_const1 = [0.75]*(m+1)
c_const2 = [0.95]*(m+1)
c_const = np.zeros((m+1),dtype='float')

def Abs(x):
    if(x>0):
        return x
    return -x

def tao(t):
    return Abs(0.1*math.sin(t))

def tao1(t):
    return Abs(0.1 * math.cos(t))

def fL(nowStep):
    ans = [0.0]*(m+1)
    for i in range(1,(m+1)):
        # ans[i] += (zxt[i]-zx[i])*(zxt[i]-zx[i])
        ans[i] += (zyt[i] - zy[i]) * (zyt[i] - zy[i])
    for i in range(1,m+1):
        ans[i]*=thate[i]
    return ans

def fV(nowStep):
    ans = [0.0]*(m+1)
    for i in range(1,(m+1)):
        ans[i] += sigma[i] * rou[i] * zyt[i] * zyt[i]
    for i in range(1, (m + 1)):
        ans[i] *= thate[i]
    return ans

def caputoEuler(k):
    global cntS,cntT,cntts1,cntts2,ts1,ts2,zxt,zyt,kx,ky,zx,zy,L,V
    N = len(t)
    h = (t[N-1] - t[0])/(N - 1)
    w = (special.rgamma(k) * np.power(h,k) / k) * np.diff(np.power(np.arange(N), k))
    d = (m+1)
    fhistoryY = np.zeros((N - 1, d), dtype=type(y0[0]))
    fhistoryX = np.zeros((N - 1, d), dtype=type(y0[0]))
    fhistoryYita = np.zeros((N - 1, d), dtype=type(y0[0]))
    fhistoryYCtrl = np.zeros((N - 1, d), dtype=type(y0[0]))
    fhistoryXCtrl = np.zeros((N - 1, d), dtype=type(y0[0]))
    fhistoryK = np.zeros((N - 1, d), dtype=type(y0[0]))
    global x,y,yita,xCtrl,yCtrl,flag1
    y[1] = y0
    x[1] = x0
    yita[1] = yita0
    yita[0] = yita0
    yCtrl[1] = yCtrl0
    xCtrl[1] = xCtrl0
    for n in range(1, N-1):
        #z_i
        for i in range(1,(m+1)):
            zx[i]=xCtrl[n][i]
            zy[i]=yCtrl[n][i]

        nowT = n/step*Tspan
        if (isctrl[n]==1.0  and isctrl[n-1]==0.0):
            for i in range(1,m+1):
                flag1[i] = n
                cntts1[i] += 1
                ts1[i][cntts1[i]] = t[n]
                zxt[i] = zx[i]
                zyt[i] = zy[i]

        for i in range(1,m+1):
            omega[n][i]=omega1[i]*math.sin(nowT/2)*math.sin(nowT/2)+omega2[i]*math.cos(nowT/2)*math.cos(nowT/2)
            Q[i] = Q1[i] * math.sin(nowT / 2) * math.sin(nowT / 2) + Q2[i] * math.cos(nowT / 2) * math.cos(nowT / 2)
            E[i] = E1[i] * math.sin(nowT / 2) * math.sin(nowT / 2) + E2[i] * math.cos(nowT / 2) * math.cos(nowT / 2)
            c_const[i] = c_const1[i] * math.sin(nowT / 2) * math.sin(nowT / 2)+c_const2[i] * math.cos(nowT / 2) * math.cos(nowT / 2)
        tmp = int((nowT - tao(nowT)) / s - 1)
        if isctrl[n]==1:
            for i in range(1,(m+1)):
                for j in range(1,(m+1)):
                    ux[n][i] += 0.15*-kk[n][i]*omega[tmp][j] * zxt[i]
                    uy[n][i] += 0.15*-kk[n][i]*omega[tmp][j] * zyt[i]

        tn = n
        yn = y[n]
        xn = x[n]
        yitan = yita[n-1]
        yCtrln = yCtrl[n]
        xCtrln = xCtrl[n]
        kn = kk[n]
        fhistoryY[n] = fy(yn, tn)
        fhistoryX[n] = fx(xn, tn)
        fhistoryYita[n] = fyita(yitan, tn)
        fhistoryYCtrl[n] = fyCtrl(yCtrln, tn)
        fhistoryXCtrl[n] = fxCtrl(xCtrln, tn)
        fhistoryK[n] = fk(kn, tn)
        yita[n] = yita0 + np.dot(w[0:n + 1], fhistoryYita[n::-1])

        L[n] = fL(n)
        V[n] = fV(n)+yita[n]
        # V[n] = fV(n)

        for i in range(1,m+1):
            if L[n][i] >= V[n][i]:
                flag1[i] = n
                cntts1[i] += 1
                ts1[i][cntts1[i]] = t[n]
                zxt[i] = zx[i]
                zyt[i] = zy[i]

        y[n + 1] = y0 + np.dot(w[0:n+1], fhistoryY[n::-1])
        x[n + 1] = x0 + np.dot(w[0:n + 1], fhistoryX[n::-1])
        yCtrl[n + 1] = yCtrl0 + np.dot(w[0:n + 1], fhistoryYCtrl[n::-1])
        xCtrl[n + 1] = xCtrl0 + np.dot(w[0:n + 1], fhistoryXCtrl[n::-1])
        kk[n + 1] = kk[0] + np.dot(w[0:n + 1], fhistoryK[n::-1])

    return (x,y,yita,xCtrl,yCtrl)

# 参数原则上按照matlab代码的参数命名 任何数据下标均从1开始
def fx(u,nowStep):
    nowT = nowStep/step*Tspan
    du_dt = np.zeros((m + 1), dtype='float')
    tmp = int((nowT-tao(nowT))/s-1)
    for i in range(1, (m + 1)):
        for j in range(1, (m + 1)):
            du_dt[i] += (A[i][j] * np.sin(x[tmp][j]))
        du_dt[i] = du_dt[i] - omega[nowStep][i]*x[tmp][i]+Q[i]*y[nowStep-1][i]
    return du_dt

def DIS(ta,tb):
    return np.sqrt(ta*ta+tb*tb)

def fk(u,nowStep):
    du_dt = np.zeros((m + 1), dtype='float')
    if isctrl[nowStep] == False:
        return du_dt
    for i in range(1,(m+1)):
        if (u[i] < kConst[i]):
            du_dt[i] = ksi[i] * DIS(zxt[i], zyt[i]) * DIS(zxt[i], zyt[i]) - (kk[nowStep][i]-2*a[i]-sigma[i]) / 2
        else:
            du_dt[i] = -(kk[nowStep][i]-2*a[i]-sigma[i]) / 2
    return du_dt

def fy(u,nowStep):
    du_dt = np.zeros((m+1),dtype='float')
    for i in range(1,(m+1)):
        du_dt[i]=E[i]*np.sin(x[nowStep-1][i])-c_const[i]*y[nowStep-1][i]
    return du_dt

def fyita(u,nowStep):
    du_dt = np.zeros((m+1), dtype='float')
    for i in range(1,(m+1)):
        du_dt[i] = -delta[i] * u[i] + gamma[i] * (
                    sigma[i] * rou[i] * zyt[i] * zyt[i] - (zyt[i] - zy[i]) * (zyt[i] - zy[i]))
    return du_dt

def fxCtrl(u,nowStep):
    nowT = nowStep / step * Tspan
    du_dt = np.zeros((m + 1), dtype='float')
    tmp = int((nowT - tao(nowT)) / s - 1)
    for i in range(1, (m + 1)):
        for j in range(1, (m + 1)):
            du_dt[i] += (A[i][j] * np.sin(xCtrl[tmp][j]))
        du_dt[i] = du_dt[i] - omega[nowStep][i] * xCtrl[tmp][i] + Q[i] * yCtrl[nowStep - 1][i]
    return du_dt+ux[nowStep]

def fyCtrl(u,nowStep):
    du_dt = np.zeros((m + 1), dtype='float')
    for i in range(1, (m + 1)):
        du_dt[i] = E[i] * np.sin(xCtrl[nowStep - 1][i]) - c_const[i] * yCtrl[nowStep - 1][i]
    return du_dt+uy[nowStep]

def Max(a,b):
    if a >b:
        return a
    return b

def main():
    k = 0.99
    caputoEuler(k)
    kk[0]=kk[1]

    plt.plot(t,isctrl)

    plt.show()
    plt.clf()

    for i in range(1,m+1):
        plt.plot(t,kk[:,i],label=u"$k_{"+str(i)+"}(t)$")
    plt.legend(loc=4)
    plt.legend(ncol=2)
    plt.xlim(0,Tspan)
    plt.xlabel(u"$time$ $t$")
    plt.ylabel(u"$k_i(t)(i=1...10)$")
    plt.show()
    plt.clf()
    for i in range(1,(m+1)):
        plt.plot(t, x[:,i])
        plt.plot(t, y[:, i])
    plt.xlabel(u"$time$ $t$")
    plt.ylabel(u"$x_i(t), \~{x}_i(t)(i=1...10)$")
    plt.xlim(0, Tspan)
    plt.show()
    plt.clf()
    for i in range(1, (m+1)):
        plt.plot(t, xCtrl[:, i])
        plt.plot(t, yCtrl[:, i])
    plt.xlabel(u"$time$ $t$")
    plt.ylabel(u"$x_i(t), \~{x}_i(t)(i=1...10)$")
    plt.xlim(0,Tspan/2)
    plt.show()
    plt.clf()
    for i in range(1, (m+1)):
        plt.plot(t, xCtrl[:, i]*xCtrl[:, i])
        plt.plot(t, yCtrl[:, i]*yCtrl[:, i])
    plt.xlabel(u"$time$ $t$")
    plt.ylabel(u"$|x_i(t)|^2, |\~{x}_i(t)|^2(i=1...10)$")
    plt.xlim(0,Tspan/2)
    plt.show()
    plt.clf()
    dif = np.zeros(int(Tspan/s),dtype='float')
    for i in range(1,(m+1)):
        dif+=np.abs(xCtrl[:, i]-x[:,i])
        dif+=np.abs(yCtrl[:, i]-y[:,i])
    dis = [0]*int(Tspan/s)
    for l in range(1,5+1):
        print("node"+str(l)+":",cntts1[l])
        maxx = 0
        for i in range(2,cntts1[l]):
            dis[i-1]=ts1[l][i]-ts1[l][i-1]
            maxx = Max(dis[i-1],maxx)
        fig = plt.subplot(510+l)
        plt.xlim(0, Tspan)
        plt.ylim(0, maxx+0.2)
        fig.stem(ts1[l][1:cntts1[l]-1],dis[1:cntts1[l]-1])
        fig.set_xlabel(u"$node$ "+str(l),fontsize = 12)
        if l == 3:
            fig.set_ylabel(u"inter-event intervals",fontsize = 12)
        fig.tick_params(labelsize=13)
    plt.subplots_adjust(wspace=0.6, hspace=0.5)
    plt.show()
    plt.clf()
    for l in range(6,10+1):
        print("节点"+str(l)+":",cntts1[l])
        maxx = 0
        for i in range(2,cntts1[l]):
            dis[i-1]=ts1[l][i]-ts1[l][i-1]
            maxx = Max(dis[i - 1], maxx)
        fig = plt.subplot(510+l-5)
        plt.xlim(0, Tspan)
        plt.ylim(0, maxx + 0.2)
        fig.stem(ts1[l][1:cntts1[l]-1],dis[1:cntts1[l]-1])
        fig.set_xlabel(u"$node$ "+str(l),fontsize = 12)
        if l == 8:
            fig.set_ylabel(u"inter-event intervals",fontsize = 12)
        fig.tick_params(labelsize=13)
    plt.subplots_adjust(wspace=0.7, hspace=0.5)

    plt.show()
    plt.clf()

if __name__ == '__main__':
    main()
