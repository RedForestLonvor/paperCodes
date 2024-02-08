# python@3.9.13
# @redforest

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

# constants

INF = 1e9
m = 12
P1 = 0.75
P2 = 0.4

# constants
# utlis

def Max(a, b):
    if a > b:
        return a
    return b


def MatMax(t1, t2):
    ans = np.zeros((m, m), dtype='float')
    for i in range(0, m):
        for j in range(0, m):
            if t1[i][j] != 0:
                ans[i][j] = t1[i][j]
            if t2[i][j] != 0:
                ans[i][j] = t2[i][j]
    return ans


def create_matrix(nonzero_elements):
    matrix = np.zeros((m, m))
    for indices, value in nonzero_elements.items():
        row, col = indices
        matrix[row - 1, col - 1] = value
    return matrix


def getTime(nowStep):
    return tSpan * (nowStep / totStep)


def laplacian_matrix(B):
    degrees = np.sum(B, axis=1)
    D = np.diag(degrees)
    L = D - B
    return L


def cofactor_matrix(L):
    size = L.shape[0]
    cofactor_mat = np.empty_like(L)
    for i in range(size):
        for j in range(size):
            if not i == j:
                continue
            sub_matrix = np.delete(np.delete(L, i, axis=0), j, axis=1)
            cofactor_mat[i, j] = ((-1) ** (i + j)) * np.linalg.det(sub_matrix)
    return cofactor_mat


# utlis
# parameter

b0 = {(1, 7): 0.025, (6, 7): 0.025, (6, 12): 0.025, (12, 5): 0.025,
      (11, 5): 0.025, (4, 10): 0.025, (10, 3): 0.025, (9, 3): 0.025,
      (2, 9): 0.025, (2, 8): 0.012, (8, 1): 0.012, (11, 4): -0.017}

b1 = {(7, 12): 0.016, (11, 12): 0.016, (6, 1): 0.016, (5, 6): 0.016,
      (1, 5): 0.016, (3, 2): 0.016, (3, 4): 0.016, (4, 2): 0.016,
      (8, 9): 0.023, (10, 9): 0.023, (1, 2): 0.023, (2, 6): 0.023,
      (6, 4): 0.023, (7, 8): 0.023, (10, 11): -0.016}

b0 = create_matrix(b0)
b1 = create_matrix(b1)
B = MatMax(b0, b1)

tSpan = 90 #50
stepLength = 0.02
totStep = int(tSpan / stepLength)
Laplace = laplacian_matrix(B)
cofactors = cofactor_matrix(Laplace)
C = np.zeros(m + 1)
C[1:] = np.diag(cofactors)
tau = 0.1
N1 = {1, 5, 6, 7, 11, 12}
N2 = {2, 3, 4, 8, 9, 10}
gamma = 0.75
a_k = 1.1
startStep = int(tau / stepLength)
Psi = [0,0.1,0.01]
K_1 = [0, 0.05, 0.05]
H = 1.18
beta = -0.76

# parameter
# variable

PTS = 0  # preTriggerStep
cntTrigger = 0
V0 = np.zeros((m + 1), dtype='float')  # time start from 0 , node start from 1
X = np.zeros((2+1, totStep, m + 1), dtype='float')
XCtrl = np.zeros((2+1, totStep, m + 1), dtype='float')
X0 = np.zeros((2+1, totStep), dtype='float')
Z = np.zeros((2+1,totStep, m + 1), dtype='float')
fhistoryX = np.zeros((2+1, totStep, m + 1), dtype='float')
fhistoryX0 = np.zeros((2+1, totStep), dtype='float')
fhistoryXCtrl = np.zeros((2+1, totStep, m + 1), dtype='float')
t = np.linspace(0, tSpan, totStep)
triggerStep = np.zeros((totStep), dtype='float')
U = np.zeros((2+1, m + 1), dtype='float')
atk = np.zeros((2+1,totStep),dtype='int')

# variable
# initial value
vs0 = 0.0
for nowStep in range(0, startStep + 1):
    X0[1][nowStep] = 0.1*(np.cos(0.1) + np.sin(0.01 * nowStep))
    X0[2][nowStep] = 0.1*(np.sin(0.2) + np.sin(0.01 * nowStep))
for r in range(1, m + 1):
    V0[r] = -INF
    for nowStep in range(0, startStep + 1):
        X[1][nowStep][r] = 1 + np.cos(r) - np.cos(0.01 * nowStep) + int(r in N1)-0.5
        X[2][nowStep][r] = 1 + np.sin(r) - np.cos(0.01 * nowStep) + int(r in N1)-0.5
        XCtrl[1][nowStep][r] = 1 + np.cos(r) - np.cos(0.01 * nowStep) + int(r in N1)-0.5
        XCtrl[2][nowStep][r] = 1 + np.sin(r) - np.cos(0.01 * nowStep) + int(r in N1)-0.5
        X[1][nowStep][r] *= 0.3
        X[2][nowStep][r] *= 0.3
        XCtrl[1][nowStep][r] *= 0.3
        XCtrl[2][nowStep][r] *= 0.3
        if nowStep == 0:
            if r in N1:
                tmp = 1
            else:
                tmp = -1
            Z[1][nowStep][r] = XCtrl[1][nowStep][r] + tmp * X0[1][nowStep]
            Z[2][nowStep][r] = XCtrl[2][nowStep][r] + tmp * X0[2][nowStep]
        if r in N1:
            V0[r] = max(V0[r], ((X[1][nowStep][r] - X0[1][nowStep]) * (X[1][nowStep][r] - X0[1][nowStep])
                                + (X[2][nowStep][r] - X0[2][nowStep]) * (X[2][nowStep][r] - X0[2][nowStep])))
        else:
            V0[r] = max(V0[r], ((X[1][nowStep][r] + X0[1][nowStep]) * (X[1][nowStep][r] + X0[1][nowStep]) +
                                (X[2][nowStep][r] + X0[2][nowStep]) * (X[2][nowStep][r] + X0[2][nowStep])))
    vs0 += C[r] * V0[r]
vs0 *= np.exp(gamma*tau)

# initial value

def OMEGA(nowStep):
    ans = 0.0
    for r in range(1, m + 1):
        if r in N1:
            ans += np.abs(C[r]) * (
                        (XCtrl[1][nowStep][r] - X0[1][nowStep]) * (XCtrl[1][nowStep][r] - X0[1][nowStep])
                        + (XCtrl[2][nowStep][r] - X0[2][nowStep]) * (XCtrl[2][nowStep][r] - X0[2][nowStep]))
        else :
            ans += np.abs(C[r]) * (
                    (XCtrl[1][nowStep][r] + X0[1][nowStep]) * (XCtrl[1][nowStep][r] + X0[1][nowStep])
                    + (XCtrl[2][nowStep][r] + X0[2][nowStep]) * (XCtrl[2][nowStep][r] + X0[2][nowStep]))
    vs = 0.0
    for r in range(1, m + 1):
        if r in N1:
            vs += np.abs(C[r]) * (
                (XCtrl[1][PTS+1][r]-X0[1][PTS+1]) * (XCtrl[1][PTS+1][r]-X0[1][PTS+1])
                + (XCtrl[2][PTS+1][r]-X0[2][PTS+1]) * (XCtrl[2][PTS+1][r]-X0[2][PTS+1]))
        else :
            vs += np.abs(C[r]) * (
                (XCtrl[1][PTS+1][r]+X0[1][PTS+1]) * (XCtrl[1][PTS+1][r]+X0[1][PTS+1])
                + (XCtrl[2][PTS+1][r]+X0[2][PTS+1]) * (XCtrl[2][PTS+1][r]+X0[2][PTS+1]))
    vs *= np.exp(gamma * getTime(PTS))
    ans -= Max(vs, vs0) * np.exp(a_k - gamma * getTime(nowStep))
    return ans

def delay(time):
    ans = 0.01*np.cos(time)*np.cos(time)
    return ans/stepLength

def delay1(time):
    ans = 0.01*np.abs(np.sin(time))
    return ans/stepLength

def delay2(time):
    ans = 0.01 * np.cos(time)*np.cos(time)
    return ans/stepLength


def fX01(nowStep):
    du_dt = X0[2][nowStep-1]
    return du_dt


def fX02(nowStep):
    du_dt = -Psi[1] * np.sin(X0[1][nowStep-1-int(delay(getTime(nowStep)))])-Psi[2] * X0[2][nowStep-1]
    return du_dt


def fX1(nowStep):
    du_dt = np.zeros((m + 1), dtype='float')
    for i in range(1,m+1):
        du_dt[i] = X[2][nowStep-1][i]# + f1(nowStep-1-int(delay(getTime(nowStep))),i) : equal to 0
        for j in range(1,m+1):
            du_dt[i] -= np.abs(b0[i-1][j-1])*(X[1][nowStep-1][i] - np.sign(b0[i-1][j-1])*X[1][nowStep-1][j])
        for j in range(1, m + 1):
            du_dt[i] -= np.abs(b1[i - 1][j - 1]) * (
                        X[1][nowStep-1 - int(delay1(getTime(nowStep)))][i] - np.sign(b1[i - 1][j - 1]) * X[1][nowStep-1 - int(delay1(getTime(nowStep)))][j])
    return du_dt


def fX2(nowStep):
    du_dt = np.zeros((m + 1), dtype='float')
    for i in range(1, m + 1):
        du_dt[i] = -Psi[2] * X[2][nowStep-1][i] + -Psi[1] * np.sin(X[1][nowStep-int(delay(getTime(nowStep)))][i])
        for j in range(1, m + 1):
            du_dt[i] -= np.abs(b0[i - 1][j - 1]) * (
                        X[2][nowStep - 1][i] - np.sign(b0[i - 1][j - 1]) * X[2][nowStep - 1][j])
        for j in range(1, m + 1):
            du_dt[i] -= np.abs(b1[i - 1][j - 1]) * (
                    X[2][nowStep-1 - int(delay1(getTime(nowStep)))][i] - np.sign(b1[i - 1][j - 1]) *
                    X[2][nowStep-1 - int(delay1(getTime(nowStep)))][j])
    return du_dt


def fX1Ctrl(nowStep):
    du_dt = np.zeros((m + 1), dtype='float')
    for i in range(1,m+1):
        du_dt[i] = XCtrl[2][nowStep-1][i]# + f1(nowStep-1-int(delay(getTime(nowStep))),i) : equal to 0
        for j in range(1,m+1):
            du_dt[i] -= np.abs(b0[i-1][j-1])*(X[1][nowStep-1][i] - np.sign(b0[i-1][j-1])*X[1][nowStep-1][j])
        for j in range(1, m + 1):
            du_dt[i] -= np.abs(b1[i - 1][j - 1]) * (
                        X[1][nowStep-1 - int(delay1(getTime(nowStep)))][i] - np.sign(b1[i - 1][j - 1]) * X[1][nowStep-1 - int(delay1(getTime(nowStep)))][j])
    return du_dt+U[1]


def fX2Ctrl(nowStep):
    du_dt = np.zeros((m + 1), dtype='float')
    for i in range(1, m + 1):
        du_dt[i] = -Psi[2] * XCtrl[2][nowStep-1][i] + -Psi[1] * np.sin(XCtrl[1][nowStep-int(delay(getTime(nowStep)))][i])
        for j in range(1, m + 1):
            du_dt[i] -= np.abs(b0[i - 1][j - 1]) * (
                        X[2][nowStep - 1][i] - np.sign(b0[i - 1][j - 1]) * X[2][nowStep - 1][j])
        for j in range(1, m + 1):
            du_dt[i] -= np.abs(b1[i - 1][j - 1]) * (
                    X[2][nowStep-1 - int(delay1(getTime(nowStep)))][i] - np.sign(b1[i - 1][j - 1]) *
                    X[2][nowStep-1 - int(delay1(getTime(nowStep)))][j])
    return du_dt+U[2]


def z(nowStep,num,alpha,GAMMA):
    ans = np.zeros(m+1)
    for r in range(1,m+1):
        ans[r] = (1 - alpha) * ((1 - GAMMA) * Z[num][nowStep][r] + GAMMA * H * Z[num][nowStep][r])
    return ans


def Etk(nowStep,num, lam, alpha, GAMMA):
    etk = np.zeros((m+1),dtype=float)
    ztk = z(nowStep, num, alpha, GAMMA)
    for r in range(1,m+1):
        etk[r] = (beta * ztk[r]) / special.rgamma(lam + 1) + Z[num][PTS][r]
    return etk


def caputoEuler(k):
    # TODO : define variables
    global cntTrigger, atk, PPTS, PTS, Z, X, X0, U, fhistoryX0, fhistoryX, fhistoryXCtrl, XCtrl, triggerStep
    N = len(t)
    h = (t[N - 1] - t[0]) / (N - 1)
    w = (special.rgamma(k) * np.power(h, k) / k) * np.diff(np.power(np.arange(N), k))
    curStartStep = startStep
    # time start from startTime+1 to maxTime // n
    for nowStep in range(startStep, totStep - 1):
        if nowStep % 400 == 0:
            print(nowStep / totStep * 100 ,"%")
        # TODO : judge if nowStep is triggered
        if OMEGA(nowStep) >= 0:
            cntTrigger += 1
            triggerStep[cntTrigger] = getTime(nowStep)
            PTS = nowStep
            curStartStep = nowStep+1
        alpha = 0
        GAMMA = 0
        if PTS == nowStep:
            alpha = np.random.binomial(n=1, p=P1)
            # alpha = 0
            GAMMA = np.random.binomial(n=1, p=P2)
            # print(alpha,GAMMA)
        # TODO : calculate $Dx_{01}(t),Dx_{02}(t),Dx_{03}(t)$ of nowStep
        fhistoryX0[1][nowStep] = fX01(nowStep)  # return number
        fhistoryX0[2][nowStep] = fX02(nowStep)
        # print(fX01(nowStep))
        # TODO : calculate $Dx_{k1}(t),Dx_{k2}(t),Dx_{k3}(t)$ of nowStep
        fhistoryX[1][nowStep] = fX1(nowStep)  # return vector m+1
        fhistoryX[2][nowStep] = fX2(nowStep)
        # TODO : calculate $Dx_{k1}(t),Dx_{k2}(t),Dx_{k3}(t)$ haveCtrl
        if PTS != nowStep:
            fhistoryXCtrl[1][nowStep] = fX1Ctrl(nowStep)  # return vector m+1
            fhistoryXCtrl[2][nowStep] = fX2Ctrl(nowStep)

        # TODO : calculate primitive of nowStep
        for i in range(1, 2 + 1):
            if PTS == nowStep:
                for r in range(1,m+1):
                    if r in N1:
                        tmp = 1
                    else:
                        tmp = -1
                    Z[i][nowStep][r] = XCtrl[i][nowStep][r] - tmp * X0[i][nowStep]
                atk[i][nowStep] = 10 * alpha + GAMMA  # encode
                etk = Etk(nowStep, i, k, alpha, GAMMA)
            for r in range(1,m+1):
                X0[i][nowStep + 1] = X0[i][startStep] + np.dot(w[startStep-startStep:nowStep-startStep + 1],
                                                                  fhistoryX0[i][nowStep:startStep - 1:-1])
                X[i][nowStep + 1] = X[i][startStep] + np.dot(w[startStep-startStep:nowStep-startStep + 1],
                                                                fhistoryX[i][nowStep:startStep - 1:-1])
                if PTS != nowStep:
                    XCtrl[i][nowStep + 1] = XCtrl[i][curStartStep] + np.dot(w[curStartStep-curStartStep:nowStep-curStartStep + 1],
                                                                            fhistoryXCtrl[i][nowStep:curStartStep - 1:-1])
                else:
                    if r in N1:
                        XCtrl[i][nowStep + 1][r] = etk[r] + X0[i][nowStep+1]# return vector m+1
                        # if getTime(nowStep) > 43:
                        #     print("***",XCtrl[i][nowStep + 1][1]-X0,XCtrl[i][nowStep][1])
                    else :
                        XCtrl[i][nowStep + 1][r] = etk[r] - X0[i][nowStep+1]
        # TODO : calculate $u_k(t)$ of nowStep
        # must calculate here
        #遇到不连续的点的时候需要注意边界情况。具体情况可以参考一下这个地方遇到的。要在跳跃间断点跳跃了之后再计算误差
        if PTS == nowStep:
            for r in range(1, m + 1):
                if r in N1:
                    tmp = 1
                else:
                    tmp = -1
                Z[1][nowStep+1][r] = XCtrl[1][nowStep+1][r] - tmp * X0[1][nowStep+1]
                Z[2][nowStep+1][r] = XCtrl[2][nowStep+1][r] - tmp * X0[2][nowStep+1]
            for i in range(1, 2+1):
                U[i] = -K_1[i] * z(PTS+1,i, alpha, GAMMA)

    # return (X0,X,XCtrl)


def main():
    color0 = [0, (0.8, 0, 0.8, 1), (0.1, 0.1, 0.1, 0.9), (0, 0, 1, 0.8)]
    # colors_base = [
    #     (0.8, 0.2, 0.2, 0.7),
    #     (0.2, 0.8, 0.2, 0.7),
    #     (0.2, 0.2, 0.8, 0.7)
    # ]

    # color_matrix = []
    # for color_base in colors_base:
    #     color_class = []
    #     for _ in range(20):
    #         color = [min(max(c + np.random.uniform(-0.35, 0.35), 0), 1) for c in color_base]
    #         color_class.append(tuple(color))
    #     color_matrix.append(color_class)

    # color_matrix = np.array(color_matrix)

    soft_blue = '#1971ae'
    k = 0.98
    caputoEuler(k)

    diff = np.zeros((m + 1, totStep))
    for A in range(1, 2 + 1):
        for i in range(1, (m + 1)):
            if i in N1:
                diff[i] = XCtrl[A][:, i] - X0[A][:]
            else:
                diff[i] = XCtrl[A][:, i] + X0[A][:]
            plt.plot(t, diff[i], linewidth=0.99)

        for i in range(0, totStep - 1):
            if atk[A][i] == 1 or atk[A][i] == 11:
                plt.scatter(getTime(i), 1.4, color='red', marker='*', label='Dec Attack')
            if atk[A][i] == 10:
                plt.scatter(getTime(i), 1.2, color='green', marker='*', label='DoS Attack')
        handles, labels = plt.gca().get_legend_handles_labels()
        num = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[1]:
                num = i
                break
    filtered_handles = [handles[1], handles[num]]
    filtered_labels = [labels[1], labels[num]]
    plt.legend(filtered_handles, filtered_labels, loc='lower right')
    plt.xlabel(u"$Time t$")
    plt.ylabel(u"e$_{i}(t)$,i=1,2...20")
    plt.xlim(0, tSpan)
    plt.grid(True)
    plt.savefig('eit_haveU.eps', format='eps')
    plt.savefig('eit_haveU.png', format='png')
    plt.show()
    plt.clf()

    for A in range(1, 2 + 1):
        for i in range(1, (m + 1)):
            if i in N1:
                diff[i] = (X[A][:, i] - X0[A][:])
            else:
                diff[i] = (X[A][:, i] + X0[A][:])
            plt.plot(t, diff[i], linewidth=0.95)
    plt.xlabel(u"$Time t$")
    plt.ylabel(u"e$_{i}(t)$,i=1,2...20")
    plt.xlim(0, tSpan)
    plt.grid(True)
    plt.savefig('eit_noU.eps', format='eps')
    plt.savefig('eit_noU.png', format='png')
    plt.show()
    plt.clf()
    maxx = 0
    for A in range(1, 2 + 1):
        for i in range(1, (m + 1)):
            if i in N1:
                diff[i] = (XCtrl[A][:, i] - X0[A][:]) * (XCtrl[A][:, i] - X0[A][:])
            else:
                diff[i] = (XCtrl[A][:, i] + X0[A][:]) * (XCtrl[A][:, i] + X0[A][:])
            plt.plot(t, diff[i], linewidth=0.95)
            maxx = max(maxx,np.max(diff[i]))

        for i in range(0, totStep - 1):
            if atk[A][i] == 1 or atk[A][i] == 11:
                plt.scatter(getTime(i), maxx + 0.1, color='red', marker='*', label='Dec Attack')
            if atk[A][i] == 10:
                plt.scatter(getTime(i), maxx + 0.3, color='green', marker='*', label='DoS Attack')
        handles, labels = plt.gca().get_legend_handles_labels()
        num = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[1]:
                num = i
                break
    filtered_handles = [handles[1], handles[num]]
    filtered_labels = [labels[1], labels[num]]
    plt.legend(filtered_handles, filtered_labels, loc='center right')
    plt.xlabel(u"$Time t$")
    plt.ylabel(u"E|e$_{i}(t)|^2$,i=1,2...20")
    plt.ylim(-0.05, maxx + 0.5)
    plt.xlim(0, tSpan)
    plt.grid(True)
    plt.savefig('ee.eps', format='eps')
    plt.savefig('ee.png', format='png')
    plt.show()
    plt.clf()

    dis = [0] * cntTrigger
    for i in range(2, cntTrigger):
        dis[i - 1] = triggerStep[i] - triggerStep[i - 1]
    plt.vlines(triggerStep[1:cntTrigger - 1], 0, dis[1:cntTrigger - 1], linestyles='-', linewidths=0.9,
               colors=soft_blue)
    plt.plot(triggerStep[1:cntTrigger - 1], dis[1:cntTrigger - 1], 'o', markerfacecolor='none',
             markeredgecolor=soft_blue)
    plt.grid(True)
    plt.xlabel(u"$Time t$")
    plt.ylabel(u"Inter-triggered intervals")
    plt.ylim(0, max(dis) + 0.2)
    plt.xlim([0, tSpan])
    plt.savefig('Inter-triggered intervals.eps', format='eps')
    plt.savefig('Inter-triggered intervals.png', format='png')
    plt.show()

    for A in range(1, 2 + 1):
        for i in range(1, (m + 1)):
            plt.plot(t, X[A][:, i], linewidth=0.97)
        plt.plot(t, X0[A][:], linestyle='--', color = color0[A], linewidth = 1.7)
    plt.xlabel(u"$Time t$")
    plt.ylabel(u"x$(t)$ and y$_{i}(t)$,i=1,2...20")
    plt.grid(True)
    plt.xlim(0, tSpan)
    plt.savefig('x.eps', format='eps')
    plt.savefig('x.png', format='png')
    plt.show()
    plt.clf()

    maxx = 0
    for A in range(1, 2 + 1):
        for i in range(1, (m + 1)):
            plt.plot(t, XCtrl[A][:, i], linewidth=0.97)
            maxx = max(maxx,np.max(XCtrl[A][:, i]))
        plt.plot(t, X0[A][:], linestyle='--', color = color0[A], linewidth = 1.7)
        for i in range(0, totStep - 1):
            if atk[A][i] == 1 or atk[A][i] == 11:
                plt.scatter(getTime(i), maxx + 0.3, color='red', marker='*', label='Dec Attack')
            if atk[A][i] == 10:
                plt.scatter(getTime(i), maxx + 0.1, color='green', marker='*', label='DoS Attack')
        handles, labels = plt.gca().get_legend_handles_labels()
        num = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[1]:
                num = i
                break
    filtered_handles = [handles[1], handles[num]]
    filtered_labels = [labels[1], labels[num]]
    plt.legend(filtered_handles, filtered_labels, loc='lower right')
    plt.xlabel(u"$Time t$")
    plt.ylabel(u"x$(t)$ and y$_{i}(t)$,i=1,2...20")
    plt.grid(True)
    plt.xlim(0, tSpan)
    plt.savefig('xCtrl.eps', format='eps')
    plt.savefig('xCtrl.png', format='png')
    plt.show()
    plt.clf()

if __name__ == '__main__':
    main()