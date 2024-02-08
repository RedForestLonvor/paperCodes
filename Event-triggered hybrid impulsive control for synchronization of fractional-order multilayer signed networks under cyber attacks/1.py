# python@3.9.13
# @redforest

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

# constants

INF = 1e9
m = 20
P = 0.4

# constants
# utlis

def Max(a, b):
    if a > b:
        return a
    return b


def MatMax(t1, t2, t3):
    ans = np.zeros((m, m), dtype='float')
    for i in range(0, m):
        for j in range(0, m):
            if t1[i][j] != 0:
                ans[i][j] = t1[i][j]
            if t2[i][j] != 0:
                ans[i][j] = t2[i][j]
            if t3[i][j] != 0:
                ans[i][j] = t3[i][j]
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

b0 = {(12, 4): 0.015, (4, 11): 0.015, (11, 3): 0.015, (3, 10): 0.015,
(10, 2): 0.015, (2, 9): 0.015, (9, 1): 0.015, (18, 17): 0.015,
(16, 8): 0.022, (8, 15): 0.022, (15, 7): 0.022, (7, 14): 0.022,
(14, 6): 0.022, (6, 13): 0.022, (13, 5): 0.022, (20, 19): 0.022,
(1, 16): -0.017, (17, 20): -0.017, (19, 18): -0.017, (5, 12): -0.017}

b1 = {(1, 2): 0.024, (2, 3): 0.024, (3, 4): 0.024, (3, 17): 0.024,
(5, 6): 0.036, (6, 7): 0.036, (7, 8): 0.036, (7, 19): 0.036,
(1, 20): -0.025, (8, 1): -0.025, (4, 5): -0.025, (5, 18): -0.025}

b2 = {(9, 10): 0.036, (10, 11): 0.036, (11, 12): 0.036, (17, 3): 0.036,
(13, 14): 0.043, (14, 15): 0.043, (15, 16): 0.043, (19, 7): 0.043,
(16, 9): -0.02, (20, 1): -0.02, (19, 17): -0.02, (17, 19): -0.02,(18, 5): -0.02}

b0 = create_matrix(b0)
b1 = create_matrix(b1)
b2 = create_matrix(b2)
B = MatMax(b0, b1, b2)
tSpan = 30 #50
stepLength = 0.05
totStep = int(tSpan / stepLength)
tau = 0.05
K_1 = [0, 0.4, 0.43, 0.5]
H = 1.2
dp = 0
Laplace = laplacian_matrix(B)
cofactors = cofactor_matrix(Laplace)
C = np.zeros(m + 1)
C[1:] = np.diag(cofactors)
N1 = {1, 2, 3, 4, 9, 10, 11, 12, 17, 18}
N2 = {5, 6, 7, 8, 13, 14, 15, 16, 19, 20}
gamma = 0.75
a_k = 1.1
startStep = int(tau / stepLength)
R = 1.4
C_1 = 0.08
C_2 = 1.2
m_1 = -0.6
m_0 = -0.3
R_0 = 0.43
L = 1.3
beta = -0.6

# parameter
# variable

PTS = 0  # preTriggerStep
cntTrigger = 0
V0 = np.zeros((m + 1), dtype='float')  # time start from 0 , node start from 1
X = np.zeros((3 + 1, totStep, m + 1), dtype='float')
XCtrl = np.zeros((3 + 1, totStep, m + 1), dtype='float')
X0 = np.zeros((3 + 1, totStep), dtype='float')
Z = np.zeros((3 + 1,totStep, m + 1), dtype='float')
fhistoryX = np.zeros((3 + 1, totStep, m + 1), dtype='float')
fhistoryX0 = np.zeros((3 + 1, totStep), dtype='float')
fhistoryXCtrl = np.zeros((3 + 1, totStep, m + 1), dtype='float')
t = np.linspace(0, tSpan, totStep)
triggerStep = np.zeros((totStep), dtype='float')
U = np.zeros((3 + 1, m + 1), dtype='float')
atk = np.zeros((3+1,totStep),dtype='int')

# variable
# initial value
vs0 = 0.0
for nowStep in range(0, startStep + 1):
    X0[1][nowStep] = np.cos(0.1) + np.sin(0.01 * nowStep)
    X0[2][nowStep] = np.sin(0.2) + np.sin(0.01 * nowStep)
    X0[3][nowStep] = np.cos(0.3) + np.sin(0.01 * nowStep)
for r in range(1, m + 1):
    V0[r] = -INF
    for nowStep in range(0, startStep + 1):
        X[1][nowStep][r] = 1 + np.cos(r) - np.cos(0.01 * nowStep)
        X[2][nowStep][r] = 1 + np.sin(r) - np.cos(0.01 * nowStep)
        X[3][nowStep][r] = -np.cos(r) - np.sin(0.01 * nowStep)
        XCtrl[1][nowStep][r] = 1 + np.cos(r) - np.cos(0.01 * nowStep)
        XCtrl[2][nowStep][r] = 1 + np.sin(r) - np.cos(0.01 * nowStep)
        XCtrl[3][nowStep][r] = -np.cos(r) - np.sin(0.01 * nowStep)
        if nowStep == 0:
            if r in N1:
                tmp = 1
            else:
                tmp = -1
            Z[1][nowStep][r] = XCtrl[1][nowStep][r] + tmp * X0[1][nowStep]
            Z[2][nowStep][r] = XCtrl[2][nowStep][r] + tmp * X0[2][nowStep]
            Z[3][nowStep][r] = XCtrl[3][nowStep][r] + tmp * X0[3][nowStep]
        if r in N1:
            V0[r] = max(V0[r], ((X[1][nowStep][r] - X0[1][nowStep]) * (X[1][nowStep][r] - X0[1][nowStep])
                                + (X[2][nowStep][r] - X0[2][nowStep]) * (X[2][nowStep][r] - X0[2][nowStep])
                                + (X[3][nowStep][r] - X0[3][nowStep]) * (X[3][nowStep][r] - X0[3][nowStep])))
        else:
            V0[r] = max(V0[r], ((X[1][nowStep][r] + X0[1][nowStep]) * (X[1][nowStep][r] + X0[1][nowStep]) +
                                (X[2][nowStep][r] + X0[2][nowStep]) * (X[2][nowStep][r] + X0[2][nowStep]) +
                                (X[3][nowStep][r] + X0[3][nowStep]) * (X[3][nowStep][r] + X0[3][nowStep])))
    vs0 += C[r] * V0[r]
vs0 *= np.exp(gamma*tau)

# initial value

def OMEGA(nowStep):
    ans = 0.0
    for r in range(1, m + 1):
        if r in N1:
            ans += np.abs(C[r]) * (
                        (XCtrl[1][nowStep][r] - X0[1][nowStep]) * (XCtrl[1][nowStep][r] - X0[1][nowStep])
                        + (XCtrl[2][nowStep][r] - X0[2][nowStep]) * (XCtrl[2][nowStep][r] - X0[2][nowStep])
                        + (XCtrl[3][nowStep][r] - X0[3][nowStep]) * (XCtrl[3][nowStep][r] - X0[3][nowStep]))
        else :
            ans += np.abs(C[r]) * (
                    (XCtrl[1][nowStep][r] + X0[1][nowStep]) * (XCtrl[1][nowStep][r] + X0[1][nowStep])
                    + (XCtrl[2][nowStep][r] + X0[2][nowStep]) * (XCtrl[2][nowStep][r] + X0[2][nowStep])
                    + (XCtrl[3][nowStep][r] + X0[3][nowStep]) * (XCtrl[3][nowStep][r] + X0[3][nowStep]))
    vs = 0.0
    for r in range(1, m + 1):
        if r in N1:
            vs += np.abs(C[r]) * (
                (XCtrl[1][PTS+1][r]-X0[1][PTS+1]) * (XCtrl[1][PTS+1][r]-X0[1][PTS+1])
                + (XCtrl[2][PTS+1][r]-X0[2][PTS+1]) * (XCtrl[2][PTS+1][r]-X0[2][PTS+1])
                + (XCtrl[3][PTS+1][r]-X0[3][PTS+1]) * (XCtrl[3][PTS+1][r]-X0[3][PTS+1]))
        else :
            vs += np.abs(C[r]) * (
                (XCtrl[1][PTS+1][r]+X0[1][PTS+1]) * (XCtrl[1][PTS+1][r]+X0[1][PTS+1])
                + (XCtrl[2][PTS+1][r]+X0[2][PTS+1]) * (XCtrl[2][PTS+1][r]+X0[2][PTS+1])
                + (XCtrl[3][PTS+1][r]+X0[3][PTS+1]) * (XCtrl[3][PTS+1][r]+X0[3][PTS+1]))
    vs *= np.exp(gamma * getTime(PTS))
    ans -= Max(vs, vs0) * np.exp(a_k - gamma * getTime(nowStep))
    return ans


def F(x):
    return 0.1 * m_0 * x + 1 / 2 * (m_1 - m_0) * (np.abs(x + 1) - np.abs(x - 1))  # change

def delay(time):
    ans = 0.01*np.cos(time)*np.cos(time)
    return ans/stepLength

def delay1(time):
    ans = 0.01 * np.abs(np.sin(time))
    return ans/stepLength

def delay2(time):
    ans = 0.01*np.sin(time)*np.sin(time)
    return ans/stepLength

def fX01(nowStep):
    du_dt = -1 / (R * C_1) * X0[1][nowStep] + 1 / (R * C_1) * X0[2][nowStep] - 1 / C_1 * F(X0[1][nowStep-int(delay(getTime(nowStep)))])
    return du_dt


def fX02(nowStep):
    du_dt = 1 / (R * C_2) * X0[1][nowStep] - 1 / (R * C_2) * X0[2][nowStep] + 1 / C_2 * X0[3][nowStep]
    return du_dt


def fX03(nowStep):
    du_dt = -1 / L * X0[2][nowStep] - R_0 / L * X0[3][nowStep]
    return du_dt


def fX1(nowStep):
    du_dt = np.zeros((m + 1), dtype='float')
    for k in range(1, (m + 1)):
        du_dt[k] = -1 / (R * C_1) * X[1][nowStep][k] + 1 / (R * C_1) * X[2][nowStep][k] - 1 / C_1 * F(X[1][nowStep-int(delay(getTime(nowStep)))][k])
        for h in range(1, m + 1):
            du_dt[k] -= (np.abs(b0[k - 1][h - 1]) * (X[1][nowStep][k] - np.sign(b0[k - 1][h - 1]) * X[1][nowStep][h]))
        for h in range(1, m + 1):
            du_dt[k] -= (np.abs(b1[k - 1][h - 1]) * (
                    X[1][nowStep - -int(delay1(getTime(nowStep)))][k] - np.sign(b1[k - 1][h - 1]) * X[1][nowStep - -int(delay1(getTime(nowStep)))][h]))
            du_dt[k] -= (np.abs(b2[k - 1][h - 1]) * (
                    X[1][nowStep - -int(delay2(getTime(nowStep)))][k] - np.sign(b2[k - 1][h - 1]) * X[1][nowStep - -int(delay2(getTime(nowStep)))][h]))
    return du_dt


def fX2(nowStep):
    du_dt = np.zeros((m + 1), dtype='float')
    for k in range(1, (m + 1)):
        du_dt[k] = 1 / (R * C_2) * X[1][nowStep][k] - 1 / (R * C_2) * X[2][nowStep][k] + 1 / C_2 * X[3][nowStep][k]
        for h in range(1, m + 1):
            du_dt[k] -= (np.abs(b0[k - 1][h - 1]) * (
                    X[2][nowStep][k] - np.sign(b0[k - 1][h - 1]) * X[2][nowStep][h]))
        for h in range(1, m + 1):
            du_dt[k] -= (np.abs(b1[k - 1][h - 1]) * (
                    X[2][nowStep - int(delay1(getTime(nowStep)))][k] - np.sign(b1[k - 1][h - 1]) * X[2][nowStep - int(delay1(getTime(nowStep)))][h]))
            du_dt[k] -= (np.abs(b2[k - 1][h - 1]) * (
                    X[2][nowStep - int(delay2(getTime(nowStep)))][k] - np.sign(b2[k - 1][h - 1]) * X[2][nowStep - int(delay2(getTime(nowStep)))][h]))
    return du_dt


def fX3(nowStep):
    du_dt = np.zeros((m + 1), dtype='float')
    for k in range(1, m + 1):
        du_dt[k] = -1 / L * X[2][nowStep][k] - R_0 / L * X[3][nowStep][k]
        for h in range(1, m + 1):
            du_dt[k] -= (np.abs(b0[k - 1][h - 1]) * (X[3][nowStep][k] - np.sign(b0[k - 1][h - 1]) * X[3][nowStep][h]))
        for h in range(1, m + 1):
            du_dt[k] -= (np.abs(b1[k - 1][h - 1]) * (
                    X[3][nowStep - dp][k] - np.sign(b1[k - 1][h - 1]) * X[3][nowStep][h]))
            du_dt[k] -= (np.abs(b2[k - 1][h - 1]) * (
                    X[3][nowStep - dp][k] - np.sign(b2[k - 1][h - 1]) * X[3][nowStep][h]))
    return du_dt


def fX1Ctrl(nowStep):
    du_dt = np.zeros((m + 1), dtype='float')
    for k in range(1, (m + 1)):
        du_dt[k] = -1 / (R * C_1) * XCtrl[1][nowStep][k] + 1 / (R * C_1) * XCtrl[2][nowStep][k] - 1 / C_1 * F(
            XCtrl[1][nowStep][k])
        for h in range(1, m + 1):
            du_dt[k] -= (np.abs(b0[k - 1][h - 1]) * (
                    XCtrl[1][nowStep][k] - np.sign(b0[k - 1][h - 1]) * XCtrl[1][nowStep][h]))
        for h in range(1, m + 1):
            du_dt[k] -= (np.abs(b1[k - 1][h - 1]) * (
                    XCtrl[1][nowStep - dp][k] - np.sign(b1[k - 1][h - 1]) * XCtrl[1][nowStep - dp][h]))
            du_dt[k] -= (np.abs(b2[k - 1][h - 1]) * (
                    XCtrl[1][nowStep - dp][k] - np.sign(b2[k - 1][h - 1]) * XCtrl[1][nowStep - dp][h]))
    return du_dt+U[1]


def fX2Ctrl(nowStep):
    du_dt = np.zeros((m + 1), dtype='float')
    for k in range(1, (m + 1)):
        du_dt[k] = 1 / (R * C_2) * XCtrl[1][nowStep][k] - 1 / (R * C_2) * XCtrl[2][nowStep][k] + 1 / C_2 * \
                   XCtrl[3][nowStep][k]
        for h in range(1, m + 1):
            du_dt[k] -= (np.abs(b0[k - 1][h - 1]) * (
                    XCtrl[2][nowStep][k] - np.sign(b0[k - 1][h - 1]) * XCtrl[2][nowStep][h]))
        for h in range(1, m + 1):
            du_dt[k] -= (np.abs(b1[k - 1][h - 1]) * (
                    XCtrl[2][nowStep - dp][k] - np.sign(b1[k - 1][h - 1]) * XCtrl[2][nowStep - dp][h]))
            du_dt[k] -= (np.abs(b2[k - 1][h - 1]) * (
                    XCtrl[2][nowStep - dp][k] - np.sign(b2[k - 1][h - 1]) * XCtrl[2][nowStep - dp][h]))
    return du_dt+U[2]


def fX3Ctrl(nowStep):
    du_dt = np.zeros((m + 1), dtype='float')
    for k in range(1, m + 1):
        du_dt[k] = -1 / L * XCtrl[2][nowStep][k] - R_0 / L * XCtrl[3][nowStep][k]
        for h in range(1, m + 1):
            du_dt[k] -= (np.abs(b0[k - 1][h - 1]) * (
                    XCtrl[3][nowStep][k] - np.sign(b0[k - 1][h - 1]) * XCtrl[3][nowStep][h]))
        for h in range(1, m + 1):
            du_dt[k] -= (np.abs(b1[k - 1][h - 1]) * (
                    XCtrl[3][nowStep - dp][k] - np.sign(b1[k - 1][h - 1]) * XCtrl[3][nowStep][h]))
            du_dt[k] -= (np.abs(b2[k - 1][h - 1]) * (
                    XCtrl[3][nowStep - dp][k] - np.sign(b2[k - 1][h - 1]) * XCtrl[3][nowStep][h]))
    return du_dt+U[3]


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
            PPTS = PTS
            PTS = nowStep
            curStartStep = nowStep+1
        alpha = 0
        GAMMA = 0
        if PTS == nowStep:
            alpha = np.random.binomial(n=1, p=P)
            # alpha = 0
            GAMMA = np.random.binomial(n=1, p=P)
            # print(alpha,GAMMA)
        # TODO : calculate $Dx_{01}(t),Dx_{02}(t),Dx_{03}(t)$ of nowStep
        fhistoryX0[1][nowStep] = fX01(nowStep)  # return number
        fhistoryX0[2][nowStep] = fX02(nowStep)
        fhistoryX0[3][nowStep] = fX03(nowStep)
        # print(fX01(nowStep))
        # TODO : calculate $Dx_{k1}(t),Dx_{k2}(t),Dx_{k3}(t)$ of nowStep
        fhistoryX[1][nowStep] = fX1(nowStep)  # return vector m+1
        fhistoryX[2][nowStep] = fX2(nowStep)
        fhistoryX[3][nowStep] = fX3(nowStep)
        # TODO : calculate $Dx_{k1}(t),Dx_{k2}(t),Dx_{k3}(t)$ haveCtrl
        if PTS != nowStep:
            fhistoryXCtrl[1][nowStep] = fX1Ctrl(nowStep)  # return vector m+1
            fhistoryXCtrl[2][nowStep] = fX2Ctrl(nowStep)
            fhistoryXCtrl[3][nowStep] = fX3Ctrl(nowStep)

        # TODO : calculate primitive of nowStep
        for i in range(1, 3 + 1):
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
                Z[3][nowStep+1][r] = XCtrl[3][nowStep+1][r] - tmp * X0[3][nowStep+1]
            for i in range(1, 3 + 1):
                U[i] = -K_1[i] * z(PTS+1,i, alpha, GAMMA)

    # return (X0,X,XCtrl)


def main():

    color0 = [0,(0.8, 0, 0.8, 1),(0.1, 0.1, 0.1, 0.9),(0, 0, 1, 0.8)]

    # colors_base = [
    #     (0.8, 0.2, 0.2, 0.7),
    #     (0.2, 0.8, 0.2, 0.7),
    #     (0.2, 0.2, 0.8, 0.7)
    # ]
    #
    # color_matrix = []
    # for color_base in colors_base:
    #     color_class = []
    #     for _ in range(20):
    #         color = [min(max(c + np.random.uniform(-0.35, 0.35), 0), 1) for c in color_base]
    #         color_class.append(tuple(color))
    #     color_matrix.append(color_class)
    #
    # color_matrix = np.array(color_matrix)

    soft_blue = '#1971ae'
    k = 0.998
    caputoEuler(k)

    diff = np.zeros((m + 1, totStep))
    for A in range(1, 3 + 1):
        for i in range(1, (m + 1)):
            if i in N1:
                diff[i] = XCtrl[A][:, i] - X0[A][:]
            else:
                diff[i] = XCtrl[A][:, i] + X0[A][:]
            plt.plot(t, diff[i], linewidth=0.97)

        for i in range(0, totStep - 1):
            if atk[A][i] == 1 or atk[A][i] == 11:
                plt.scatter(getTime(i), 3.6, color='red', marker='*', label='Dec Attack')
            if atk[A][i] == 10:
                plt.scatter(getTime(i), 3.4, color='green', marker='*', label='DoS Attack')
        handles, labels = plt.gca().get_legend_handles_labels()
        num = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[1]:
                num = i
                break
    filtered_handles = [handles[1], handles[num]]
    filtered_labels = [labels[1], labels[num]]
    plt.legend(filtered_handles, filtered_labels, loc='best')
    plt.xlabel(u"$Time t$")
    plt.ylabel(u"e$_{i}(t)$,i=1,2...20")
    plt.xlim(0, tSpan)
    plt.grid(True)
    plt.savefig('eit_haveU.eps', format='eps')
    plt.savefig('eit_haveU.png', format='png')
    plt.show()
    plt.clf()

    for A in range(1, 3 + 1):
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
    for A in range(1, 3 + 1):
        for i in range(1, (m + 1)):
            if i in N1:
                diff[i] = (XCtrl[A][:, i] - X0[A][:]) * (XCtrl[A][:, i] - X0[A][:])
            else:
                diff[i] = (XCtrl[A][:, i] + X0[A][:]) * (XCtrl[A][:, i] + X0[A][:])
            plt.plot(t, diff[i], linewidth=0.95)
            maxx = max(maxx,np.max(diff[i]))
        for i in range(0, totStep - 1):
            if atk[A][i] == 1 or atk[A][i] == 11:
                plt.scatter(getTime(i), maxx + 0.2, color='red', marker='*', label='Dec Attack')
            if atk[A][i] == 10:
                plt.scatter(getTime(i), maxx + 0.4, color='green', marker='*', label='DoS Attack')
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
    plt.ylim(0 , maxx + 0.6)
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
    plt.clf()

    for A in range(1, 3 + 1):
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
    for A in range(1, 3 + 1):
        for i in range(1, (m + 1)):
            plt.plot(t, XCtrl[A][:, i], linewidth=0.97)
            maxx = max(maxx,np.max(XCtrl[A][:, i]))
        plt.plot(t, X0[A][:], linestyle='--', color = color0[A], linewidth = 1.7)
        for i in range(0, totStep - 1):
            if atk[A][i] == 1 or atk[A][i] == 11:
                plt.scatter(getTime(i), maxx+0.5, color= 'red', marker='*', label='Dec Attack')
            if atk[A][i] == 10:
                plt.scatter(getTime(i), maxx+0.3, color='green', marker='*', label='DoS Attack')
        handles, labels = plt.gca().get_legend_handles_labels()
        num = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[1]:
                num = i
                break
    filtered_handles = [handles[1], handles[num]]
    filtered_labels = [labels[1], labels[num]]
    plt.legend(filtered_handles, filtered_labels, loc='lower right')
    plt.xlabel(u"$Time$ $t$")
    plt.ylabel(u"x$(t)$ and y$_{i}(t)$,i=1,2...20")
    plt.grid(True)
    plt.xlim(0, tSpan)
    plt.savefig('xCtrl.eps', format='eps')
    plt.savefig('xCtrl.png', format='png')
    plt.show()
    plt.clf()

if __name__ == '__main__':
    main()
