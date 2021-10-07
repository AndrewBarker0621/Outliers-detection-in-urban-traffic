import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.DataFrame
data = pd.read_csv("E:\\数据挖掘2020.7\\7.27\\flow\\20111116.csv", header = None)
data.drop(data.columns[0:3], axis=1, inplace=True)
data = data.values
print("used to be 96 lines：")
print(data)

# create new list to store data 
list = np.zeros((data.shape[0], 24), dtype=int, order='C')
# 96 -> 24
for i in range(data.shape[0]):
    k = 0
    for j in range(0, data.shape[1], 4):
        count = 0
        count = data[i, j] + data[i, j+1] + data[i, j+2] + data[i, j+3]
        list[i, k] = count
        k = k + 1
print("24 matrix：")
print(list)


#NMF
def Nmf(V, r, k, e):
    """
    :param V: matrix to be decomposed
    :param r: number of road patterns
    :param k: itteration 
    :param e: threshold
    :return: coefficient matrix, basic mode matrix
    """
    m, n = np.shape(V)
    W = np.mat(np.random.random((m, r)))
    H = np.mat(np.random.random((r, n)))
    ERR = []

    for x in range(k):
        VV = W * H
        E = V - VV
        err = np.sqrt(np.sum(np.square(E)))
        ERR.append(err)
        print(err)

        #err = euclidean_distances(V, VV)
        """
        for i in range(m):
            for j in range(n):
                err += E[i, j] * E[i, j]
        """

        if err < e:
            break

        A = W.T * V
        B = W.T * W * H
        for i1 in range(r):
            for j1 in range(n):
                if B[i1, j1] != 0:
                    H[i1, j1] = H[i1, j1] * A[i1, j1] / B[i1, j1]

        C = V * H.T
        D = W * H * H.T
        for i2 in range(m):
            for j2 in range(r):
                if D[i2, j2] != 0:
                    W[i2, j2] = W[i2, j2] * C[i2, j2] / D[i2, j2]

    return W, H, ERR

W, H , error = Nmf(list, 3, 100, 5000)

print("------------------------------------------")
print(W)
print(W.shape)

"""
print('\n')
print(H)


def Figure():
    L0 = []
    for i in range(96):
        L0.append(H[0, i])
    L1 = []
    for i in range(96):
        L1.append(H[1, i])
    L2 = []
    for i in range(96):
        L2.append(H[2, i])

    x = range(0, 1440, 15)
    y0 = L0
    y1 = L1
    y2 = L2
    plt.figure()

    plt.plot(x, y0)

    plt.plot(x, y1)

    plt.plot(x, y2)
    plt.xticks(x)
    plt.yticks(range(0, 5))
    plt.show()


Figure()
"""