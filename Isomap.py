import math
import queue

import numpy as np


def dis(p1, p2):
    distant = 0
    for i in range(len(p1)):
        distant += (p1[i]-p2[i])*(p1[i]-p2[i])
    return math.pow(distant, 1/2)

#设置k近邻
def spfa(disFunction = dis, data = [], k = 3):
    #使用spfa计算全员最短路
    dist = []
    #将每个点最近的k个点放入数组
    route = []
    for i in range(len(data)):
        tmpDis = []
        minPts = []
        for j in range(len(data)):
            tmpDis.append(disFunction(data[i], data[j]))
        sort_indices = np.argsort(tmpDis)
        index = 0
        while len(minPts) != k:
            if i != sort_indices[index]:
                minPts.append(sort_indices[index])
            index += 1
        route.append(minPts)
    for i in range(len(data)):
        d = []
        for j in range(len(data)):
            d.append(math.inf)
        dist.append(d)
    for i in range(len(data)):
        vis = [False] * len(data)
        q = queue.Queue()
        vis[i] = True
        dist[i][i] = 0
        #投入位置和距离
        q.put(i)
        while not q.empty():
            tp = q.get()
            vis[tp] = False
            for j in route[tp]:
                if dist[i][tp] + disFunction(data[tp], data[j]) < dist[i][j]:
                    dist[i][j] = dist[i][tp] + disFunction(data[tp], data[j])
                    if not vis[j]:
                        vis[j] = True
                        q.put(j)
    return dist


class Isomap:
    def __init__(self):
        self.data = []
        #dist由迪杰斯特拉或者spfa得到
        self.dist = []
        self.dim = 0
        self.output = []

    def Train(self, data = [], dim = 2, disFunction = dis, k = 3,  routeFunction = spfa):
        self.dim = dim
        self.data = data
        self.dist = routeFunction(disFunction = disFunction, data = self.data, k = k)
        disti_ = []
        for i in range(len(data)):
            sumj = 0
            for j in range(len(data)):
                sumj += self.dist[i][j] * self.dist[i][j]
            disti_.append(sumj/len(data))
        dist_j = []
        for j in range(len(data)):
            sumi = 0
            for i in range(len(data)):
                sumi += self.dist[i][j] * self.dist[i][j]
            dist_j.append(sumi/len(data))
        for i in range(len(data)):
            sumij = 0
            for j in range(len(data)):
                sumij += self.dist[i][j] * self.dist[i][j]
        dist__ = sumij/(len(data)*len(data))
        #使用MDS算法
        B = np.zeros([len(data), len(data)])
        for i in range(len(data)):
            for j in range(len(data)):
                B[i][j] = -1/2*(self.dist[i][j]*self.dist[i][j]-disti_[i]-dist_j[j]+dist__)
        eigenvalue, featurevector = np.linalg.eig(B)
        gama = np.zeros([dim, dim], dtype=complex)
        for i in range(dim):
            for j in range(dim):
                if i==j:
                    gama[i][j] = math.sqrt(eigenvalue[i])
        V = np.zeros([dim, len(data)], dtype=complex)
        for i in range(dim):
            for j in range(len(data)):
                V[i][j] = featurevector[i][j]
        res = np.dot(gama, V)
        self.output = []
        for i in range(len(data)):
            tmp = []
            for j in range(dim):
                tmp.append(float(res[j][i]))
            self.output.append(tmp)

    def GetOutput(self):
        return self.output



if __name__ == "__main__":
    datas = [[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318],
             [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211],
             [0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099],
             [0.639, 0.161], [0.657, 0.198], [0.360, 0.370], [0.593, 0.042],
             [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
             [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437],
             [0.525, 0.369], [0.751, 0.489], [0.532, 0.472], [0.473, 0.376],
             [0.725, 0.445], [0.446, 0.459]]
    isomap = Isomap()
    isomap.Train(data=datas , k=4)
    print(isomap.GetOutput())