import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio


def KMeans():
    data = spio.loadmat("data.mat")
    X = data['X']
    K = 3   # 总类数
    initial_centroids = np.array([[3,3],[6,2],[8,5]])   # 初始化类中心
    idx = findClosestCentroids(X,initial_centroids)     # 找到每条数据属于哪个类
    
    centroids = computerCentroids(X,idx,K)  # 重新计算类中心
    print centroids
    
# 找到每条数据距离哪个类中心最近    
def findClosestCentroids(X,initial_centroids):
    m = X.shape[0]                  # 数据条数
    K = initial_centroids.shape[0]  # 类的总数
    dis = np.zeros((m,K))           # 存储计算每个点分别到K个类的距离
    idx = np.zeros((m,1))           # 要返回的每条数据属于哪个类
    
    '''计算每个点到每个类中心的距离'''
    for i in range(m):
        for j in range(K):
            dis[i,j] = np.dot((X[i,:]-initial_centroids[j,:]).reshape(1,-1),(X[i,:]-initial_centroids[j,:]).reshape(-1,1))
    
    '''返回dis每一行的最小值对应的列号，即为对应的类别'''    
    idx = np.array(np.where(dis[0,:] == np.min(dis, axis=1)[0]))  
    for i in np.arange(1, m):
        t = np.array(np.where(dis[i,:] == np.min(dis, axis=1)[i]))
        idx = np.vstack((idx,t))
    return idx
             

# 计算类中心
def computerCentroids(X,idx,K):
    n = X.shape[1]
    centroids = np.zeros((K,n))
    for i in range(K):
        centroids[i,:] = np.mean(X[np.array(np.where(idx==i)),:], axis=0).reshape(1,-1)   # axis=0为每一列
    return centroids

if __name__ == "__main__":
    KMeans()
    