#-*- coding: utf-8 -*-
import numpy as np
from scipy import io as spio
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def kMenas():
    data = spio.loadmat("data.mat")
    X = data['X']   
    model = KMeans(n_clusters=3).fit(X) # n_clusters指定3类，拟合数据
    centroids = model.cluster_centers_  # 聚类中心
    
    plt.scatter(X[:,0], X[:,1])     # 原数据的散点图
    plt.plot(centroids[:,0],centroids[:,1],'r^',markersize=10)  # 聚类中心
    plt.show()

if __name__ == "__main__":
    kMenas()
