#-*- coding: utf-8 -*-
# Author: Bob
# Date:   2016.11.24
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio

'''
主成分分析运行方法
'''
def PCA():
    data_2d = spio.loadmat("data.mat")
    X = data_2d['X']
    m = X.shape[0]
    plt = plot_data_2d(X) # 显示二维的数据
    plt.show()
    
    X_copy = X.copy()
    X_norm,mu,sigma = featureNormalize(X_copy)    # 归一化数据
    plot_data_2d(X_norm)    # 显示归一化后的数据
    
    Sigma = np.dot(np.transpose(X_norm),X_norm)/m
    U,S,V = np.linalg.svd(Sigma)
    
    plt = plot_data_2d(X_norm)
    plt.plot(mu.reshape(-1,1),mu.reshape(-1,1)+S[0]*(U[:,0].reshape(-1,1)),'k-')
    plt.show()
    
    K = 1
    Z = projectData(X_norm,U,K)
    print Z[0]
    


# 可视化二维数据
def plot_data_2d(X):
    plt.plot(X[:,0],X[:,1],'bo')
    return plt

# 归一化数据
def featureNormalize(X):
    '''（每一个数据-当前列的均值）/当前列的标准差'''
    n = X.shape[1]
    mu = np.zeros((1,n));
    sigma = np.zeros((1,n))
    
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    for i in range(n):
        X[:,i] = (X[:,i]-mu[i])/sigma[i]
    return X,mu,sigma


# 映射数据
def projectData(X_norm,U,K):
    Z = np.zeros((X_norm.shape[0],K))
    
    U_reduce = U[:,0:K]
    Z = np.dot(X_norm,U_reduce)
    return Z


if __name__ == "__main__":
    PCA()