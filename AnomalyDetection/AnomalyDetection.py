#-*- coding: utf-8 -*-
# Author: Bob
# Date:   2016.12.22
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio

'''异常检测主运行程序'''
def anomalyDetection_example():
    '''加载并显示数据'''
    data = spio.loadmat('data1.mat')
    X = data['X']
    plt = display_2d_data(X, 'bx')
    plt.title("origin data")
    plt.show()
    
    mu,sigma2 = estimateGaussian(X)
    print mu,sigma2
    p = multivariateGaussian(X,mu,sigma2)
    print p
    
    visualizeFit(X,mu,sigma2)
    
    
    
# 显示二维数据    
def display_2d_data(X,marker):
    plt.plot(X[:,0],X[:,1],marker)
    plt.axis('square')
    return plt

# 参数估计函数（就是求均值和方差）
def estimateGaussian(X):
    m,n = X.shape
    mu = np.zeros((n,1))
    sigma2 = np.zeros((n,1))
    
    mu = np.mean(X, axis=0) # axis=0表示列，每列的均值
    sigma2 = np.var(X,axis=0) # 求每列的方差
    return mu,sigma2
   
# 多元高斯分布函数    
def multivariateGaussian(X,mu,Sigma2):
    k = len(mu)
    if (Sigma2.shape[0]>1):
        Sigma2 = np.diag(Sigma2)
        
    X = X-mu
    argu = (2*np.pi)**(-k/2)*np.linalg.det(Sigma2)**(-0.5)
    p = argu*np.exp(-0.5*np.sum(np.dot(X,np.linalg.inv(Sigma2))*X,axis=1))  # axis表示每行
    return p
    
# 可视化边界
def visualizeFit(X,mu,sigma2):
    X1,X2 = np.meshgrid(0,0.5,35)
    Z = multivariateGaussian(np.vstack((X1,X2)), mu, Sigma2)


if __name__ == '__main__':
    anomalyDetection_example()
    