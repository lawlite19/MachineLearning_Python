#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from scipy import optimize
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 解决windows环境下画图汉字乱码问题


def logisticRegression_OneVsAll():
    data = loadmat_data("data_digits.mat") 
    X = data['X']   # 获取X数据，每一行对应一个数字20x20px
    y = data['y']
    m,n = X.shape
    num_labels = 10
    
    rand_indices = [t for t in [np.random.randint(x-x, m) for x in range(100)]]  # 生成100个0-m的随机数
    display_data(X[rand_indices,:])     # 显示100个数字
    
    Lambda = 0.1    # 正则化系数
    #y = y.reshape(-1,1)
    all_theta = oneVsAll(X, y, num_labels, Lambda)  # 计算所有的theta
    
    p = predict_oneVsAll(all_theta,X)               # 预测
    # 将预测结果和真实结果保存到文件中
    #res = np.hstack((p,y.reshape(-1,1)))
    #np.savetxt("predict.csv", res, delimiter=',')
    
    print u"预测准确度为：%f%%"%np.mean(np.float64(p == y.reshape(-1,1))*100)
     
# 加载mat文件
def loadmat_data(fileName):
    return spio.loadmat(fileName)
    
# 显示10个数字
def display_data(imgData):
    sum = 0
    display_array = np.ones((200,200))
    for i in range(10):
        for j in range(10):
            display_array[i*20:(i+1)*20,j*20:(j+1)*20] = imgData[sum,:].reshape(20,20)
            sum += 1
            
    plt.imshow(display_array,cmap='gray')
    plt.axis('off')
    plt.show()

# 求每个分类的theta    
def oneVsAll(X,y,num_labels,Lambda):
    # 初始化变量
    m,n = X.shape
    all_theta = np.zeros((n+1,num_labels))
    X = np.hstack((np.ones((m,1)),X))
    class_y = np.zeros((m,num_labels))
    initial_theta = np.zeros((n+1,1))
    
    # 格式化y，将y两两分类
    for i in range(num_labels):
        class_y[:,i] = np.int32(y==i).reshape(1,-1)
    
    for i in range(num_labels):
        #all_theta[:,i] = optimize.fmin(costFunction,initial_theta,args=(X,class_y[:,i].reshape(-1,1),Lambda),maxiter=50)
        result = optimize.fmin_bfgs(costFunction, initial_theta, fprime=gradient, args=(X,class_y[:,i],Lambda))
        all_theta[:,i] = result.reshape(1,-1)
        
    all_theta = np.transpose(all_theta) 
    return all_theta

# 代价函数
def costFunction(initial_theta,X,y,inital_lambda):
    m = len(y)
    J = 0
    
    h = sigmoid(np.dot(X,initial_theta))    # 计算h(z)
    theta1 = initial_theta.copy()           # 因为正则化j=1从1开始，不包含0，所以复制一份，前theta(0)值为0 
    theta1[0] = 0   
    
    temp = np.dot(np.transpose(theta1),theta1)
    J = (-np.dot(np.transpose(y),np.log(h))-np.dot(np.transpose(1-y),np.log(1-h))+temp*inital_lambda/2)/m   # 正则化的代价方程
    return J

# 计算梯度
def gradient(initial_theta,X,y,inital_lambda):
    m = len(y)
    grad = np.zeros((initial_theta.shape[0]))
    
    h = sigmoid(np.dot(X,initial_theta))  # 计算h(z)
    theta1 = initial_theta.copy()
    theta1[0] = 0

    grad = np.dot(np.transpose(X),h-y)/m+inital_lambda/m*theta1 #正则化的梯度
    return grad   
    
# S型函数    
def sigmoid(z):
    h = np.zeros((len(z),1))    # 初始化，与z的长度一致
    
    h = 1.0/(1.0+np.exp(-z))
    return h

# 预测
def predict_oneVsAll(all_theta,X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    p = np.zeros((m,1))
    X = np.hstack((np.ones((m,1)),X))   #在X最前面加一列1
    
    h = sigmoid(np.dot(X,np.transpose(all_theta)))  #预测

    '''
    返回h中每一行最大值所在的列号
    - np.max(h, axis=1)返回h中每一行的最大值（是某个数字的最大概率）
    - 最后where找到的最大概率所在的列号（列号即是对应的数字）
    '''
    p = np.array(np.where(h[0,:] == np.max(h, axis=1)[0]))  
    for i in np.arange(1, m):
        t = np.array(np.where(h[i,:] == np.max(h, axis=1)[i]))
        p = np.vstack((p,t))
    return p
        
        
if __name__ == "__main__":
    logisticRegression_OneVsAll()