#-*- coding: utf-8 -*-
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler    #引入归一化的包

def linearRegression():
    print u"加载数据...\n"
    data = loadtxtAndcsv_data("data.txt",",",np.float64)  #读取数据
    X = np.array(data[:,0:-1],dtype=np.float64)      # X对应0到倒数第2列                  
    y = np.array(data[:,-1],dtype=np.float64)        # y对应最后一列  
        
    # 归一化操作
    scaler = StandardScaler()   
    scaler.fit(X)
    x_train = scaler.transform(X)
    x_test = scaler.transform(np.array([1650,3]))
    
    # 线性模型拟合
    model = linear_model.LinearRegression()
    model.fit(x_train, y)
    
    #预测结果
    result = model.predict(x_test)
    print model.coef_       # Coefficient of the features 决策函数中的特征系数
    print model.intercept_  # 又名bias偏置,若设置为False，则为0
    print result            # 预测结果


# 加载txt和csv文件
def loadtxtAndcsv_data(fileName,split,dataType):
    return np.loadtxt(fileName,delimiter=split,dtype=dataType)

# 加载npy文件
def loadnpy_data(fileName):
    return np.load(fileName)




if __name__ == "__main__":
    linearRegression()