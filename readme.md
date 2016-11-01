机器学习算法Python实现
=========

## 一、[线性回归](/LinearRegression)
- [全部代码](/LinearRegression/LinearRegression.py)

### 1、代价函数
- ![J(\theta ) = \frac{1}{{2{\text{m}}}}\sum\limits_{i = 1}^m {{{({h_\theta }({x^{(i)}}) - {y^{(i)}})}^2}} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=J%28%5Ctheta%20%29%20%3D%20%5Cfrac%7B1%7D%7B%7B2%7B%5Ctext%7Bm%7D%7D%7D%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%7B%7B%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%20-%20%7By%5E%7B%28i%29%7D%7D%29%7D%5E2%7D%7D%20)
- 其中：
![{h_\theta }(x) = {\theta _0} + {\theta _1}{x_1} + {\theta _2}{x_2} + ...](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bh_%5Ctheta%20%7D%28x%29%20%3D%20%7B%5Ctheta%20_0%7D%20%2B%20%7B%5Ctheta%20_1%7D%7Bx_1%7D%20%2B%20%7B%5Ctheta%20_2%7D%7Bx_2%7D%20%2B%20...)

- 下面就是要求出theta，使代价最小，即代表我们拟合出来的方程距离真实值最近
- 共有m条数据，其中![{{{({h_\theta }({x^{(i)}}) - {y^{(i)}})}^2}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%7B%7B%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%20-%20%7By%5E%7B%28i%29%7D%7D%29%7D%5E2%7D%7D)代表我们要拟合出来的方程到真实值距离的平方，平方的原因是因为可能有负值，正负可能会抵消
- 前面有系数2的原因是下面求梯度是对每个变量求偏导，2可以消去

- 实现代码：
```
# 计算代价函数
def computerCost(X,y,theta):
    m = len(y)
    J = 0
    
    J = (np.transpose(X*theta-y))*(X*theta-y)/(2*m) #计算代价J
    return J
```
 - 注意这里的X是真实数据前加了一列1，因为有theta(0)

### 2、梯度下降算法
- 代价函数对![{{\theta _j}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%7B%5Ctheta%20_j%7D%7D)求偏导得到：   
![\frac{{\partial J(\theta )}}{{\partial {\theta _j}}} = \frac{1}{m}\sum\limits_{i = 1}^m {[({h_\theta }({x^{(i)}}) - {y^{(i)}})x_j^{(i)}]} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Cfrac%7B%7B%5Cpartial%20J%28%5Ctheta%20%29%7D%7D%7B%7B%5Cpartial%20%7B%5Ctheta%20_j%7D%7D%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%20-%20%7By%5E%7B%28i%29%7D%7D%29x_j%5E%7B%28i%29%7D%5D%7D%20)
- 所以对theta的更新可以写为：   
![{\theta _j} = {\theta _j} - \alpha \frac{1}{m}\sum\limits_{i = 1}^m {[({h_\theta }({x^{(i)}}) - {y^{(i)}})x_j^{(i)}]} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Ctheta%20_j%7D%20%3D%20%7B%5Ctheta%20_j%7D%20-%20%5Calpha%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%20-%20%7By%5E%7B%28i%29%7D%7D%29x_j%5E%7B%28i%29%7D%5D%7D%20)
- 其中![\alpha ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Calpha%20)为学习速率，控制梯度下降的速度，一般取**0.01,0.03,0.1,0.3.....**
- 实现代码
```
# 梯度下降算法
def gradientDescent(X,y,theta,alpha,num_iters):
    m = len(y)      
    n = len(theta)
    
    temp = np.matrix(np.zeros((n,num_iters)))   # 暂存每次迭代计算的theta，转化为矩阵形式
    
    
    J_history = np.zeros((num_iters,1)) #记录每次迭代计算的代价值
    
    for i in range(num_iters):  # 遍历迭代次数    
        h = np.dot(X,theta)     # 计算内积，matrix可以直接乘
        temp[:,i] = theta - ((alpha/m)*(np.dot(np.transpose(X),h-y)))   #梯度的计算
        theta = temp[:,i]
        J_history[i] = computerCost(X,y,theta)      #调用计算代价函数
        print '.',      
    return theta,J_history  
```

### 3、均值归一化
- 目的是使数据都缩放到一个范围内，便于使用梯度下降算法
- ![{x_i} = \frac{{{x_i} - {\mu _i}}}{{{s_i}}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bx_i%7D%20%3D%20%5Cfrac%7B%7B%7Bx_i%7D%20-%20%7B%5Cmu%20_i%7D%7D%7D%7B%7B%7Bs_i%7D%7D%7D)
- 其中 ![{{\mu _i}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%7B%5Cmu%20_i%7D%7D) 为所有此feture数据的平均值
- ![{{s_i}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%7Bs_i%7D%7D)可以是**最大值-最小值**，也可以是这个feature对应的数据的**标准差**
- 实现代码：
```
# 归一化feature
def featureNormaliza(X):
    X_norm = np.array(X)            #将X转化为numpy数组对象，才可以进行矩阵的运算
    #定义所需变量
    mu = np.zeros((1,X.shape[1]))   
    sigma = np.zeros((1,X.shape[1]))
    
    mu = np.mean(X_norm,0)          # 求每一列的平均值（0指定为列，1代表行）
    sigma = np.std(X_norm,0)        # 求每一列的标准差
    for i in range(X.shape[1]):     # 遍历列
        X_norm[:,i] = (X_norm[:,i]-mu[i])/sigma[i]  # 归一化
    
    return X_norm,mu,sigma
```
- 注意预测的时候也需要均值归一化数据

### 4、最终运行结果
- 代价随迭代次数的变化   
![enter description here][1]


### 5、[使用scikit-learn库中的线性模型实现](/LinearRegression/LinearRegression_scikit-learn.py)
- 导入包
```
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler    #引入缩放的包
```
- 归一化
```
    # 归一化操作
    scaler = StandardScaler()   
    scaler.fit(X)
    x_train = scaler.transform(X)
    x_test = scaler.transform(np.array([1650,3]))
```
- 线性模型拟合
```
    # 线性模型拟合
    model = linear_model.LinearRegression()
    model.fit(x_train, y)
``` 
- 预测
```
    #预测结果
    result = model.predict(x_test)
```
  
## 二、[逻辑回归](/LogisticRegression)
- [全部代码](/LogisticRegression/LogisticRegression.py)

### 1、代价函数
- ![\left\{ \begin{gathered}
  J(\theta ) = \frac{1}{m}\sum\limits_{i = 1}^m {\cos t({h_\theta }({x^{(i)}}),{y^{(i)}})}  \hfill \\
  \cos t({h_\theta }(x),y) = \left\{ {\begin{array}{c}    { - \log ({h_\theta }(x))} \\    { - \log (1 - {h_\theta }(x))}  \end{array} \begin{array}{c}    {y = 1} \\    {y = 0}  \end{array} } \right. \hfill \\ 
\end{gathered}  \right.](http://latex.codecogs.com/gif.latex?%5Clarge%20%5Cleft%5C%7B%20%5Cbegin%7Bgathered%7D%20J%28%5Ctheta%20%29%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5Ccos%20t%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%2C%7By%5E%7B%28i%29%7D%7D%29%7D%20%5Chfill%20%5C%5C%20%5Ccos%20t%28%7Bh_%5Ctheta%20%7D%28x%29%2Cy%29%20%3D%20%5Cleft%5C%7B%20%7B%5Cbegin%7Barray%7D%7Bc%7D%20%7B%20-%20%5Clog%20%28%7Bh_%5Ctheta%20%7D%28x%29%29%7D%20%5C%5C%20%7B%20-%20%5Clog%20%281%20-%20%7Bh_%5Ctheta%20%7D%28x%29%29%7D%20%5Cend%7Barray%7D%20%5Cbegin%7Barray%7D%7Bc%7D%20%7By%20%3D%201%7D%20%5C%5C%20%7By%20%3D%200%7D%20%5Cend%7Barray%7D%20%7D%20%5Cright.%20%5Chfill%20%5C%5C%20%5Cend%7Bgathered%7D%20%5Cright.)
- 可以综合起来为：
![J(\theta ) =  - \frac{1}{m}\sum\limits_{i = 1}^m {[{y^{(i)}}\log ({h_\theta }({x^{(i)}}) + (1 - } {y^{(i)}})\log (1 - {h_\theta }({x^{(i)}})]](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=J%28%5Ctheta%20%29%20%3D%20%20-%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%7By%5E%7B%28i%29%7D%7D%5Clog%20%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%20%2B%20%281%20-%20%7D%20%7By%5E%7B%28i%29%7D%7D%29%5Clog%20%281%20-%20%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%5D)
其中：
![{h_\theta }(x) = \frac{1}{{1 + {e^{ - x}}}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bh_%5Ctheta%20%7D%28x%29%20%3D%20%5Cfrac%7B1%7D%7B%7B1%20%2B%20%7Be%5E%7B%20-%20x%7D%7D%7D%7D)
- 为什么不用线性回归的代价函数表示，因为线性回归的代价函数可能是非凸的，对于分类问题，使用梯度下降很难得到最小值，上面的代价函数是凸函数
- ![{ - \log ({h_\theta }(x))}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%20-%20%5Clog%20%28%7Bh_%5Ctheta%20%7D%28x%29%29%7D)的图像如下，即`y=1`时：
![enter description here][2]

可以看出，当![{{h_\theta }(x)}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%7Bh_%5Ctheta%20%7D%28x%29%7D)趋于`1`，`y=1`,与预测值一致，此时付出的代价`cost`趋于`0`，若![{{h_\theta }(x)}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%7Bh_%5Ctheta%20%7D%28x%29%7D)趋于`0`，`y=1`,此时的代价`cost`值非常大，我们最终的目的是最小化代价值
- 同理![{ - \log (1 - {h_\theta }(x))}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%20-%20%5Clog%20%281%20-%20%7Bh_%5Ctheta%20%7D%28x%29%29%7D)的图像如下（`y=0`）：   
![enter description here][3]

### 2、梯度
- 同样对代价函数求偏导：
![\frac{{\partial J(\theta )}}{{\partial {\theta _j}}} = \frac{1}{m}\sum\limits_{i = 1}^m {[({h_\theta }({x^{(i)}}) - {y^{(i)}})x_j^{(i)}]} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Cfrac%7B%7B%5Cpartial%20J%28%5Ctheta%20%29%7D%7D%7B%7B%5Cpartial%20%7B%5Ctheta%20_j%7D%7D%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%20-%20%7By%5E%7B%28i%29%7D%7D%29x_j%5E%7B%28i%29%7D%5D%7D%20)   
可以看出与线性回归的偏导数一致
- 推到过程
![enter description here][4]

### 3、正则化
- 目的是为了防止过拟合
- 在代价函数中加上一项![\frac{\lambda }{{2m}}\sum\limits_{j = 1}^m {\theta _j^2} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Cfrac%7B%5Clambda%20%7D%7B%7B2m%7D%7D%5Csum%5Climits_%7Bj%20%3D%201%7D%5Em%20%7B%5Ctheta%20_j%5E2%7D%20)，所以最终的代价函数为：  
![J(\theta ) =  - \frac{1}{m}\sum\limits_{i = 1}^m {[{y^{(i)}}\log ({h_\theta }({x^{(i)}}) + (1 - } {y^{(i)}})\log (1 - {h_\theta }({x^{(i)}})] + \frac{\lambda }{{2m}}\sum\limits_{j = 1}^m {\theta _j^2} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=J%28%5Ctheta%20%29%20%3D%20%20-%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%7By%5E%7B%28i%29%7D%7D%5Clog%20%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%20%2B%20%281%20-%20%7D%20%7By%5E%7B%28i%29%7D%7D%29%5Clog%20%281%20-%20%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%5D%20%2B%20%5Cfrac%7B%5Clambda%20%7D%7B%7B2m%7D%7D%5Csum%5Climits_%7Bj%20%3D%201%7D%5Em%20%7B%5Ctheta%20_j%5E2%7D%20)
- 注意j是重1开始的，因为theta(0)为一个常数项，X中最前面一列会加上1列1，所以乘积还是theta(0),feature没有关系，没有必要正则化
- 正则化后的代价：
```
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
```
- 正则化后的代价的梯度
```
# 计算梯度
def gradient(initial_theta,X,y,inital_lambda):
    m = len(y)
    grad = np.zeros((initial_theta.shape[0]))
    
    h = sigmoid(np.dot(X,initial_theta))# 计算h(z)
    theta1 = initial_theta.copy()
    theta1[0] = 0

    grad = np.dot(np.transpose(X),h-y)/m+inital_lambda/m*theta1 #正则化的梯度
    return grad  
```

### 4、S型函数（即![{{h_\theta }(x)}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%7Bh_%5Ctheta%20%7D%28x%29%7D)）
- 实现代码：
```
# S型函数    
def sigmoid(z):
    h = np.zeros((len(z),1))    # 初始化，与z的长度一置
    
    h = 1.0/(1.0+np.exp(-z))
    return h
```

### 5、映射为多项式
- 因为数据的feture可能很少，导致偏差大，所以创造出一些feture结合
- eg:映射为2次方的形式:![1 + {x_1} + {x_2} + x_1^2 + {x_1}{x_2} + x_2^2](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=1%20%2B%20%7Bx_1%7D%20%2B%20%7Bx_2%7D%20%2B%20x_1%5E2%20%2B%20%7Bx_1%7D%7Bx_2%7D%20%2B%20x_2%5E2)
- 实现代码：
```
# 映射为多项式 
def mapFeature(X1,X2):
    degree = 3;                     # 映射的最高次方
    out = np.ones((X1.shape[0],1))  # 映射后的结果数组（取代X）
    '''
    这里以degree=2为例，映射为1,x1,x2,x1^2,x1,x2,x2^2
    '''
    for i in np.arange(1,degree+1): 
        for j in range(i+1):
            temp = X1**(i-j)*(X2**j)    #矩阵直接乘相当于matlab中的点乘.*
            out = np.hstack((out, temp.reshape(-1,1)))
    return out
```

### 6、使用`scipy`的优化方法
- 梯度下降使用`scipy`中`optimize`中的`fmin_bfgs`函数
- 调用scipy中的优化算法fmin_bfgs（拟牛顿法Broyden-Fletcher-Goldfarb-Shanno
 - costFunction是自己实现的一个求代价的函数，
 - initial_theta表示初始化的值,
 - fprime指定costFunction的梯度
 - args是其余测参数，以元组的形式传入，最后会将最小化costFunction的theta返回 
```
    result = optimize.fmin_bfgs(costFunction, initial_theta, fprime=gradient, args=(X,y,initial_lambda))    
```   

### 7、运行结果
- data1决策边界和准确度  
![enter description here][5]
![enter description here][6]
- data2决策边界和准确度  
![enter description here][7]
![enter description here][8]

### 8、[使用scikit-learn库中的逻辑回归模型实现](/LogisticRegression/LogisticRegression_scikit-learn.py)
- 导入包
```
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
```
- 划分训练集和测试集
```
    # 划分为训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
```
- 归一化
```
    # 归一化
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
```
- 逻辑回归
```
    #逻辑回归
    model = LogisticRegression()
    model.fit(x_train,y_train)
``` 
- 预测
```
    # 预测
    predict = model.predict(x_test)
    right = sum(predict == y_test)
    
    predict = np.hstack((predict.reshape(-1,1),y_test.reshape(-1,1)))   # 将预测值和真实值放在一块，好观察
    print predict
    print ('测试集准确率：%f%%'%(right*100.0/predict.shape[0]))          #计算在测试集上的准确度
```


-------------

 ## [逻辑回归_手写数字识别_OneVsAll](/LogisticRegression)
- [全部代码](/LogisticRegression/LogisticRegression_OneVsAll.py)

### 1、随机显示100个数字
- 我没有使用scikit-learn中的数据集，像素是20*20px，彩色图如下
![enter description here][9]
灰度图：
![enter description here][10]
- 实现代码：
```
# 显示100个数字
def display_data(imgData):
    sum = 0
    '''
    显示100个数（若是一个一个绘制将会非常慢，可以将要画的数字整理好，放到一个矩阵中，显示这个矩阵即可）
    - 初始化一个二维数组
    - 将每行的数据调整成图像的矩阵，放进二维数组
    - 显示即可
    '''
    pad = 1
    display_array = -np.ones((pad+10*(20+pad),pad+10*(20+pad)))
    for i in range(10):
        for j in range(10):
            display_array[pad+i*(20+pad):pad+i*(20+pad)+20,pad+j*(20+pad):pad+j*(20+pad)+20] = (imgData[sum,:].reshape(20,20,order="F"))    # order=F指定以列优先，在matlab中是这样的，python中需要指定，默认以行
            sum += 1
            
    plt.imshow(display_array,cmap='gray')   #显示灰度图像
    plt.axis('off')
    plt.show()
```

### 2、OneVsAll
- 如何利用逻辑回归解决多分类的问题，OneVsAll就是把当前某一类看成一类，其他所有类别看作一类，这样有成了二分类的问题了
- 如下图，把途中的数据分成三类，先把红色的看成一类，把其他的看作另外一类，进行逻辑回归，然后把蓝色的看成一类，其他的再看成一类，以此类推...
![enter description here][11]
- 可以看出大于2类的情况下，有多少类就要进行多少次的逻辑回归分类

### 3、手写数字识别
- 共有0-9，10个数字，需要10次分类
- 由于**数据集y**给出的是`0,1,2...9`的数字，而进行逻辑回归需要`0/1`的label标记，所以需要对y处理
- 说一下数据集，前`500`个是`0`,`500-1000`是`1`,`...`,所以如下图，处理后的`y`，**前500行的第一列是1，其余都是0,500-1000行第二列是1，其余都是0....**
![enter description here][12]
- 然后调用**梯度下降算法**求解`theta`
- 实现代码：
```
# 求每个分类的theta，最后返回所有的all_theta    
def oneVsAll(X,y,num_labels,Lambda):
    # 初始化变量
    m,n = X.shape
    all_theta = np.zeros((n+1,num_labels))  # 每一列对应相应分类的theta,共10列
    X = np.hstack((np.ones((m,1)),X))       # X前补上一列1的偏置bias
    class_y = np.zeros((m,num_labels))      # 数据的y对应0-9，需要映射为0/1的关系
    initial_theta = np.zeros((n+1,1))       # 初始化一个分类的theta
    
    # 映射y
    for i in range(num_labels):
        class_y[:,i] = np.int32(y==i).reshape(1,-1) # 注意reshape(1,-1)才可以赋值
    
    #np.savetxt("class_y.csv", class_y[0:600,:], delimiter=',')    
    
    '''遍历每个分类，计算对应的theta值'''
    for i in range(num_labels):
        result = optimize.fmin_bfgs(costFunction, initial_theta, fprime=gradient, args=(X,class_y[:,i],Lambda)) # 调用梯度下降的优化方法
        all_theta[:,i] = result.reshape(1,-1)   # 放入all_theta中
        
    all_theta = np.transpose(all_theta) 
    return all_theta
```

### 4、预测
- 之前说过，预测的结果是一个**概率值**，利用学习出来的`theta`代入预测的**S型函数**中，每行的最大值就是是某个数字的最大概率，所在的**列号**就是预测的数字的真实值,因为在分类时，所有为`0`的将`y`映射在第一列，为1的映射在第二列，依次类推
- 实现代码：
```
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
```

### 5、运行结果
- 10次分类，在训练集上的准确度：
![enter description here][13]

### 6、[使用scikit-learn库中的逻辑回归模型实现](/LogisticRegression/LogisticRegression_OneVsAll_scikit-learn.py)
- 1、导入包
```
from scipy import io as spio
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
```
- 2、加载数据
```
    data = loadmat_data("data_digits.mat") 
    X = data['X']   # 获取X数据，每一行对应一个数字20x20px
    y = data['y']   # 这里读取mat文件y的shape=(5000, 1)
    y = np.ravel(y) # 调用sklearn需要转化成一维的(5000,)
```
- 3、拟合模型
```
    model = LogisticRegression()
    model.fit(X, y) # 拟合
```
- 4、预测
```
    predict = model.predict(X) #预测
    
    print u"预测准确度为：%f%%"%np.mean(np.float64(predict == y)*100)
```
- 5、输出结果（在训练集上的准确度）
![enter description here][14]


  [1]: ./images/LinearRegression_01.png "LinearRegression_01.png"
  [2]: ./images/LogisticRegression_01.png "LogisticRegression_01.png"
  [3]: ./images/LogisticRegression_02.png "LogisticRegression_02.png"
  [4]: ./images/LogisticRegression_03.jpg "LogisticRegression_03.jpg"
  [5]: ./images/LogisticRegression_04.png "LogisticRegression_04.png"
  [6]: ./images/LogisticRegression_05.png "LogisticRegression_05.png"
  [7]: ./images/LogisticRegression_06.png "LogisticRegression_06.png"
  [8]: ./images/LogisticRegression_07.png "LogisticRegression_07.png"
  [9]: ./images/LogisticRegression_08.png "LogisticRegression_08.png"
  [10]: ./images/LogisticRegression_09.png "LogisticRegression_09.png"
  [11]: ./images/LogisticRegression_11.png "LogisticRegression_11.png"
  [12]: ./images/LogisticRegression_10.png "LogisticRegression_10.png"
  [13]: ./images/LogisticRegression_12.png "LogisticRegression_12.png"
  [14]: ./images/LogisticRegression_13.png "LogisticRegression_13.png"