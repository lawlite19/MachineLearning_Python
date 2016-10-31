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
  
## 二、逻辑回归
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


  [1]: ./images/LinearRegression_01.png "LinearRegression_01.png"
  [2]: ./images/LogisticRegression_01.png "LogisticRegression_01.png"
  [3]: ./images/LogisticRegression_02.png "LogisticRegression_02.png"
  [4]: ./images/LogisticRegression_03.jpg "LogisticRegression_03.jpg"
  [5]: ./images/LogisticRegression_04.png "LogisticRegression_04.png"
  [6]: ./images/LogisticRegression_05.png "LogisticRegression_05.png"
  [7]: ./images/LogisticRegression_06.png "LogisticRegression_06.png"
  [8]: ./images/LogisticRegression_07.png "LogisticRegression_07.png"