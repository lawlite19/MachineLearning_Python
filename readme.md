机器学习算法Python实现
=========

[![MIT license](https://img.shields.io/dub/l/vibe-d.svg)](https://github.com/lawlite19/MachineLearning_Python/blob/master/LICENSE)

## 目录
* [机器学习算法Python实现](#机器学习算法python实现)
	* [一、线性回归](#一线性回归)
		* [1、代价函数](#1代价函数)
		* [2、梯度下降算法](#2梯度下降算法)
		* [3、均值归一化](#3均值归一化)
		* [4、最终运行结果](#4最终运行结果)
		* [5、使用scikit-learn库中的线性模型实现](#5使用scikit-learn库中的线性模型实现)
	* [二、逻辑回归](#二逻辑回归)
		* [1、代价函数](#1代价函数)
		* [2、梯度](#2梯度)
		* [3、正则化](#3正则化)
		* [4、S型函数（即）](#4s型函数即)
		* [5、映射为多项式](#5映射为多项式)
		* [6、使用的优化方法](#6使用scipy的优化方法)
		* [7、运行结果](#7运行结果)
		* [8、使用scikit-learn库中的逻辑回归模型实现](#8使用scikit-learn库中的逻辑回归模型实现)
	* [逻辑回归_手写数字识别_OneVsAll](#逻辑回归_手写数字识别_onevsall)
		* [1、随机显示100个数字](#1随机显示100个数字)
		* [2、OneVsAll](#2onevsall)
		* [3、手写数字识别](#3手写数字识别)
		* [4、预测](#4预测)
		* [5、运行结果](#5运行结果)
		* [6、使用scikit-learn库中的逻辑回归模型实现](#6使用scikit-learn库中的逻辑回归模型实现)
	* [三、BP神经网络](#三bp神经网络)
		* [1、神经网络model](#1神经网络model)
		* [2、代价函数](#2代价函数)
		* [3、正则化](#3正则化)
		* [4、反向传播BP](#4反向传播bp)
		* [5、BP可以求梯度的原因](#5bp可以求梯度的原因)
		* [6、梯度检查](#6梯度检查)
		* [7、权重的随机初始化](#7权重的随机初始化)
		* [8、预测](#8预测)
		* [9、输出结果](#9输出结果)
	* [四、SVM支持向量机](#四svm支持向量机)
		* [1、代价函数](#1代价函数)
		* [2、Large Margin](#2large-margin)
		* [3、SVM Kernel（核函数）](#3svm-kernel核函数)
		* [4、使用中的模型代码](#4使用scikit-learn中的svm模型代码)
		* [5、运行结果](#5运行结果)
	* [五、K-Means聚类算法](#五k-means聚类算法)
		* [1、聚类过程](#1聚类过程)
		* [2、目标函数](#2目标函数)
		* [3、聚类中心的选择](#3聚类中心的选择)
		* [4、聚类个数K的选择](#4聚类个数k的选择)
		* [5、应用——图片压缩](#5应用图片压缩)
		* [6、使用scikit-learn库中的线性模型实现聚类](#6使用scikit-learn库中的线性模型实现聚类)
		* [7、运行结果](#7运行结果)
	* [六、PCA主成分分析（降维）](#六pca主成分分析降维)
		* [1、用处](#1用处)
		* [2、2D-->1D，nD-->kD](#22d--1dnd--kd)
		* [3、主成分分析PCA与线性回归的区别](#3主成分分析pca与线性回归的区别)
		* [4、PCA降维过程](#4pca降维过程)
		* [5、数据恢复](#5数据恢复)
		* [6、主成分个数的选择（即要降的维度）](#6主成分个数的选择即要降的维度)
		* [7、使用建议](#7使用建议)
		* [8、运行结果](#8运行结果)
		* [9、使用scikit-learn库中的PCA实现降维](#9使用scikit-learn库中的pca实现降维)
	* [七、异常检测 Anomaly Detection](#七异常检测-anomaly-detection)
		* [1、高斯分布（正态分布）](#1高斯分布正态分布gaussian-distribution)
		* [2、异常检测算法](#2异常检测算法)
		* [3、评价的好坏，以及的选取](#3评价px的好坏以及ε的选取)
		* [4、选择使用什么样的feature（单元高斯分布）](#4选择使用什么样的feature单元高斯分布)
		* [5、多元高斯分布](#5多元高斯分布)
		* [6、单元和多元高斯分布特点](#6单元和多元高斯分布特点)
		* [7、程序运行结果](#7程序运行结果)

## 一、[线性回归](/LinearRegression)
- [全部代码](/LinearRegression/LinearRegression.py)

### 1、代价函数
- ![J(\theta ) = \frac{1}{{2{\text{m}}}}\sum\limits_{i = 1}^m {{{({h_\theta }({x^{(i)}}) - {y^{(i)}})}^2}} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=J%28%5Ctheta%20%29%20%3D%20%5Cfrac%7B1%7D%7B%7B2%7B%5Ctext%7Bm%7D%7D%7D%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%7B%7B%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%20-%20%7By%5E%7B%28i%29%7D%7D%29%7D%5E2%7D%7D%20)
- 其中：
![{h_\theta }(x) = {\theta _0} + {\theta _1}{x_1} + {\theta _2}{x_2} + ...](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bh_%5Ctheta%20%7D%28x%29%20%3D%20%7B%5Ctheta%20_0%7D%20%2B%20%7B%5Ctheta%20_1%7D%7Bx_1%7D%20%2B%20%7B%5Ctheta%20_2%7D%7Bx_2%7D%20%2B%20...)

- 下面就是要求出theta，使代价最小，即代表我们拟合出来的方程距离真实值最近
- 共有m条数据，其中![{{{({h_\theta }({x^{(i)}}) - {y^{(i)}})}^2}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%7B%7B%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%20-%20%7By%5E%7B%28i%29%7D%7D%29%7D%5E2%7D%7D)代表我们要拟合出来的方程到真实值距离的平方，平方的原因是因为可能有负值，正负可能会抵消
- 前面有系数`2`的原因是下面求梯度是对每个变量求偏导，`2`可以消去

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
- 为什么梯度下降可以逐步减小代价函数
 - 假设函数`f(x)`
 - 泰勒展开：`f(x+△x)=f(x)+f'(x)*△x+o(△x)`
 - 令：`△x=-α*f'(x)`   ,即负梯度方向乘以一个很小的步长`α`
 - 将`△x`代入泰勒展开式中：`f(x+△x)=f(x)-α*[f'(x)]²+o(△x)`
 - 可以看出，`α`是取得很小的正数，`[f'(x)]²`也是正数，所以可以得出：`f(x+△x)<=f(x)`
 - 所以沿着**负梯度**放下，函数在减小，多维情况一样。
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

-------------------

  
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
- 在代价函数中加上一项![J(\theta ) =  - \frac{1}{m}\sum\limits_{i = 1}^m {[{y^{(i)}}\log ({h_\theta }({x^{(i)}}) + (1 - } {y^{(i)}})\log (1 - {h_\theta }({x^{(i)}})] + \frac{\lambda }{{2m}}\sum\limits_{j = 1}^n {\theta _j^2} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=J%28%5Ctheta%20%29%20%3D%20%20-%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%7By%5E%7B%28i%29%7D%7D%5Clog%20%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%20%2B%20%281%20-%20%7D%20%7By%5E%7B%28i%29%7D%7D%29%5Clog%20%281%20-%20%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%5D%20%2B%20%5Cfrac%7B%5Clambda%20%7D%7B%7B2m%7D%7D%5Csum%5Climits_%7Bj%20%3D%201%7D%5En%20%7B%5Ctheta%20_j%5E2%7D%20)
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


----------

## 三、BP神经网络
- [全部代码](/NeuralNetwok/NeuralNetwork.py)

### 1、神经网络model
- 先介绍个三层的神经网络，如下图所示
 - 输入层（input layer）有三个units（![{x_0}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bx_0%7D)为补上的bias，通常设为`1`）
 - ![a_i^{(j)}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=a_i%5E%7B%28j%29%7D)表示第`j`层的第`i`个激励，也称为为单元unit
 - ![{\theta ^{(j)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Ctheta%20%5E%7B%28j%29%7D%7D)为第`j`层到第`j+1`层映射的权重矩阵，就是每条边的权重
![enter description here][15]

- 所以可以得到：
 - 隐含层：  
![a_1^{(2)} = g(\theta _{10}^{(1)}{x_0} + \theta _{11}^{(1)}{x_1} + \theta _{12}^{(1)}{x_2} + \theta _{13}^{(1)}{x_3})](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=a_1%5E%7B%282%29%7D%20%3D%20g%28%5Ctheta%20_%7B10%7D%5E%7B%281%29%7D%7Bx_0%7D%20%2B%20%5Ctheta%20_%7B11%7D%5E%7B%281%29%7D%7Bx_1%7D%20%2B%20%5Ctheta%20_%7B12%7D%5E%7B%281%29%7D%7Bx_2%7D%20%2B%20%5Ctheta%20_%7B13%7D%5E%7B%281%29%7D%7Bx_3%7D%29)   
![a_2^{(2)} = g(\theta _{20}^{(1)}{x_0} + \theta _{21}^{(1)}{x_1} + \theta _{22}^{(1)}{x_2} + \theta _{23}^{(1)}{x_3})](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=a_2%5E%7B%282%29%7D%20%3D%20g%28%5Ctheta%20_%7B20%7D%5E%7B%281%29%7D%7Bx_0%7D%20%2B%20%5Ctheta%20_%7B21%7D%5E%7B%281%29%7D%7Bx_1%7D%20%2B%20%5Ctheta%20_%7B22%7D%5E%7B%281%29%7D%7Bx_2%7D%20%2B%20%5Ctheta%20_%7B23%7D%5E%7B%281%29%7D%7Bx_3%7D%29)   
![a_3^{(2)} = g(\theta _{30}^{(1)}{x_0} + \theta _{31}^{(1)}{x_1} + \theta _{32}^{(1)}{x_2} + \theta _{33}^{(1)}{x_3})](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=a_3%5E%7B%282%29%7D%20%3D%20g%28%5Ctheta%20_%7B30%7D%5E%7B%281%29%7D%7Bx_0%7D%20%2B%20%5Ctheta%20_%7B31%7D%5E%7B%281%29%7D%7Bx_1%7D%20%2B%20%5Ctheta%20_%7B32%7D%5E%7B%281%29%7D%7Bx_2%7D%20%2B%20%5Ctheta%20_%7B33%7D%5E%7B%281%29%7D%7Bx_3%7D%29)
 - 输出层   
![{h_\theta }(x) = a_1^{(3)} = g(\theta _{10}^{(2)}a_0^{(2)} + \theta _{11}^{(2)}a_1^{(2)} + \theta _{12}^{(2)}a_2^{(2)} + \theta _{13}^{(2)}a_3^{(2)})](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bh_%5Ctheta%20%7D%28x%29%20%3D%20a_1%5E%7B%283%29%7D%20%3D%20g%28%5Ctheta%20_%7B10%7D%5E%7B%282%29%7Da_0%5E%7B%282%29%7D%20%2B%20%5Ctheta%20_%7B11%7D%5E%7B%282%29%7Da_1%5E%7B%282%29%7D%20%2B%20%5Ctheta%20_%7B12%7D%5E%7B%282%29%7Da_2%5E%7B%282%29%7D%20%2B%20%5Ctheta%20_%7B13%7D%5E%7B%282%29%7Da_3%5E%7B%282%29%7D%29) 其中，**S型函数**![g(z) = \frac{1}{{1 + {e^{ - z}}}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=g%28z%29%20%3D%20%5Cfrac%7B1%7D%7B%7B1%20%2B%20%7Be%5E%7B%20-%20z%7D%7D%7D%7D)，也成为**激励函数**
- 可以看出![{\theta ^{(1)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Ctheta%20%5E%7B%281%29%7D%7D) 为3x4的矩阵，![{\theta ^{(2)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Ctheta%20%5E%7B%282%29%7D%7D)为1x4的矩阵
 - ![{\theta ^{(j)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Ctheta%20%5E%7B%28j%29%7D%7D) ==》`j+1`的单元数x（`j`层的单元数+1）

### 2、代价函数
- 假设最后输出的![{h_\Theta }(x) \in {R^K}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bh_%5CTheta%20%7D%28x%29%20%5Cin%20%7BR%5EK%7D)，即代表输出层有K个单元
- ![J(\Theta ) =  - \frac{1}{m}\sum\limits_{i = 1}^m {\sum\limits_{k = 1}^K {[y_k^{(i)}\log {{({h_\Theta }({x^{(i)}}))}_k}} }  + (1 - y_k^{(i)})\log {(1 - {h_\Theta }({x^{(i)}}))_k}]](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=J%28%5CTheta%20%29%20%3D%20%20-%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5Csum%5Climits_%7Bk%20%3D%201%7D%5EK%20%7B%5By_k%5E%7B%28i%29%7D%5Clog%20%7B%7B%28%7Bh_%5CTheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%29%7D_k%7D%7D%20%7D%20%20%2B%20%281%20-%20y_k%5E%7B%28i%29%7D%29%5Clog%20%7B%281%20-%20%7Bh_%5CTheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%29_k%7D%5D) 其中，![{({h_\Theta }(x))_i}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%28%7Bh_%5CTheta%20%7D%28x%29%29_i%7D)代表第`i`个单元输出
- 与逻辑回归的代价函数![J(\theta ) =  - \frac{1}{m}\sum\limits_{i = 1}^m {[{y^{(i)}}\log ({h_\theta }({x^{(i)}}) + (1 - } {y^{(i)}})\log (1 - {h_\theta }({x^{(i)}})]](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=J%28%5Ctheta%20%29%20%3D%20%20-%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%7By%5E%7B%28i%29%7D%7D%5Clog%20%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%20%2B%20%281%20-%20%7D%20%7By%5E%7B%28i%29%7D%7D%29%5Clog%20%281%20-%20%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%5D)差不多，就是累加上每个输出（共有K个输出）



### 3、正则化
- `L`-->所有层的个数
- ![{S_l}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7BS_l%7D)-->第`l`层unit的个数
- 正则化后的**代价函数**为  
![enter description here][16]
 - ![\theta ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Ctheta%20)共有`L-1`层，
 - 然后是累加对应每一层的theta矩阵，注意不包含加上偏置项对应的theta(0)
- 正则化后的代价函数实现代码：
```
# 代价函数
def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda):
    length = nn_params.shape[0] # theta的中长度
    # 还原theta1和theta2
    Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):length].reshape(num_labels,hidden_layer_size+1)
    
    # np.savetxt("Theta1.csv",Theta1,delimiter=',')
    
    m = X.shape[0]
    class_y = np.zeros((m,num_labels))      # 数据的y对应0-9，需要映射为0/1的关系
    # 映射y
    for i in range(num_labels):
        class_y[:,i] = np.int32(y==i).reshape(1,-1) # 注意reshape(1,-1)才可以赋值
     
    '''去掉theta1和theta2的第一列，因为正则化时从1开始'''    
    Theta1_colCount = Theta1.shape[1]    
    Theta1_x = Theta1[:,1:Theta1_colCount]
    Theta2_colCount = Theta2.shape[1]    
    Theta2_x = Theta2[:,1:Theta2_colCount]
    # 正则化向theta^2
    term = np.dot(np.transpose(np.vstack((Theta1_x.reshape(-1,1),Theta2_x.reshape(-1,1)))),np.vstack((Theta1_x.reshape(-1,1),Theta2_x.reshape(-1,1))))
    
    '''正向传播,每次需要补上一列1的偏置bias'''
    a1 = np.hstack((np.ones((m,1)),X))      
    z2 = np.dot(a1,np.transpose(Theta1))    
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m,1)),a2))
    z3 = np.dot(a2,np.transpose(Theta2))
    h  = sigmoid(z3)    
    '''代价'''    
    J = -(np.dot(np.transpose(class_y.reshape(-1,1)),np.log(h.reshape(-1,1)))+np.dot(np.transpose(1-class_y.reshape(-1,1)),np.log(1-h.reshape(-1,1)))-Lambda*term/2)/m   
    
    return np.ravel(J)
```

### 4、反向传播BP
- 上面正向传播可以计算得到`J(θ)`,使用梯度下降法还需要求它的梯度
- BP反向传播的目的就是求代价函数的梯度
- 假设4层的神经网络,![\delta _{\text{j}}^{(l)}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Cdelta%20_%7B%5Ctext%7Bj%7D%7D%5E%7B%28l%29%7D)记为-->`l`层第`j`个单元的误差
 - ![\delta _{\text{j}}^{(4)} = a_j^{(4)} - {y_i}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Cdelta%20_%7B%5Ctext%7Bj%7D%7D%5E%7B%284%29%7D%20%3D%20a_j%5E%7B%284%29%7D%20-%20%7By_i%7D)《===》![{\delta ^{(4)}} = {a^{(4)}} - y](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Cdelta%20%5E%7B%284%29%7D%7D%20%3D%20%7Ba%5E%7B%284%29%7D%7D%20-%20y)（向量化）
 - ![{\delta ^{(3)}} = {({\theta ^{(3)}})^T}{\delta ^{(4)}}.*{g^}({a^{(3)}})](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Cdelta%20%5E%7B%283%29%7D%7D%20%3D%20%7B%28%7B%5Ctheta%20%5E%7B%283%29%7D%7D%29%5ET%7D%7B%5Cdelta%20%5E%7B%284%29%7D%7D.%2A%7Bg%5E%7D%28%7Ba%5E%7B%283%29%7D%7D%29)
 - ![{\delta ^{(2)}} = {({\theta ^{(2)}})^T}{\delta ^{(3)}}.*{g^}({a^{(2)}})](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Cdelta%20%5E%7B%282%29%7D%7D%20%3D%20%7B%28%7B%5Ctheta%20%5E%7B%282%29%7D%7D%29%5ET%7D%7B%5Cdelta%20%5E%7B%283%29%7D%7D.%2A%7Bg%5E%7D%28%7Ba%5E%7B%282%29%7D%7D%29)
 - 没有![{\delta ^{(1)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Cdelta%20%5E%7B%281%29%7D%7D)，因为对于输入没有误差
- 因为S型函数![{\text{g(z)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Ctext%7Bg%28z%29%7D%7D)的导数为：![{g^}(z){\text{ = g(z)(1 - g(z))}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bg%5E%7D%28z%29%7B%5Ctext%7B%20%3D%20g%28z%29%281%20-%20g%28z%29%29%7D%7D)，所以上面的![{g^}({a^{(3)}})](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bg%5E%7D%28%7Ba%5E%7B%283%29%7D%7D%29)和![{g^}({a^{(2)}})](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bg%5E%7D%28%7Ba%5E%7B%282%29%7D%7D%29)可以在前向传播中计算出来

- 反向传播计算梯度的过程为：
 - ![\Delta _{ij}^{(l)} = 0](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5CDelta%20_%7Bij%7D%5E%7B%28l%29%7D%20%3D%200)（![\Delta ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5CDelta%20)是大写的![\delta ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Cdelta%20)）
 - for i=1-m:     
 -![{a^{(1)}} = {x^{(i)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Ba%5E%7B%281%29%7D%7D%20%3D%20%7Bx%5E%7B%28i%29%7D%7D)       
-正向传播计算![{a^{(l)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Ba%5E%7B%28l%29%7D%7D)（l=2,3,4...L）      
-反向计算![{\delta ^{(L)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Cdelta%20%5E%7B%28L%29%7D%7D)、![{\delta ^{(L - 1)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Cdelta%20%5E%7B%28L%20-%201%29%7D%7D)...![{\delta ^{(2)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Cdelta%20%5E%7B%282%29%7D%7D)；       
-![\Delta _{ij}^{(l)} = \Delta _{ij}^{(l)} + a_j^{(l)}{\delta ^{(l + 1)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5CDelta%20_%7Bij%7D%5E%7B%28l%29%7D%20%3D%20%5CDelta%20_%7Bij%7D%5E%7B%28l%29%7D%20%2B%20a_j%5E%7B%28l%29%7D%7B%5Cdelta%20%5E%7B%28l%20%2B%201%29%7D%7D)          
-![D_{ij}^{(l)} = \frac{1}{m}\Delta _{ij}^{(l)} + \lambda \theta _{ij}^l\begin{array}{c}    {}&amp; {(j \ne 0)}  \end{array} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=D_%7Bij%7D%5E%7B%28l%29%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5CDelta%20_%7Bij%7D%5E%7B%28l%29%7D%20%2B%20%5Clambda%20%5Ctheta%20_%7Bij%7D%5El%5Cbegin%7Barray%7D%7Bc%7D%20%20%20%20%7B%7D%26%20%7B%28j%20%5Cne%200%29%7D%20%20%5Cend%7Barray%7D%20)      
![D_{ij}^{(l)} = \frac{1}{m}\Delta _{ij}^{(l)} + \lambda \theta _{ij}^lj = 0\begin{array}{c}    {}&amp; {j = 0}  \end{array} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=D_%7Bij%7D%5E%7B%28l%29%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5CDelta%20_%7Bij%7D%5E%7B%28l%29%7D%20%2B%20%5Clambda%20%5Ctheta%20_%7Bij%7D%5Elj%20%3D%200%5Cbegin%7Barray%7D%7Bc%7D%20%20%20%20%7B%7D%26%20%7Bj%20%3D%200%7D%20%20%5Cend%7Barray%7D%20)     

- 最后![\frac{{\partial J(\Theta )}}{{\partial \Theta _{ij}^{(l)}}} = D_{ij}^{(l)}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Cfrac%7B%7B%5Cpartial%20J%28%5CTheta%20%29%7D%7D%7B%7B%5Cpartial%20%5CTheta%20_%7Bij%7D%5E%7B%28l%29%7D%7D%7D%20%3D%20D_%7Bij%7D%5E%7B%28l%29%7D)，即得到代价函数的梯度
- 实现代码：
```
# 梯度
def nnGradient(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda):
    length = nn_params.shape[0]
    Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1).copy()   # 这里使用copy函数，否则下面修改Theta的值，nn_params也会一起修改
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):length].reshape(num_labels,hidden_layer_size+1).copy()
    m = X.shape[0]
    class_y = np.zeros((m,num_labels))      # 数据的y对应0-9，需要映射为0/1的关系    
    # 映射y
    for i in range(num_labels):
        class_y[:,i] = np.int32(y==i).reshape(1,-1) # 注意reshape(1,-1)才可以赋值
     
    '''去掉theta1和theta2的第一列，因为正则化时从1开始'''
    Theta1_colCount = Theta1.shape[1]    
    Theta1_x = Theta1[:,1:Theta1_colCount]
    Theta2_colCount = Theta2.shape[1]    
    Theta2_x = Theta2[:,1:Theta2_colCount]
    
    Theta1_grad = np.zeros((Theta1.shape))  #第一层到第二层的权重
    Theta2_grad = np.zeros((Theta2.shape))  #第二层到第三层的权重
      
   
    '''正向传播，每次需要补上一列1的偏置bias'''
    a1 = np.hstack((np.ones((m,1)),X))
    z2 = np.dot(a1,np.transpose(Theta1))
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m,1)),a2))
    z3 = np.dot(a2,np.transpose(Theta2))
    h  = sigmoid(z3)
    
    
    '''反向传播，delta为误差，'''
    delta3 = np.zeros((m,num_labels))
    delta2 = np.zeros((m,hidden_layer_size))
    for i in range(m):
        #delta3[i,:] = (h[i,:]-class_y[i,:])*sigmoidGradient(z3[i,:])  # 均方误差的误差率
        delta3[i,:] = h[i,:]-class_y[i,:]                              # 交叉熵误差率
        Theta2_grad = Theta2_grad+np.dot(np.transpose(delta3[i,:].reshape(1,-1)),a2[i,:].reshape(1,-1))
        delta2[i,:] = np.dot(delta3[i,:].reshape(1,-1),Theta2_x)*sigmoidGradient(z2[i,:])
        Theta1_grad = Theta1_grad+np.dot(np.transpose(delta2[i,:].reshape(1,-1)),a1[i,:].reshape(1,-1))
    
    Theta1[:,0] = 0
    Theta2[:,0] = 0          
    '''梯度'''
    grad = (np.vstack((Theta1_grad.reshape(-1,1),Theta2_grad.reshape(-1,1)))+Lambda*np.vstack((Theta1.reshape(-1,1),Theta2.reshape(-1,1))))/m
    return np.ravel(grad)
```

### 5、BP可以求梯度的原因
- 实际是利用了`链式求导`法则
- 因为下一层的单元利用上一层的单元作为输入进行计算
- 大体的推导过程如下，最终我们是想预测函数与已知的`y`非常接近，求均方差的梯度沿着此梯度方向可使代价函数最小化。可对照上面求梯度的过程。
![enter description here][17]
- 求误差更详细的推导过程：
![enter description here][18]

### 6、梯度检查
- 检查利用`BP`求的梯度是否正确
- 利用导数的定义验证：
![\frac{{dJ(\theta )}}{{d\theta }} \approx \frac{{J(\theta  + \varepsilon ) - J(\theta  - \varepsilon )}}{{2\varepsilon }}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Cfrac%7B%7BdJ%28%5Ctheta%20%29%7D%7D%7B%7Bd%5Ctheta%20%7D%7D%20%5Capprox%20%5Cfrac%7B%7BJ%28%5Ctheta%20%20%2B%20%5Cvarepsilon%20%29%20-%20J%28%5Ctheta%20%20-%20%5Cvarepsilon%20%29%7D%7D%7B%7B2%5Cvarepsilon%20%7D%7D)
- 求出来的数值梯度应该与BP求出的梯度非常接近
- 验证BP正确后就不需要再执行验证梯度的算法了
- 实现代码：
```
# 检验梯度是否计算正确
# 检验梯度是否计算正确
def checkGradient(Lambda = 0):
    '''构造一个小型的神经网络验证，因为数值法计算梯度很浪费时间，而且验证正确后之后就不再需要验证了'''
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    initial_Theta1 = debugInitializeWeights(input_layer_size,hidden_layer_size); 
    initial_Theta2 = debugInitializeWeights(hidden_layer_size,num_labels)
    X = debugInitializeWeights(input_layer_size-1,m)
    y = 1+np.transpose(np.mod(np.arange(1,m+1), num_labels))# 初始化y
    
    y = y.reshape(-1,1)
    nn_params = np.vstack((initial_Theta1.reshape(-1,1),initial_Theta2.reshape(-1,1)))  #展开theta 
    '''BP求出梯度'''
    grad = nnGradient(nn_params, input_layer_size, hidden_layer_size, 
                     num_labels, X, y, Lambda)  
    '''使用数值法计算梯度'''
    num_grad = np.zeros((nn_params.shape[0]))
    step = np.zeros((nn_params.shape[0]))
    e = 1e-4
    for i in range(nn_params.shape[0]):
        step[i] = e
        loss1 = nnCostFunction(nn_params-step.reshape(-1,1), input_layer_size, hidden_layer_size, 
                              num_labels, X, y, 
                              Lambda)
        loss2 = nnCostFunction(nn_params+step.reshape(-1,1), input_layer_size, hidden_layer_size, 
                              num_labels, X, y, 
                              Lambda)
        num_grad[i] = (loss2-loss1)/(2*e)
        step[i]=0
    # 显示两列比较
    res = np.hstack((num_grad.reshape(-1,1),grad.reshape(-1,1)))
    print res
```

### 7、权重的随机初始化
- 神经网络不能像逻辑回归那样初始化`theta`为`0`,因为若是每条边的权重都为0，每个神经元都是相同的输出，在反向传播中也会得到同样的梯度，最终只会预测一种结果。
- 所以应该初始化为接近0的数
- 实现代码
```
# 随机初始化权重theta
def randInitializeWeights(L_in,L_out):
    W = np.zeros((L_out,1+L_in))    # 对应theta的权重
    epsilon_init = (6.0/(L_out+L_in))**0.5
    W = np.random.rand(L_out,1+L_in)*2*epsilon_init-epsilon_init # np.random.rand(L_out,1+L_in)产生L_out*(1+L_in)大小的随机矩阵
    return W
```

### 8、预测
- 正向传播预测结果
- 实现代码
```
# 预测
def predict(Theta1,Theta2,X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    #p = np.zeros((m,1))
    '''正向传播，预测结果'''
    X = np.hstack((np.ones((m,1)),X))
    h1 = sigmoid(np.dot(X,np.transpose(Theta1)))
    h1 = np.hstack((np.ones((m,1)),h1))
    h2 = sigmoid(np.dot(h1,np.transpose(Theta2)))
    
    '''
    返回h中每一行最大值所在的列号
    - np.max(h, axis=1)返回h中每一行的最大值（是某个数字的最大概率）
    - 最后where找到的最大概率所在的列号（列号即是对应的数字）
    '''
    #np.savetxt("h2.csv",h2,delimiter=',')
    p = np.array(np.where(h2[0,:] == np.max(h2, axis=1)[0]))  
    for i in np.arange(1, m):
        t = np.array(np.where(h2[i,:] == np.max(h2, axis=1)[i]))
        p = np.vstack((p,t))
    return p 
```

### 9、输出结果
- 梯度检查：     
![enter description here][19]
- 随机显示100个手写数字     
![enter description here][20]
- 显示theta1权重     
![enter description here][21]
- 训练集预测准确度     
![enter description here][22]
- 归一化后训练集预测准确度     
![enter description here][23]

--------------------

## 四、SVM支持向量机

### 1、代价函数
- 在逻辑回归中，我们的代价为：   
![\cos t({h_\theta }(x),y) = \left\{ {\begin{array}{c}    { - \log ({h_\theta }(x))} \\    { - \log (1 - {h_\theta }(x))}  \end{array} \begin{array}{c}    {y = 1} \\    {y = 0}  \end{array} } \right.](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Ccos%20t%28%7Bh_%5Ctheta%20%7D%28x%29%2Cy%29%20%3D%20%5Cleft%5C%7B%20%7B%5Cbegin%7Barray%7D%7Bc%7D%20%20%20%20%7B%20-%20%5Clog%20%28%7Bh_%5Ctheta%20%7D%28x%29%29%7D%20%5C%5C%20%20%20%20%7B%20-%20%5Clog%20%281%20-%20%7Bh_%5Ctheta%20%7D%28x%29%29%7D%20%20%5Cend%7Barray%7D%20%5Cbegin%7Barray%7D%7Bc%7D%20%20%20%20%7By%20%3D%201%7D%20%5C%5C%20%20%20%20%7By%20%3D%200%7D%20%20%5Cend%7Barray%7D%20%7D%20%5Cright.)，    
其中：![{h_\theta }({\text{z}}) = \frac{1}{{1 + {e^{ - z}}}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bh_%5Ctheta%20%7D%28%7B%5Ctext%7Bz%7D%7D%29%20%3D%20%5Cfrac%7B1%7D%7B%7B1%20%2B%20%7Be%5E%7B%20-%20z%7D%7D%7D%7D)，![z = {\theta ^T}x](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=z%20%3D%20%7B%5Ctheta%20%5ET%7Dx)
- 如图所示，如果`y=1`，`cost`代价函数如图所示    
![enter description here][24]    
我们想让![{\theta ^T}x &gt;  &gt; 0](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Ctheta%20%5ET%7Dx%20%3E%20%20%3E%200)，即`z>>0`，这样的话`cost`代价函数才会趋于最小（这是我们想要的），所以用途中**红色**的函数![\cos {t_1}(z)](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Ccos%20%7Bt_1%7D%28z%29)代替逻辑回归中的cost
- 当`y=0`时同样，用![\cos {t_0}(z)](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Ccos%20%7Bt_0%7D%28z%29)代替
![enter description here][25]
- 最终得到的代价函数为：    
![J(\theta ) = C\sum\limits_{i = 1}^m {[{y^{(i)}}\cos {t_1}({\theta ^T}{x^{(i)}}) + (1 - {y^{(i)}})\cos {t_0}({\theta ^T}{x^{(i)}})} ] + \frac{1}{2}\sum\limits_{j = 1}^{\text{n}} {\theta _j^2} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=J%28%5Ctheta%20%29%20%3D%20C%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%7By%5E%7B%28i%29%7D%7D%5Ccos%20%7Bt_1%7D%28%7B%5Ctheta%20%5ET%7D%7Bx%5E%7B%28i%29%7D%7D%29%20%2B%20%281%20-%20%7By%5E%7B%28i%29%7D%7D%29%5Ccos%20%7Bt_0%7D%28%7B%5Ctheta%20%5ET%7D%7Bx%5E%7B%28i%29%7D%7D%29%7D%20%5D%20%2B%20%5Cfrac%7B1%7D%7B2%7D%5Csum%5Climits_%7Bj%20%3D%201%7D%5E%7B%5Ctext%7Bn%7D%7D%20%7B%5Ctheta%20_j%5E2%7D%20)   
最后我们想要![\mathop {\min }\limits_\theta  J(\theta )](http://latex.codecogs.com/gif.latex?%5Clarge%20%5Cmathop%20%7B%5Cmin%20%7D%5Climits_%5Ctheta%20J%28%5Ctheta%20%29)
- 之前我们逻辑回归中的代价函数为：   
![J(\theta ) =  - \frac{1}{m}\sum\limits_{i = 1}^m {[{y^{(i)}}\log ({h_\theta }({x^{(i)}}) + (1 - } {y^{(i)}})\log (1 - {h_\theta }({x^{(i)}})] + \frac{\lambda }{{2m}}\sum\limits_{j = 1}^n {\theta _j^2} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=J%28%5Ctheta%20%29%20%3D%20%20-%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%7By%5E%7B%28i%29%7D%7D%5Clog%20%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%20%2B%20%281%20-%20%7D%20%7By%5E%7B%28i%29%7D%7D%29%5Clog%20%281%20-%20%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%5D%20%2B%20%5Cfrac%7B%5Clambda%20%7D%7B%7B2m%7D%7D%5Csum%5Climits_%7Bj%20%3D%201%7D%5En%20%7B%5Ctheta%20_j%5E2%7D%20)   
可以认为这里的![C = \frac{m}{\lambda }](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=C%20%3D%20%5Cfrac%7Bm%7D%7B%5Clambda%20%7D)，只是表达形式问题，这里`C`的值越大，SVM的决策边界的`margin`也越大，下面会说明

### 2、Large Margin
- 如下图所示,SVM分类会使用最大的`margin`将其分开    
![enter description here][26]
- 先说一下向量内积
 - ![u = \left[ {\begin{array}{c}    {{u_1}} \\    {{u_2}}  \end{array} } \right]](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=u%20%3D%20%5Cleft%5B%20%7B%5Cbegin%7Barray%7D%7Bc%7D%20%20%20%20%7B%7Bu_1%7D%7D%20%5C%5C%20%20%20%20%7B%7Bu_2%7D%7D%20%20%5Cend%7Barray%7D%20%7D%20%5Cright%5D)，![v = \left[ {\begin{array}{c}    {{v_1}} \\    {{v_2}}  \end{array} } \right]](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=v%20%3D%20%5Cleft%5B%20%7B%5Cbegin%7Barray%7D%7Bc%7D%20%20%20%20%7B%7Bv_1%7D%7D%20%5C%5C%20%20%20%20%7B%7Bv_2%7D%7D%20%20%5Cend%7Barray%7D%20%7D%20%5Cright%5D)    
 - ![||u||](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7C%7Cu%7C%7C)表示`u`的**欧几里得范数**（欧式范数），![||u||{\text{ = }}\sqrt {{\text{u}}_1^2 + u_2^2} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7C%7Cu%7C%7C%7B%5Ctext%7B%20%3D%20%7D%7D%5Csqrt%20%7B%7B%5Ctext%7Bu%7D%7D_1%5E2%20%2B%20u_2%5E2%7D%20)
 - `向量V`在`向量u`上的投影的长度记为`p`，则：向量内积：    
 ![{{\text{u}}^T}v = p||u|| = {u_1}{v_1} + {u_2}{v_2}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%7B%5Ctext%7Bu%7D%7D%5ET%7Dv%20%3D%20p%7C%7Cu%7C%7C%20%3D%20%7Bu_1%7D%7Bv_1%7D%20%2B%20%7Bu_2%7D%7Bv_2%7D)      
 ![enter description here][27]  
根据向量夹角公式推导一下即可，![\cos \theta  = \frac{{\overrightarrow {\text{u}} \overrightarrow v }}{{|\overrightarrow {\text{u}} ||\overrightarrow v |}}](http://latex.codecogs.com/gif.latex?%5Clarge%20%5Ccos%20%5Ctheta%20%3D%20%5Cfrac%7B%7B%5Coverrightarrow%20%7B%5Ctext%7Bu%7D%7D%20%5Coverrightarrow%20v%20%7D%7D%7B%7B%7C%5Coverrightarrow%20%7B%5Ctext%7Bu%7D%7D%20%7C%7C%5Coverrightarrow%20v%20%7C%7D%7D)

- 前面说过，当`C`越大时，`margin`也就越大，我们的目的是最小化代价函数`J(θ)`,当`margin`最大时，`C`的乘积项![\sum\limits_{i = 1}^m {[{y^{(i)}}\cos {t_1}({\theta ^T}{x^{(i)}}) + (1 - {y^{(i)}})\cos {t_0}({\theta ^T}{x^{(i)}})} ]](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%7By%5E%7B%28i%29%7D%7D%5Ccos%20%7Bt_1%7D%28%7B%5Ctheta%20%5ET%7D%7Bx%5E%7B%28i%29%7D%7D%29%20%2B%20%281%20-%20%7By%5E%7B%28i%29%7D%7D%29%5Ccos%20%7Bt_0%7D%28%7B%5Ctheta%20%5ET%7D%7Bx%5E%7B%28i%29%7D%7D%29%7D%20%5D)要很小，所以近似为：   
![J(\theta ) = C0 + \frac{1}{2}\sum\limits_{j = 1}^{\text{n}} {\theta _j^2}  = \frac{1}{2}\sum\limits_{j = 1}^{\text{n}} {\theta _j^2}  = \frac{1}{2}(\theta _1^2 + \theta _2^2) = \frac{1}{2}{\sqrt {\theta _1^2 + \theta _2^2} ^2}](http://latex.codecogs.com/gif.latex?%5Clarge%20J%28%5Ctheta%20%29%20%3D%20C0%20&plus;%20%5Cfrac%7B1%7D%7B2%7D%5Csum%5Climits_%7Bj%20%3D%201%7D%5E%7B%5Ctext%7Bn%7D%7D%20%7B%5Ctheta%20_j%5E2%7D%20%3D%20%5Cfrac%7B1%7D%7B2%7D%5Csum%5Climits_%7Bj%20%3D%201%7D%5E%7B%5Ctext%7Bn%7D%7D%20%7B%5Ctheta%20_j%5E2%7D%20%3D%20%5Cfrac%7B1%7D%7B2%7D%28%5Ctheta%20_1%5E2%20&plus;%20%5Ctheta%20_2%5E2%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%7B%5Csqrt%20%7B%5Ctheta%20_1%5E2%20&plus;%20%5Ctheta%20_2%5E2%7D%20%5E2%7D)，      
我们最后的目的就是求使代价最小的`θ`
- 由   
![\left\{ {\begin{array}{c}    {{\theta ^T}{x^{(i)}} \geqslant 1} \\    {{\theta ^T}{x^{(i)}} \leqslant  - 1}  \end{array} } \right.\begin{array}{c}    {({y^{(i)}} = 1)} \\    {({y^{(i)}} = 0)}  \end{array} ](http://latex.codecogs.com/gif.latex?%5Clarge%20%5Cleft%5C%7B%20%7B%5Cbegin%7Barray%7D%7Bc%7D%20%7B%7B%5Ctheta%20%5ET%7D%7Bx%5E%7B%28i%29%7D%7D%20%5Cgeqslant%201%7D%20%5C%5C%20%7B%7B%5Ctheta%20%5ET%7D%7Bx%5E%7B%28i%29%7D%7D%20%5Cleqslant%20-%201%7D%20%5Cend%7Barray%7D%20%7D%20%5Cright.%5Cbegin%7Barray%7D%7Bc%7D%20%7B%28%7By%5E%7B%28i%29%7D%7D%20%3D%201%29%7D%20%5C%5C%20%7B%28%7By%5E%7B%28i%29%7D%7D%20%3D%200%29%7D%20%5Cend%7Barray%7D)可以得到：    
![\left\{ {\begin{array}{c}    {{p^{(i)}}||\theta || \geqslant 1} \\    {{p^{(i)}}||\theta || \leqslant  - 1}  \end{array} } \right.\begin{array}{c}    {({y^{(i)}} = 1)} \\    {({y^{(i)}} = 0)}  \end{array} ](http://latex.codecogs.com/gif.latex?%5Clarge%20%5Cleft%5C%7B%20%7B%5Cbegin%7Barray%7D%7Bc%7D%20%7B%7Bp%5E%7B%28i%29%7D%7D%7C%7C%5Ctheta%20%7C%7C%20%5Cgeqslant%201%7D%20%5C%5C%20%7B%7Bp%5E%7B%28i%29%7D%7D%7C%7C%5Ctheta%20%7C%7C%20%5Cleqslant%20-%201%7D%20%5Cend%7Barray%7D%20%7D%20%5Cright.%5Cbegin%7Barray%7D%7Bc%7D%20%7B%28%7By%5E%7B%28i%29%7D%7D%20%3D%201%29%7D%20%5C%5C%20%7B%28%7By%5E%7B%28i%29%7D%7D%20%3D%200%29%7D%20%5Cend%7Barray%7D)，`p`即为`x`在`θ`上的投影
- 如下图所示，假设决策边界如图，找其中的一个点，到`θ`上的投影为`p`,则![p||\theta || \geqslant 1](http://latex.codecogs.com/gif.latex?%5Clarge%20p%7C%7C%5Ctheta%20%7C%7C%20%5Cgeqslant%201)或者![p||\theta || \leqslant  - 1](http://latex.codecogs.com/gif.latex?%5Clarge%20p%7C%7C%5Ctheta%20%7C%7C%20%5Cleqslant%20-%201)，若是`p`很小，则需要![||\theta ||](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7C%7C%5Ctheta%20%7C%7C)很大，这与我们要求的`θ`使![||\theta || = \frac{1}{2}\sqrt {\theta _1^2 + \theta _2^2} ](http://latex.codecogs.com/gif.latex?%5Clarge%20%7C%7C%5Ctheta%20%7C%7C%20%3D%20%5Cfrac%7B1%7D%7B2%7D%5Csqrt%20%7B%5Ctheta%20_1%5E2%20&plus;%20%5Ctheta%20_2%5E2%7D)最小相违背，**所以**最后求的是`large margin`   
![enter description here][28]

### 3、SVM Kernel（核函数）
- 对于线性可分的问题，使用**线性核函数**即可
- 对于线性不可分的问题，在逻辑回归中，我们是将`feature`映射为使用多项式的形式![1 + {x_1} + {x_2} + x_1^2 + {x_1}{x_2} + x_2^2](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=1%20%2B%20%7Bx_1%7D%20%2B%20%7Bx_2%7D%20%2B%20x_1%5E2%20%2B%20%7Bx_1%7D%7Bx_2%7D%20%2B%20x_2%5E2)，`SVM`中也有**多项式核函数**，但是更常用的是**高斯核函数**，也称为**RBF核**
- 高斯核函数为：![f(x) = {e^{ - \frac{{||x - u|{|^2}}}{{2{\sigma ^2}}}}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=f%28x%29%20%3D%20%7Be%5E%7B%20-%20%5Cfrac%7B%7B%7C%7Cx%20-%20u%7C%7B%7C%5E2%7D%7D%7D%7B%7B2%7B%5Csigma%20%5E2%7D%7D%7D%7D%7D)     
假设如图几个点，
![enter description here][29]
令：   
![{f_1} = similarity(x,{l^{(1)}}) = {e^{ - \frac{{||x - {l^{(1)}}|{|^2}}}{{2{\sigma ^2}}}}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bf_1%7D%20%3D%20similarity%28x%2C%7Bl%5E%7B%281%29%7D%7D%29%20%3D%20%7Be%5E%7B%20-%20%5Cfrac%7B%7B%7C%7Cx%20-%20%7Bl%5E%7B%281%29%7D%7D%7C%7B%7C%5E2%7D%7D%7D%7B%7B2%7B%5Csigma%20%5E2%7D%7D%7D%7D%7D)   
![{f_2} = similarity(x,{l^{(2)}}) = {e^{ - \frac{{||x - {l^{(2)}}|{|^2}}}{{2{\sigma ^2}}}}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bf_2%7D%20%3D%20similarity%28x%2C%7Bl%5E%7B%282%29%7D%7D%29%20%3D%20%7Be%5E%7B%20-%20%5Cfrac%7B%7B%7C%7Cx%20-%20%7Bl%5E%7B%282%29%7D%7D%7C%7B%7C%5E2%7D%7D%7D%7B%7B2%7B%5Csigma%20%5E2%7D%7D%7D%7D%7D)
.
.
.
- 可以看出，若是`x`与![{l^{(1)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bl%5E%7B%281%29%7D%7D)距离较近，==》![{f_1} \approx {e^0} = 1](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bf_1%7D%20%5Capprox%20%7Be%5E0%7D%20%3D%201)，（即相似度较大）   
若是`x`与![{l^{(1)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bl%5E%7B%281%29%7D%7D)距离较远，==》![{f_2} \approx {e^{ - \infty }} = 0](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bf_2%7D%20%5Capprox%20%7Be%5E%7B%20-%20%5Cinfty%20%7D%7D%20%3D%200)，（即相似度较低）
- 高斯核函数的`σ`越小，`f`下降的越快      
![enter description here][30]
![enter description here][31]

- 如何选择初始的![{l^{(1)}}{l^{(2)}}{l^{(3)}}...](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bl%5E%7B%281%29%7D%7D%7Bl%5E%7B%282%29%7D%7D%7Bl%5E%7B%283%29%7D%7D...)
 - 训练集：![(({x^{(1)}},{y^{(1)}}),({x^{(2)}},{y^{(2)}}),...({x^{(m)}},{y^{(m)}}))](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%28%28%7Bx%5E%7B%281%29%7D%7D%2C%7By%5E%7B%281%29%7D%7D%29%2C%28%7Bx%5E%7B%282%29%7D%7D%2C%7By%5E%7B%282%29%7D%7D%29%2C...%28%7Bx%5E%7B%28m%29%7D%7D%2C%7By%5E%7B%28m%29%7D%7D%29%29)
 - 选择：![{l^{(1)}} = {x^{(1)}},{l^{(2)}} = {x^{(2)}}...{l^{(m)}} = {x^{(m)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bl%5E%7B%281%29%7D%7D%20%3D%20%7Bx%5E%7B%281%29%7D%7D%2C%7Bl%5E%7B%282%29%7D%7D%20%3D%20%7Bx%5E%7B%282%29%7D%7D...%7Bl%5E%7B%28m%29%7D%7D%20%3D%20%7Bx%5E%7B%28m%29%7D%7D)
 - 对于给出的`x`，计算`f`,令：![f_0^{(i)} = 1](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=f_0%5E%7B%28i%29%7D%20%3D%201)所以：![{f^{(i)}} \in {R^{m + 1}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bf%5E%7B%28i%29%7D%7D%20%5Cin%20%7BR%5E%7Bm%20%2B%201%7D%7D)
 - 最小化`J`求出`θ`，          
 ![J(\theta ) = C\sum\limits_{i = 1}^m {[{y^{(i)}}\cos {t_1}({\theta ^T}{f^{(i)}}) + (1 - {y^{(i)}})\cos {t_0}({\theta ^T}{f^{(i)}})} ] + \frac{1}{2}\sum\limits_{j = 1}^{\text{n}} {\theta _j^2} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=J%28%5Ctheta%20%29%20%3D%20C%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%7By%5E%7B%28i%29%7D%7D%5Ccos%20%7Bt_1%7D%28%7B%5Ctheta%20%5ET%7D%7Bf%5E%7B%28i%29%7D%7D%29%20%2B%20%281%20-%20%7By%5E%7B%28i%29%7D%7D%29%5Ccos%20%7Bt_0%7D%28%7B%5Ctheta%20%5ET%7D%7Bf%5E%7B%28i%29%7D%7D%29%7D%20%5D%20%2B%20%5Cfrac%7B1%7D%7B2%7D%5Csum%5Climits_%7Bj%20%3D%201%7D%5E%7B%5Ctext%7Bn%7D%7D%20%7B%5Ctheta%20_j%5E2%7D%20)
 - 如果![{\theta ^T}f \geqslant 0](http://latex.codecogs.com/gif.latex?%5Clarge%20%7B%5Ctheta%20%5ET%7Df%20%5Cgeqslant%200)，==》预测`y=1`

### 4、使用`scikit-learn`中的`SVM`模型代码
- [全部代码](/SVM/SVM_scikit-learn.py)
- 线性可分的,指定核函数为`linear`：
```
    '''data1——线性分类'''
    data1 = spio.loadmat('data1.mat')
    X = data1['X']
    y = data1['y']
    y = np.ravel(y)
    plot_data(X,y)
    
    model = svm.SVC(C=1.0,kernel='linear').fit(X,y) # 指定核函数为线性核函数
```
- 非线性可分的，默认核函数为`rbf`
```
    '''data2——非线性分类'''
    data2 = spio.loadmat('data2.mat')
    X = data2['X']
    y = data2['y']
    y = np.ravel(y)
    plt = plot_data(X,y)
    plt.show()
    
    model = svm.SVC(gamma=100).fit(X,y)     # gamma为核函数的系数，值越大拟合的越好
```
### 5、运行结果
- 线性可分的决策边界：    
![enter description here][32]
- 线性不可分的决策边界：   
![enter description here][33]

--------------------------

## 五、K-Means聚类算法
- [全部代码](/K-Means/K-Menas.py)

### 1、聚类过程
- 聚类属于无监督学习，不知道y的标记分为K类
- K-Means算法分为两个步骤
 - 第一步：簇分配，随机选`K`个点作为中心，计算到这`K`个点的距离，分为`K`个簇
 - 第二步：移动聚类中心：重新计算每个**簇**的中心，移动中心，重复以上步骤。
- 如下图所示：
 - 随机分配的聚类中心  
 ![enter description here][34]
 - 重新计算聚类中心，移动一次  
 ![enter description here][35]
 - 最后`10`步之后的聚类中心  
 ![enter description here][36]

- 计算每条数据到哪个中心最近实现代码：
```
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
    
    '''返回dis每一行的最小值对应的列号，即为对应的类别
    - np.min(dis, axis=1)返回每一行的最小值
    - np.where(dis == np.min(dis, axis=1).reshape(-1,1)) 返回对应最小值的坐标
     - 注意：可能最小值对应的坐标有多个，where都会找出来，所以返回时返回前m个需要的即可（因为对于多个最小值，属于哪个类别都可以）
    '''  
    dummy,idx = np.where(dis == np.min(dis, axis=1).reshape(-1,1))
    return idx[0:dis.shape[0]]  # 注意截取一下
```
- 计算类中心实现代码：
```
# 计算类中心
def computerCentroids(X,idx,K):
    n = X.shape[1]
    centroids = np.zeros((K,n))
    for i in range(K):
        centroids[i,:] = np.mean(X[np.ravel(idx==i),:], axis=0).reshape(1,-1)   # 索引要是一维的,axis=0为每一列，idx==i一次找出属于哪一类的，然后计算均值
    return centroids
```

### 2、目标函数
- 也叫做**失真代价函数**
- ![J({c^{(1)}}, \cdots ,{c^{(m)}},{u_1}, \cdots ,{u_k}) = \frac{1}{m}\sum\limits_{i = 1}^m {||{x^{(i)}} - {u_{{c^{(i)}}}}|{|^2}} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=J%28%7Bc%5E%7B%281%29%7D%7D%2C%20%5Ccdots%20%2C%7Bc%5E%7B%28m%29%7D%7D%2C%7Bu_1%7D%2C%20%5Ccdots%20%2C%7Bu_k%7D%29%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%7C%7C%7Bx%5E%7B%28i%29%7D%7D%20-%20%7Bu_%7B%7Bc%5E%7B%28i%29%7D%7D%7D%7D%7C%7B%7C%5E2%7D%7D%20)
- 最后我们想得到：  
![enter description here][37]
- 其中![{c^{(i)}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bc%5E%7B%28i%29%7D%7D)表示第`i`条数据距离哪个类中心最近，
- 其中![{u_i}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bu_i%7D)即为聚类的中心

### 3、聚类中心的选择
- 随机初始化，从给定的数据中随机抽取K个作为聚类中心
- 随机一次的结果可能不好，可以随机多次，最后取使代价函数最小的作为中心
- 实现代码：(这里随机一次)
```
# 初始化类中心--随机取K个点作为聚类中心
def kMeansInitCentroids(X,K):
    m = X.shape[0]
    m_arr = np.arange(0,m)      # 生成0-m-1
    centroids = np.zeros((K,X.shape[1]))
    np.random.shuffle(m_arr)    # 打乱m_arr顺序    
    rand_indices = m_arr[:K]    # 取前K个
    centroids = X[rand_indices,:]
    return centroids
```

### 4、聚类个数K的选择
- 聚类是不知道y的label的，所以不知道真正的聚类个数
- 肘部法则（Elbow method）
 - 作代价函数`J`和`K`的图，若是出现一个拐点，如下图所示，`K`就取拐点处的值，下图此时`K=3`
 ![enter description here][38]
 - 若是很平滑就不明确，人为选择。
- 第二种就是人为观察选择

### 5、应用——图片压缩
- 将图片的像素分为若干类，然后用这个类代替原来的像素值
- 执行聚类的算法代码：
```
# 聚类算法
def runKMeans(X,initial_centroids,max_iters,plot_process):
    m,n = X.shape                   # 数据条数和维度
    K = initial_centroids.shape[0]  # 类数
    centroids = initial_centroids   # 记录当前类中心
    previous_centroids = centroids  # 记录上一次类中心
    idx = np.zeros((m,1))           # 每条数据属于哪个类
    
    for i in range(max_iters):      # 迭代次数
        print u'迭代计算次数：%d'%(i+1)
        idx = findClosestCentroids(X, centroids)
        if plot_process:    # 如果绘制图像
            plt = plotProcessKMeans(X,centroids,previous_centroids) # 画聚类中心的移动过程
            previous_centroids = centroids  # 重置
        centroids = computerCentroids(X, idx, K)    # 重新计算类中心
    if plot_process:    # 显示最终的绘制结果
        plt.show()
    return centroids,idx    # 返回聚类中心和数据属于哪个类
```

### 6、[使用scikit-learn库中的线性模型实现聚类](/K-Means/K-Means_scikit-learn.py)

- 导入包
```
    from sklearn.cluster import KMeans
```
- 使用模型拟合数据
```
    model = KMeans(n_clusters=3).fit(X) # n_clusters指定3类，拟合数据
```
- 聚类中心
```
    centroids = model.cluster_centers_  # 聚类中心
```

### 7、运行结果
- 二维数据类中心的移动  
![enter description here][39]
- 图片压缩  
![enter description here][40]


----------------------

## 六、PCA主成分分析（降维）
- [全部代码](/PCA/PCA.py)

### 1、用处
- 数据压缩（Data Compression）,使程序运行更快
- 可视化数据，例如`3D-->2D`等
- ......

### 2、2D-->1D，nD-->kD
- 如下图所示，所有数据点可以投影到一条直线，是**投影距离的平方和**（投影误差）最小
![enter description here][41]
- 注意数据需要`归一化`处理
- 思路是找`1`个`向量u`,所有数据投影到上面使投影距离最小
- 那么`nD-->kD`就是找`k`个向量![$${u^{(1)}},{u^{(2)}} \ldots {u^{(k)}}$$](http://latex.codecogs.com/gif.latex?%24%24%7Bu%5E%7B%281%29%7D%7D%2C%7Bu%5E%7B%282%29%7D%7D%20%5Cldots%20%7Bu%5E%7B%28k%29%7D%7D%24%24)，所有数据投影到上面使投影误差最小
 - eg:3D-->2D,2个向量![$${u^{(1)}},{u^{(2)}}$$](http://latex.codecogs.com/gif.latex?%24%24%7Bu%5E%7B%281%29%7D%7D%2C%7Bu%5E%7B%282%29%7D%7D%24%24)就代表一个平面了，所有点投影到这个平面的投影误差最小即可

### 3、主成分分析PCA与线性回归的区别
- 线性回归是找`x`与`y`的关系，然后用于预测`y`
- `PCA`是找一个投影面，最小化data到这个投影面的投影误差

### 4、PCA降维过程
- 数据预处理（均值归一化）
 - 公式：![$${\rm{x}}_j^{(i)} = {{{\rm{x}}_j^{(i)} - {u_j}} \over {{s_j}}}$$](http://latex.codecogs.com/gif.latex?%24%24%7B%5Crm%7Bx%7D%7D_j%5E%7B%28i%29%7D%20%3D%20%7B%7B%7B%5Crm%7Bx%7D%7D_j%5E%7B%28i%29%7D%20-%20%7Bu_j%7D%7D%20%5Cover%20%7B%7Bs_j%7D%7D%7D%24%24)
 - 就是减去对应feature的均值，然后除以对应特征的标准差（也可以是最大值-最小值）
 - 实现代码：
 ```
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
 ```
- 计算`协方差矩阵Σ`（Covariance Matrix）：![$$\Sigma  = {1 \over m}\sum\limits_{i = 1}^n {{x^{(i)}}{{({x^{(i)}})}^T}} $$](http://latex.codecogs.com/gif.latex?%24%24%5CSigma%20%3D%20%7B1%20%5Cover%20m%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5En%20%7B%7Bx%5E%7B%28i%29%7D%7D%7B%7B%28%7Bx%5E%7B%28i%29%7D%7D%29%7D%5ET%7D%7D%20%24%24)
 - 注意这里的`Σ`和求和符号不同
 - 协方差矩阵`对称正定`（不理解正定的看看线代）
 - 大小为`nxn`,`n`为`feature`的维度
 - 实现代码：
 ```
 Sigma = np.dot(np.transpose(X_norm),X_norm)/m  # 求Sigma
 ```
- 计算`Σ`的特征值和特征向量
 - 可以是用`svd`奇异值分解函数：`U,S,V = svd(Σ)`
 - 返回的是与`Σ`同样大小的对角阵`S`（由`Σ`的特征值组成）[**注意**：`matlab`中函数返回的是对角阵，在`python`中返回的是一个向量，节省空间]
 - 还有两个**酉矩阵**U和V，且![$$\Sigma  = US{V^T}$$](http://latex.codecogs.com/gif.latex?%24%24%5CSigma%20%3D%20US%7BV%5ET%7D%24%24)
 - ![enter description here][42]
 - **注意**：`svd`函数求出的`S`是按特征值降序排列的，若不是使用`svd`,需要按**特征值**大小重新排列`U`
- 降维
 - 选取`U`中的前`K`列（假设要降为`K`维）
 - ![enter description here][43]
 - `Z`就是对应降维之后的数据
 - 实现代码：
 ```
     # 映射数据
    def projectData(X_norm,U,K):
        Z = np.zeros((X_norm.shape[0],K))
        
        U_reduce = U[:,0:K]          # 取前K个
        Z = np.dot(X_norm,U_reduce) 
        return Z
 ```
- 过程总结：
 - `Sigma = X'*X/m`
 - `U,S,V = svd(Sigma)`
 - `Ureduce = U[:,0:k]`
 - `Z = Ureduce'*x`

### 5、数据恢复
 - 因为：![$${Z^{(i)}} = U_{reduce}^T*{X^{(i)}}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7BZ%5E%7B%28i%29%7D%7D%20%3D%20U_%7Breduce%7D%5ET*%7BX%5E%7B%28i%29%7D%7D%24%24)
 - 所以：![$${X_{approx}} = {(U_{reduce}^T)^{ - 1}}Z$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7BX_%7Bapprox%7D%7D%20%3D%20%7B%28U_%7Breduce%7D%5ET%29%5E%7B%20-%201%7D%7DZ%24%24)     （注意这里是X的近似值）
 - 又因为`Ureduce`为正定矩阵，【正定矩阵满足：![$$A{A^T} = {A^T}A = E$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24A%7BA%5ET%7D%20%3D%20%7BA%5ET%7DA%20%3D%20E%24%24)，所以：![$${A^{ - 1}} = {A^T}$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7BA%5E%7B%20-%201%7D%7D%20%3D%20%7BA%5ET%7D%24%24)】，所以这里：
 - ![$${X_{approx}} = {(U_{reduce}^{ - 1})^{ - 1}}Z = {U_{reduce}}Z$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7BX_%7Bapprox%7D%7D%20%3D%20%7B%28U_%7Breduce%7D%5E%7B%20-%201%7D%29%5E%7B%20-%201%7D%7DZ%20%3D%20%7BU_%7Breduce%7D%7DZ%24%24)
 - 实现代码：
```
    # 恢复数据 
    def recoverData(Z,U,K):
        X_rec = np.zeros((Z.shape[0],U.shape[0]))
        U_recude = U[:,0:K]
        X_rec = np.dot(Z,np.transpose(U_recude))  # 还原数据（近似）
        return X_rec
```

### 6、主成分个数的选择（即要降的维度）
- 如何选择
 - **投影误差**（project error）：![$${1 \over m}\sum\limits_{i = 1}^m {||{x^{(i)}} - x_{approx}^{(i)}|{|^2}} $$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7B1%20%5Cover%20m%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%7C%7C%7Bx%5E%7B%28i%29%7D%7D%20-%20x_%7Bapprox%7D%5E%7B%28i%29%7D%7C%7B%7C%5E2%7D%7D%20%24%24)
 - **总变差**（total variation）:![$${1 \over m}\sum\limits_{i = 1}^m {||{x^{(i)}}|{|^2}} $$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7B1%20%5Cover%20m%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%7C%7C%7Bx%5E%7B%28i%29%7D%7D%7C%7B%7C%5E2%7D%7D%20%24%24)
 - 若**误差率**（error ratio）：![$${{{1 \over m}\sum\limits_{i = 1}^m {||{x^{(i)}} - x_{approx}^{(i)}|{|^2}} } \over {{1 \over m}\sum\limits_{i = 1}^m {||{x^{(i)}}|{|^2}} }} \le 0.01$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24%7B%7B%7B1%20%5Cover%20m%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%7C%7C%7Bx%5E%7B%28i%29%7D%7D%20-%20x_%7Bapprox%7D%5E%7B%28i%29%7D%7C%7B%7C%5E2%7D%7D%20%7D%20%5Cover%20%7B%7B1%20%5Cover%20m%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%7C%7C%7Bx%5E%7B%28i%29%7D%7D%7C%7B%7C%5E2%7D%7D%20%7D%7D%20%5Cle%200.01%24%24)，则称`99%`保留差异性
 - 误差率一般取`1%，5%，10%`等
- 如何实现
 - 若是一个个试的话代价太大
 - 之前`U,S,V = svd(Sigma)`,我们得到了`S`，这里误差率error ratio:    
 ![$$error{\kern 1pt} \;ratio = 1 - {{\sum\limits_{i = 1}^k {{S_{ii}}} } \over {\sum\limits_{i = 1}^n {{S_{ii}}} }} \le threshold$$](http://latex.codecogs.com/gif.latex?%5Cfn_cm%20%24%24error%7B%5Ckern%201pt%7D%20%5C%3Bratio%20%3D%201%20-%20%7B%7B%5Csum%5Climits_%7Bi%20%3D%201%7D%5Ek%20%7B%7BS_%7Bii%7D%7D%7D%20%7D%20%5Cover%20%7B%5Csum%5Climits_%7Bi%20%3D%201%7D%5En%20%7B%7BS_%7Bii%7D%7D%7D%20%7D%7D%20%5Cle%20threshold%24%24)
 - 可以一点点增加`K`尝试。

### 7、使用建议
- 不要使用PCA去解决过拟合问题`Overfitting`，还是使用正则化的方法（如果保留了很高的差异性还是可以的）
- 只有在原数据上有好的结果，但是运行很慢，才考虑使用PCA

### 8、运行结果
- 2维数据降为1维
 - 要投影的方向     
![enter description here][44]
 - 2D降为1D及对应关系        
![enter description here][45]
- 人脸数据降维
 - 原始数据         
 ![enter description here][46]
 - 可视化部分`U`矩阵信息    
 ![enter description here][47]
 - 恢复数据    
 ![enter description here][48]

### 9、[使用scikit-learn库中的PCA实现降维](/PCA/PCA.py_scikit-learn.py)
- 导入需要的包：
```
#-*- coding: utf-8 -*-
# Author:bob
# Date:2016.12.22
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio
from sklearn.decomposition import pca
from sklearn.preprocessing import StandardScaler
```
- 归一化数据
```
    '''归一化数据并作图'''
    scaler = StandardScaler()
    scaler.fit(X)
    x_train = scaler.transform(X)
```
- 使用PCA模型拟合数据，并降维
 - `n_components`对应要将的维度
```
    '''拟合数据'''
    K=1 # 要降的维度
    model = pca.PCA(n_components=K).fit(x_train)   # 拟合数据，n_components定义要降的维度
    Z = model.transform(x_train)    # transform就会执行降维操作
```

- 数据恢复
 - `model.components_`会得到降维使用的`U`矩阵 
```
    '''数据恢复并作图'''
    Ureduce = model.components_     # 得到降维用的Ureduce
    x_rec = np.dot(Z,Ureduce)       # 数据恢复
```



---------------------------------------------------------------


## 七、异常检测 Anomaly Detection
- [全部代码](/AnomalyDetection/AnomalyDetection.py)

### 1、高斯分布（正态分布）`Gaussian distribution` 
- 分布函数：![$$p(x) = {1 \over {\sqrt {2\pi } \sigma }}{e^{ - {{{{(x - u)}^2}} \over {2{\sigma ^2}}}}}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24p%28x%29%20%3D%20%7B1%20%5Cover%20%7B%5Csqrt%20%7B2%5Cpi%20%7D%20%5Csigma%20%7D%7D%7Be%5E%7B%20-%20%7B%7B%7B%7B%28x%20-%20u%29%7D%5E2%7D%7D%20%5Cover%20%7B2%7B%5Csigma%20%5E2%7D%7D%7D%7D%7D%24%24)
 - 其中，`u`为数据的**均值**，`σ`为数据的**标准差**
 - `σ`越**小**，对应的图像越**尖**
- 参数估计（`parameter estimation`）
 - ![$$u = {1 \over m}\sum\limits_{i = 1}^m {{x^{(i)}}} $$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24u%20%3D%20%7B1%20%5Cover%20m%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%7Bx%5E%7B%28i%29%7D%7D%7D%20%24%24)
 - ![$${\sigma ^2} = {1 \over m}\sum\limits_{i = 1}^m {{{({x^{(i)}} - u)}^2}} $$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%7B%5Csigma%20%5E2%7D%20%3D%20%7B1%20%5Cover%20m%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%7B%7B%28%7Bx%5E%7B%28i%29%7D%7D%20-%20u%29%7D%5E2%7D%7D%20%24%24)

### 2、异常检测算法
- 例子
 - 训练集：![$$\{ {x^{(1)}},{x^{(2)}}, \cdots {x^{(m)}}\} $$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%5C%7B%20%7Bx%5E%7B%281%29%7D%7D%2C%7Bx%5E%7B%282%29%7D%7D%2C%20%5Ccdots%20%7Bx%5E%7B%28m%29%7D%7D%5C%7D%20%24%24),其中![$$x \in {R^n}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24x%20%5Cin%20%7BR%5En%7D%24%24)
 - 假设![$${x_1},{x_2} \cdots {x_n}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%7Bx_1%7D%2C%7Bx_2%7D%20%5Ccdots%20%7Bx_n%7D%24%24)相互独立，建立model模型：![$$p(x) = p({x_1};{u_1},\sigma _1^2)p({x_2};{u_2},\sigma _2^2) \cdots p({x_n};{u_n},\sigma _n^2) = \prod\limits_{j = 1}^n {p({x_j};{u_j},\sigma _j^2)} $$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24p%28x%29%20%3D%20p%28%7Bx_1%7D%3B%7Bu_1%7D%2C%5Csigma%20_1%5E2%29p%28%7Bx_2%7D%3B%7Bu_2%7D%2C%5Csigma%20_2%5E2%29%20%5Ccdots%20p%28%7Bx_n%7D%3B%7Bu_n%7D%2C%5Csigma%20_n%5E2%29%20%3D%20%5Cprod%5Climits_%7Bj%20%3D%201%7D%5En%20%7Bp%28%7Bx_j%7D%3B%7Bu_j%7D%2C%5Csigma%20_j%5E2%29%7D%20%24%24)
- 过程
 - 选择具有代表异常的`feature`:xi
 - 参数估计：![$${u_1},{u_2}, \cdots ,{u_n};\sigma _1^2,\sigma _2^2 \cdots ,\sigma _n^2$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%7Bu_1%7D%2C%7Bu_2%7D%2C%20%5Ccdots%20%2C%7Bu_n%7D%3B%5Csigma%20_1%5E2%2C%5Csigma%20_2%5E2%20%5Ccdots%20%2C%5Csigma%20_n%5E2%24%24)
 - 计算`p(x)`,若是`P(x)<ε`则认为异常，其中`ε`为我们要求的概率的临界值`threshold`
- 这里只是**单元高斯分布**，假设了`feature`之间是独立的，下面会讲到**多元高斯分布**，会自动捕捉到`feature`之间的关系
- **参数估计**实现代码
```
# 参数估计函数（就是求均值和方差）
def estimateGaussian(X):
    m,n = X.shape
    mu = np.zeros((n,1))
    sigma2 = np.zeros((n,1))
    
    mu = np.mean(X, axis=0) # axis=0表示列，每列的均值
    sigma2 = np.var(X,axis=0) # 求每列的方差
    return mu,sigma2
```

### 3、评价`p(x)`的好坏，以及`ε`的选取
- 对**偏斜数据**的错误度量
 - 因为数据可能是非常**偏斜**的（就是`y=1`的个数非常少，(`y=1`表示异常)），所以可以使用`Precision/Recall`，计算`F1Score`(在**CV交叉验证集**上)
 - 例如：预测癌症，假设模型可以得到`99%`能够预测正确，`1%`的错误率，但是实际癌症的概率很小，只有`0.5%`，那么我们始终预测没有癌症y=0反而可以得到更小的错误率。使用`error rate`来评估就不科学了。
 - 如下图记录：    
 ![enter description here][49]
 - ![$$\Pr ecision = {{TP} \over {TP + FP}}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%5CPr%20ecision%20%3D%20%7B%7BTP%7D%20%5Cover%20%7BTP%20&plus;%20FP%7D%7D%24%24) ，即：**正确预测正样本/所有预测正样本**
 - ![$${\mathop{\rm Re}\nolimits} {\rm{call}} = {{TP} \over {TP + FN}}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%7B%5Cmathop%7B%5Crm%20Re%7D%5Cnolimits%7D%20%7B%5Crm%7Bcall%7D%7D%20%3D%20%7B%7BTP%7D%20%5Cover%20%7BTP%20&plus;%20FN%7D%7D%24%24) ，即：**正确预测正样本/真实值为正样本**
 - 总是让`y=1`(较少的类)，计算`Precision`和`Recall`
 - ![$${F_1}Score = 2{{PR} \over {P + R}}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%7BF_1%7DScore%20%3D%202%7B%7BPR%7D%20%5Cover%20%7BP%20&plus;%20R%7D%7D%24%24)
 - 还是以癌症预测为例，假设预测都是no-cancer，TN=199，FN=1，TP=0，FP=0，所以：Precision=0/0，Recall=0/1=0，尽管accuracy=199/200=99.5%，但是不可信。

- `ε`的选取
 - 尝试多个`ε`值，使`F1Score`的值高
- 实现代码
```
# 选择最优的epsilon，即：使F1Score最大    
def selectThreshold(yval,pval):
    '''初始化所需变量'''
    bestEpsilon = 0.
    bestF1 = 0.
    F1 = 0.
    step = (np.max(pval)-np.min(pval))/1000
    '''计算'''
    for epsilon in np.arange(np.min(pval),np.max(pval),step):
        cvPrecision = pval<epsilon
        tp = np.sum((cvPrecision == 1) & (yval == 1).ravel()).astype(float)  # sum求和是int型的，需要转为float
        fp = np.sum((cvPrecision == 1) & (yval == 0).ravel()).astype(float)
        fn = np.sum((cvPrecision == 0) & (yval == 1).ravel()).astype(float)
        precision = tp/(tp+fp)  # 精准度
        recision = tp/(tp+fn)   # 召回率
        F1 = (2*precision*recision)/(precision+recision)  # F1Score计算公式
        if F1 > bestF1:  # 修改最优的F1 Score
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon,bestF1
```

### 4、选择使用什么样的feature（单元高斯分布）
- 如果一些数据不是满足高斯分布的，可以变化一下数据，例如`log(x+C),x^(1/2)`等
- 如果`p(x)`的值无论异常与否都很大，可以尝试组合多个`feature`,(因为feature之间可能是有关系的)

### 5、多元高斯分布
- 单元高斯分布存在的问题
 - 如下图，红色的点为异常点，其他的都是正常点（比如CPU和memory的变化）   
 ![enter description here][50]
 - x1对应的高斯分布如下：   
 ![enter description here][51]
 - x2对应的高斯分布如下：   
 ![enter description here][52]
 - 可以看出对应的p(x1)和p(x2)的值变化并不大，就不会认为异常
 - 因为我们认为feature之间是相互独立的，所以如上图是以**正圆**的方式扩展
- 多元高斯分布
 - ![$$x \in {R^n}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24x%20%5Cin%20%7BR%5En%7D%24%24)，并不是建立`p(x1),p(x2)...p(xn)`，而是统一建立`p(x)`
 - 其中参数：![$$\mu  \in {R^n},\Sigma  \in {R^{n \times {\rm{n}}}}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%5Cmu%20%5Cin%20%7BR%5En%7D%2C%5CSigma%20%5Cin%20%7BR%5E%7Bn%20%5Ctimes%20%7B%5Crm%7Bn%7D%7D%7D%7D%24%24),`Σ`为**协方差矩阵**
 - ![$$p(x) = {1 \over {{{(2\pi )}^{{n \over 2}}}|\Sigma {|^{{1 \over 2}}}}}{e^{ - {1 \over 2}{{(x - u)}^T}{\Sigma ^{ - 1}}(x - u)}}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24p%28x%29%20%3D%20%7B1%20%5Cover%20%7B%7B%7B%282%5Cpi%20%29%7D%5E%7B%7Bn%20%5Cover%202%7D%7D%7D%7C%5CSigma%20%7B%7C%5E%7B%7B1%20%5Cover%202%7D%7D%7D%7D%7D%7Be%5E%7B%20-%20%7B1%20%5Cover%202%7D%7B%7B%28x%20-%20u%29%7D%5ET%7D%7B%5CSigma%20%5E%7B%20-%201%7D%7D%28x%20-%20u%29%7D%7D%24%24)
 - 同样，`|Σ|`越小，`p(x)`越尖
 - 例如：    
 ![enter description here][53]，  
 表示x1,x2**正相关**，即x1越大，x2也就越大，如下图，也就可以将红色的异常点检查出了
 ![enter description here][54]      
 若：   
  ![enter description here][55]，   
 表示x1,x2**负相关**
- 实现代码：
```
# 多元高斯分布函数    
def multivariateGaussian(X,mu,Sigma2):
    k = len(mu)
    if (Sigma2.shape[0]>1):
        Sigma2 = np.diag(Sigma2)
    '''多元高斯分布函数'''    
    X = X-mu
    argu = (2*np.pi)**(-k/2)*np.linalg.det(Sigma2)**(-0.5)
    p = argu*np.exp(-0.5*np.sum(np.dot(X,np.linalg.inv(Sigma2))*X,axis=1))  # axis表示每行
    return p
```
### 6、单元和多元高斯分布特点
- 单元高斯分布
 - 人为可以捕捉到`feature`之间的关系时可以使用
 - 计算量小
- 多元高斯分布
 - 自动捕捉到相关的feature
 - 计算量大，因为：![$$\Sigma  \in {R^{n \times {\rm{n}}}}$$](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%24%24%5CSigma%20%5Cin%20%7BR%5E%7Bn%20%5Ctimes%20%7B%5Crm%7Bn%7D%7D%7D%7D%24%24)
 - `m>n`或`Σ`可逆时可以使用。（若不可逆，可能有冗余的x，因为线性相关，不可逆，或者就是m<n）

### 7、程序运行结果
- 显示数据     
![enter description here][56]
- 等高线      
![enter description here][57]
- 异常点标注   
![enter description here][58]



----------------------------------


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
  [15]: ./images/NeuralNetwork_01.png "NeuralNetwork_01.png"
  [16]: ./images/NeuralNetwork_02.png "NeuralNetwork_02.png"
  [17]: ./images/NeuralNetwork_03.jpg "NeuralNetwork_03.jpg"
  [18]: ./images/NeuralNetwork_04.png "NeuralNetwork_04.png"
  [19]: ./images/NeuralNetwork_05.png "NeuralNetwork_05.png"
  [20]: ./images/NeuralNetwork_06.png "NeuralNetwork_06.png"
  [21]: ./images/NeuralNetwork_07.png "NeuralNetwork_07.png"
  [22]: ./images/NeuralNetwork_08.png "NeuralNetwork_08.png"
  [23]: ./images/NeuralNetwork_09.png "NeuralNetwork_09.png"
  [24]: ./images/SVM_01.png "SVM_01.png"
  [25]: ./images/SVM_02.png "SVM_02.png"
  [26]: ./images/SVM_03.png "SVM_03.png"
  [27]: ./images/SVM_04.png "SVM_04.png"
  [28]: ./images/SVM_05.png "SVM_05.png"
  [29]: ./images/SVM_06.png "SVM_06.png"
  [30]: ./images/SVM_07.png "SVM_07.png"
  [31]: ./images/SVM_08.png "SVM_08.png"
  [32]: ./images/SVM_09.png "SVM_09.png"
  [33]: ./images/SVM_10.png "SVM_10.png"
  [34]: ./images/K-Means_01.png "K-Means_01.png"
  [35]: ./images/K-Means_02.png "K-Means_02.png"
  [36]: ./images/K-Means_03.png "K-Means_03.png"
  [37]: ./images/K-Means_07.png "K-Means_07.png"
  [38]: ./images/K-Means_04.png "K-Means_04.png"
  [39]: ./images/K-Means_05.png "K-Means_05.png"
  [40]: ./images/K-Means_06.png "K-Means_06.png"
  [41]: ./images/PCA_01.png "PCA_01.png"
  [42]: ./images/PCA_02.png "PCA_02.png"
  [43]: ./images/PCA_03.png "PCA_03.png"
  [44]: ./images/PCA_04.png "PCA_04.png"
  [45]: ./images/PCA_05.png "PCA_05.png"
  [46]: ./images/PCA_06.png "PCA_06.png"
  [47]: ./images/PCA_07.png "PCA_07.png"
  [48]: ./images/PCA_08.png "PCA_08.png"
  [49]: ./images/AnomalyDetection_01.png "AnomalyDetection_01.png"
  [50]: ./images/AnomalyDetection_04.png "AnomalyDetection_04.png"
  [51]: ./images/AnomalyDetection_02.png "AnomalyDetection_02.png"
  [52]: ./images/AnomalyDetection_03.png "AnomalyDetection_03.png"
  [53]: ./images/AnomalyDetection_05.png "AnomalyDetection_05.png"
  [54]: ./images/AnomalyDetection_07.png "AnomalyDetection_07.png"
  [55]: ./images/AnomalyDetection_06.png "AnomalyDetection_06.png"
  [56]: ./images/AnomalyDetection_08.png "AnomalyDetection_08.png"
  [57]: ./images/AnomalyDetection_09.png "AnomalyDetection_09.png"
  [58]: ./images/AnomalyDetection_10.png "AnomalyDetection_10.png"
