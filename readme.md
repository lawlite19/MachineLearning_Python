机器学习算法Python实现
=========

## 一、线性回归

### 1、代价函数
- ![J(\theta ) = \frac{1}{{2{\text{m}}}}\sum\limits_{i = 1}^m {{{({h_\theta }({x^{(i)}}) - {y^{(i)}})}^2}} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=J%28%5Ctheta%20%29%20%3D%20%5Cfrac%7B1%7D%7B%7B2%7B%5Ctext%7Bm%7D%7D%7D%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%7B%7B%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%20-%20%7By%5E%7B%28i%29%7D%7D%29%7D%5E2%7D%7D%20)
- 其中：
![{h_\theta }(x) = {\theta _0} + {\theta _1}{x_1} + {\theta _2}{x_2} + ...](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7Bh_%5Ctheta%20%7D%28x%29%20%3D%20%7B%5Ctheta%20_0%7D%20%2B%20%7B%5Ctheta%20_1%7D%7Bx_1%7D%20%2B%20%7B%5Ctheta%20_2%7D%7Bx_2%7D%20%2B%20...)

- 下面就是要求出theta，是代价最小，即代表我们拟合出来的方程距离真实值最近
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
- 代价函数对![{{\theta _j}}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%7B%5Ctheta%20_j%7D%7D)求偏导得到：![\frac{{\partial J(\theta )}}{{\partial {\theta _j}}} = \frac{1}{m}\sum\limits_{i = 1}^m {[({h_\theta }({x^{(i)}}) - {y^{(i)}})x_j^{(i)}]} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Cfrac%7B%7B%5Cpartial%20J%28%5Ctheta%20%29%7D%7D%7B%7B%5Cpartial%20%7B%5Ctheta%20_j%7D%7D%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%20-%20%7By%5E%7B%28i%29%7D%7D%29x_j%5E%7B%28i%29%7D%5D%7D%20)
- 所以对theta的更新可以写为：![{\theta _j} = {\theta _j} - \alpha \frac{1}{m}\sum\limits_{i = 1}^m {[({h_\theta }({x^{(i)}}) - {y^{(i)}})x_j^{(i)}]} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Ctheta%20_j%7D%20%3D%20%7B%5Ctheta%20_j%7D%20-%20%5Calpha%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5Em%20%7B%5B%28%7Bh_%5Ctheta%20%7D%28%7Bx%5E%7B%28i%29%7D%7D%29%20-%20%7By%5E%7B%28i%29%7D%7D%29x_j%5E%7B%28i%29%7D%5D%7D%20)
- 其中![\alpha ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Calpha%20)为学习速率，控制梯度下降的速度，一般取**0.01,0.03,0.1,0.3.....**