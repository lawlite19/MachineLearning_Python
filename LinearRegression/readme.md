线性回归算法
=======
### 一、文件说明
- [main.m][1.1]
 - 主运行程序
- [featureNormalize.m][1.2]
 - 特征向量归一化函数
- [gradientDescent.m][1.3]
 - 梯度下降求解函数
- [computerCost.m][1.4]
 - 计算代价J函数
- [normalEquations.m][1.5]
 - 正规方程求解函数

### 二、重要文件说明
- main.m
 - 注意合理修改学习速率参数`alpha`
 - 归一化数据，预测时也需要归一化（因为是归一化之后求出的theta参数）

### 三、测试数据
- 代价函数随迭代次数收敛
![线性回归][3.1]





[1.1]:main.m
[1.2]:featureNormalize.m
[1.3]:gradientDescent.m
[1.4]:computerCost.m
[1.5]:normalEquations.m


  [3.1]: ../images/LinearRegression_01.png "LinearRegression_01.png"
