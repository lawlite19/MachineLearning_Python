from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np

def logisticRegression():
    data = loadtxtAndcsv_data("data1.txt", ",", np.float64)
    X = data[:,0:-1]
    y = data[:,-1]

    # ����Ϊѵ�����Ͳ��Լ�
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    # ��һ��
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    #�߼��ع�
    model = LogisticRegression()
    model.fit(x_train,y_train)

    # Ԥ��
    predict = model.predict(x_test)
    right = sum(predict == y_test)

    predict = np.hstack((predict.reshape(-1,1),y_test.reshape(-1,1)))   # ��Ԥ��ֵ����ʵֵ����һ�飬�ù۲�
    print(predict)
    print('���Լ�׼ȷ�ʣ�%f%%'%(right*100.0/predict.shape[0]))          #�����ڲ��Լ��ϵ�׼ȷ��

# ����txt��csv�ļ�
def loadtxtAndcsv_data(fileName,split,dataType):
    return np.loadtxt(fileName,delimiter=split,dtype=dataType)

# ����npy�ļ�
def loadnpy_data(fileName):
    return np.load(fileName)


if __name__ == "__main__":
    logisticRegression()
