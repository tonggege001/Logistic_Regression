__author__ = 'Administor'
import numpy as np
import math
import csv

#全局变量


'''
函数名称：一元逻辑回归
输入：数据集，标签集
输出：向量w和常数b
'''
def logistic_regression(dataSet,labelSet):
    print(labelSet)
   # if dataSet.empty():
   #     print("数据长度不一致")
    w = np.full((1,len(dataSet[0])),1.0).flatten() #权重向量
    b = 0.0 #平衡因子
    Loss = 0.0 #损失函数，采用交叉熵
    rate = np.full((1,len(w)+1),0.001).flatten()#学习率暂时设置为0.1
    circus = 2000 #学习的轮次

    for i in range(circus): #每一轮次中
        for j in range(len(w)):    #每一维度中
            #计算梯度
            grad = 0.0
            for k in range(dataSet.shape[0]):
                grad = grad-(labelSet[k]-w[j])*dataSet[k][j]
            #更新权重
            w[j] = w[j]-rate[j]*grad
        #更新b
        grad = 0.0
        for k in range(len(dataSet)):
            grad = grad-(labelSet[k]-b)*1
        b = b-rate[len(w)-1]*grad
    return w,b


'''
函数名称:testModel
函数功能：测试模型
输入：测试数据集，测试数据集的标签，权重向量，偏置
'''
def testModel(testSet,testLabel,w,b):
    ok_number = 0.0
    for i in range(len(testSet)):
        prob = f_w_b(w,b,testSet[i])  #计算概率
        ok_string = ""
        if(prob>=0.5 and testLabel[i]==1) or (prob<0.5 and testLabel[i]==0):
            ok_string = "正确"
            ok_number += 1
        else:
            ok_string = "错误"

        print("第"+str(i+1)+"个元素是类别1的概率为"+str(prob)+"，该元素真实类别为"+str(testLabel[i])+ok_string)
    print("\n\n正确率为："+str(ok_number/len(testLabel)))
    return
'''
函数名称:f_w_b
函数功能：计算当前元素的函数值
返回值：函数值
'''
def f_w_b(w,b,elem):
    if len(w) != len(elem): #维度不一致则返回0.0
        return 0.0
    else:
        num = 0.0
        for i in range(len(w)):
            num += (w[i]*elem[i])
        num = num + b
        print("num is"+str(num))
        if(num > 10) :
            number = 10
        if (num <-10):
            num = -10


        return 1/(1+math.exp(-num))

'''
正则化数据
输入：数据集，每一行代表一个数据元素，每一列代表一个属性
'''
def normalize(dataSet,testSet):
    X_train_test = np.concatenate((dataSet,testSet))
    mu = (sum(X_train_test)/X_train_test.shape[0])  #除以行数，也即除以元素个数
    sigma = np.std(X_train_test,axis=0)
    for i in range(len(sigma)):
        if sigma[i]==0 :
            sigma[i] = 1
    '''
    '''
    mu = np.tile(mu,(X_train_test.shape[0],1))
    sigma = np.tile(sigma,(X_train_test.shape[0],1))
    sigma = sigma

    X_train_test_normed = (X_train_test-mu)/sigma
    dataSet = X_train_test_normed[0:dataSet.shape[0]]
    testSet = X_train_test_normed[dataSet.shape[0]:]
    print(dataSet)
    print(testSet)
    return dataSet,testSet

if __name__ == "__main__":
    #打开数据集
    filename = "X_train"
    tmp = np.loadtxt("X_train", dtype = np.str, delimiter=",")
    dataSet = tmp[1:200,:].astype(np.float)#加载数据部分（第一行是标题，不导入，第一列是类别名称）
    tmp = np.loadtxt("Y_train", dtype = np.str, delimiter=",")
    Labelset = tmp[1:200].astype(np.float)#加载类别标签部分
    Labelset = np.reshape(Labelset,(1,-1)).flatten()
    tmp = np.loadtxt("X_test", dtype = np.str, delimiter=",")
    testSet = tmp[1:30].astype(np.float)#加载测试数据部分
    tmp = np.loadtxt("Y_test", dtype = np.str, delimiter=",")
    testLabelset = tmp[1:30,1].astype(np.float)#加载测试类别标签部分
    testLabelset = np.reshape(testLabelset,(1,-1)).flatten()
    print(testLabelset)

    print(dataSet.shape)
    print(Labelset.shape)
    print(testSet.shape)
    print(testLabelset.shape)

    dataSet,testSet = normalize(dataSet,testSet)#正则化数据集

    w,b = logistic_regression(dataSet,Labelset)
    testModel(testSet,testLabelset,w,b)