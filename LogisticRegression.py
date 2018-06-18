__author__ = 'Administor'

import numpy as np
import math
import csv
import myLog


class logistic_regression(object):

    '''
    类的属性：
    w: 权值向量
    b:偏置
    dimension:维度
    learning_rate:学习率
    circus:学习轮次
    batch_size:一个batch的大小
    max:最大最小标准化
    min:同上
    '''

    def __init__(self, dimension, batch_size):
        self.w = np.random.random(size=dimension)
        self.b = 0.0
        self.dimension = dimension
        self.learning_rate = np.full(dimension+1,0.3).flatten()#学习率暂时设置为0.1
        self.circus = 2000
        self.log = myLog.MyLog()
        self.batch_size = batch_size
        self.max = None
        self.min = None
    '''
    学习一个batch
    输入：一个batch的数据，其中每一行代表一个元素。每一列代表某个属性
    输出：
   '''
    def batch_learning(self,batch_data,batch_label):
       # if(batch_data.shape[1]!=batch_label.shape[1] or batch_label.shape[1]!= self.dimension):
        #    self.log.error("数据维度不一致")
        Loss = 0.0 #损失函数，采用交叉熵
        for j in range(self.dimension):    #每一维度中
            #计算梯度
            grad = 0.0
            for k in range(batch_data.shape[0]):
                grad = grad-(batch_label[k]-self.w[j])*batch_data[k][j]
            #更新权重
            grad = grad/batch_data.shape[0]
            self.w[j] = self.w[j]-self.learning_rate[j]*grad
        #更新b
        grad = 0.0
        for k in range(batch_data.shape[0]):
            grad = grad-(batch_label[k]-self.b)*1
        grad = grad/batch_data.shape[0]
        self.b = self.b - self.learning_rate[self.dimension] * grad

    def train(self,dataSet, labelSet,testData,testLabel):
        self.max, self.min = self.normPara(dataSet)
        dataSet = self.normalize(dataSet)
        #循环轮次
        for i in range(self.circus):
            #将一个数据内分成许多batch来循环
            for batch in range(0,dataSet.shape[0],self.batch_size):
                if(batch+self.batch_size<dataSet.shape[0]):
                    self.batch_learning(dataSet[batch:batch+self.batch_size-1], labelSet[batch:batch+self.batch_size-1])
                else:
                    self.batch_learning(dataSet[batch:],labelSet[batch:])
            if(i%50==0):
                self.test(testData,testLabel)


    def compute_probability(self,elem):
        if self.dimension != len(elem): #维度不一致则返回-1
            return -1
        num = 0.0
        for i in range(self.dimension):
            num += (self.w[i]*elem[i])
        num = num + self.b
        #防止计算出现溢出
        if(num > 10) :
            number = 10
        if (num <-10):
            num = -10
        return 1/(1+math.exp(-num))

    def normPara(self, dataSet):
        max = np.max(dataSet,axis=0)
        min = np.min(dataSet,axis=0)
        return max,min

    def normalize(self, dataSet):
        max = np.tile(self.max,(dataSet.shape[0],1))
        min = np.tile(self.min,(dataSet.shape[0],1))
        return (dataSet-min)/(max-min)

    def test(self,testData, testLabel):
        self.normalize(testData)
        ok_number = 0.0
        for i in range(testData.shape[0]):
            prob = self.compute_probability(testData[i])  #计算概率
            ok_string = ""
            if(prob>=0.5 and testLabel[i]==1) or (prob<0.5 and testLabel[i]==0):
                ok_string = "正确"
                ok_number += 1
            else:
                ok_string = "错误"

            print("第"+str(i+1)+"个元素是类别1的概率为"+str(prob)+"，该元素真实类别为"+str(testLabel[i])+ok_string)
        print("\n\n正确率为："+str(ok_number/len(testLabel)))

if __name__=="__main__":
        #打开数据集
    filename = "X_train"
    tmp = np.loadtxt("X_train", dtype = np.str, delimiter=",")
    dataSet = tmp[1:,:].astype(np.float)#加载数据部分（第一行是标题，不导入，第一列是类别名称）
    tmp = np.loadtxt("Y_train", dtype = np.str, delimiter=",")
    Labelset = tmp[1:].astype(np.float)#加载类别标签部分
    Labelset = np.reshape(Labelset,(1,-1)).flatten()
    tmp = np.loadtxt("X_test", dtype = np.str, delimiter=",")
    testSet = tmp[1:100].astype(np.float)#加载测试数据部分
    tmp = np.loadtxt("Y_test", dtype = np.str, delimiter=",")
    testLabelset = tmp[1:100,1].astype(np.float)#加载测试类别标签部分
    testLabelset = np.reshape(testLabelset,(1,-1)).flatten()

    Log_Reg = logistic_regression(dataSet.shape[1],100)
    Log_Reg.train(dataSet,Labelset,testSet,testLabelset)
    Log_Reg.test(testSet,testLabelset)










