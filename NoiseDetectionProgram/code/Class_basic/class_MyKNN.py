'''
Created on 2020年6月23日

@author: njdx
'''

import numpy as np

class MyKNN(object):
        
    def myED(self,currentInstances,otherInstances):
        """ 计算欧式距离，要求当前类别样本和其他类别样本以array([ [],[],...[] ])的形式组织，每行表示一个样本，一列表示一个属性
                         返回的distances中的每一列是当前类别样本中的一行（即一个实例）相较于其他类别样本的每一行（即每一个实例）的欧式距离
        """
        size_train=otherInstances.shape[0] # 训练样本量大小
        size_test=currentInstances.shape[0] # 测试样本大小
        XX=otherInstances**2
        sumXX=XX.sum(axis=1) # 行平方和
        YY=currentInstances**2
        sumYY=YY.sum(axis=1) # 行平方和
        Xpw2_plus_Ypw2=np.tile(np.mat(sumXX).T,[1,size_test])+np.tile(np.mat(sumYY),[size_train,1])
        EDsq=Xpw2_plus_Ypw2-2*(np.mat(otherInstances)*np.mat(currentInstances).T) # 欧式距离平方
        distances=np.array(EDsq)**0.5 #欧式距离
        return distances

    def FUNCTION_KNN(self,currentInstances,otherInstances,k=1):
        """ kNN算法主函数"""
        D=self.myED(currentInstances,otherInstances)
        Dsortindex=D.argsort(axis=0) # 距离排序，提取序号
        nearest_k=Dsortindex[0:k,:] # 每一列是提取的每个当前实例的最近k个距离的样本序号。k=1时，则是距离最近的实例。
        nearest_k = nearest_k.flatten(); # 把数组降到一维，默认是按行的方向降 
        list_nearest_k = nearest_k.tolist();
        #返回每个实例最近的实例号
        return list_nearest_k
    
    def FUNCTION_KNN_avgD(self,currentInstances,otherInstances,k=1):
        """ kNN算法主函数"""
        D=self.myED(currentInstances,otherInstances)
        D = D.flatten(); # 把数组降到一维，默认是按行的方向降 
        D.sort()
        D_k = D[:k]
        DC = D_k.sum()/k
        return DC
#         Dsortindex=D.argsort(axis=0) # 距离排序，提取序号
#         nearest_k=Dsortindex[0:k,:] # 每一列是提取的每个当前实例的最近k个距离的样本序号。k=1时，则是距离最近的实例。
#         nearest_k = nearest_k.flatten(); # 把数组降到一维，默认是按行的方向降 
#         list_nearest_k = nearest_k.tolist();
#         #返回每个实例最近的实例号
#         return list_nearest_k
    
    


