'''
Created on 2020年6月23日

@author: njdx
'''

import pandas as pd
import numpy as np
import copy
import Class_basic.class_MyKNN as KNN

class MyCLNI(object):
    
    delta = 0.6;#常数，邻近实例中标签不同的实例占前N个的比例
    epsilon = 0.99;#常数
    N=5;#前5个最近的实例
    Class_KNN = KNN.MyKNN();#KNN方法选取最近邻
    
    #参数1表示类自身；参数2表示所有实例的度量数组；参数3表示所有实例的标签
    def FUNCTION_CLNI(self,array_metrics,labels):
        df_file = pd.DataFrame(array_metrics);#转成dataframe方便计算
        """ CLNI算法主函数"""
        num_iteration=10;#迭代次数
        set_noise_A = set();#存每轮迭代（j）的噪音
        set_noise_A_last = set();#存每轮迭代（j-1）的噪音
        array_interPercent = np.zeros(num_iteration);#因为原始论文中的退出条件达不到，因此记录十次的epsilon值
        list_noise_A_all = [];#因为原始论文中的退出条件达不到，因此记录十次的epsilon值对应的检测出的噪音下标
        for j in range(num_iteration):#原始论文中的j循环，然而原论文没有给出j的值。
#             print(j)
            set_noise_A = set();
            for i in range(len(df_file)):
                label_iRow = labels[i];#当前实例的标签
                iRow = df_file.iloc[[i]].values;#当前实例的一行。双括号返回的是dataframe，单括号返回的是series
                df_otherInstances = df_file.drop(set_noise_A_last,axis=0);#原始论文中的if(Inst_k∈Aj-1)：其他实例中，移除上一轮发现的噪音实例
                if i not in set_noise_A_last:
                    df_otherInstances = df_otherInstances.drop(i,axis=0);#其他实例中，移除当前实例
                array_otherInstances = df_otherInstances.values;
                list_row_nearest_N = self.Class_KNN.FUNCTION_KNN(iRow, array_otherInstances, k=self.N);#返回当前实例的前N个最近邻的实例的行号
                num_differentLabel = 0;
                for i_nearest in list_row_nearest_N:
                    if labels[i_nearest] != label_iRow:
                        num_differentLabel += 1;
                theta = num_differentLabel / self.N;#原始论文中的θ
                if theta >= self.delta:#原始论文中的θ,δ
                    set_noise_A.add(i);
            set_nosie_intersection = set_noise_A & set_noise_A_last#原始论文中Aj∩Aj-1
            inter_percent = len(set_nosie_intersection) / max(len(set_noise_A),len(set_noise_A_last))#原始论文中(Aj∩Aj-1)/MAX(Aj,Aj-1)
            if inter_percent >= self.epsilon:#很多数据集对于这个阈值永远达不到，并且可以通过举例证明这一点。
                array_interPercent[j] = inter_percent;#因为原始论文中的退出条件达不到，因此记录十次的epsilon值
                list_noise_A_all.append(set_noise_A);#存储每一次计算出的噪音下标                
                break;
            set_noise_A_last = copy.deepcopy(set_noise_A);#原始论文中Aj-1存储上一次检测出的噪音
            
            array_interPercent[j] = inter_percent;#因为原始论文中的退出条件达不到，因此记录十次的epsilon值
            list_noise_A_all.append(set_noise_A);#存储每一次计算出的噪音下标
        max_interPercent = np.max(array_interPercent);#十次的epsilon值中的最大值
        position_max = np.where(array_interPercent==max_interPercent)[0];#最大值在list中的下标
        position_max = position_max[0];#可能有多个epsilon值相等的最大值，取其中一个
        set_noise_A = list_noise_A_all[position_max];#取该epsilon值对应的检测出的噪音下标
        list_noise_A = list(set_noise_A);
        array_noise_A = np.array(list_noise_A);#list转换成array，计算快
        #print(j,max_interPercent)
        return array_noise_A;#返回噪音下标
    

    


