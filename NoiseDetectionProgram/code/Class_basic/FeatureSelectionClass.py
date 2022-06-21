'''
Created on 2020年10月25日

@author: njdx
'''


import random
import numpy as np
from collections import deque
from treelib import Tree, Node
import copy
import Class_basic.PredictionModelClass as PMC
import rpy2.robjects as robjects#更改环境变量后，需要重启一下eclipse
from rpy2.robjects import pandas2ri


#特定指标导向的特征选择方法
#Oriented specified indicator feature selection method
class FeatureSelectionClass(object):
    
#     level = 0#当前层
    
    def __init__(self):
        self.tree = Tree()#存生成的路径树
        self.list_allUsedPaths = []#存所有已生成的路径
        self.list_allDatas = []#存所有已生成的路径的值
        self.level = 0#当前层
        self.df_original = []#数据集
        self.specifiedIndicator = ''#目标优化指标
        self.col_Label = ''#标签列
        self.Class_predictionModel =[];#需要使用该类中的预测函数
    
    def SetClassifiers(self,list_names_classifiers):
        self.Class_predictionModel = PMC.PredictionModelClass(list_names_classifiers);#需要使用该类中的预测函数
    
    # 调用R包GainRatio特征选择方法
    def GainRatio_from_R(self,df,col_Label):
        formula = col_Label + '~.'
        pandas2ri.activate()#不加这句话，下面一句会报错
        r_dataframe = pandas2ri.py2ri(df)
        res_Notdef=robjects.r.doGainRatio_mislabel(r_dataframe,formula)
        features_label = list(res_Notdef)
        features_label = ['NA' if i =='NA.' else i for i in features_label]
        features_label.append(col_Label);
        df_fs = df[features_label]
        return df_fs
    
    def FUNCTION_bulidTree(self,parentNode_outer,currentPath,candidatesSet_outer):
        #由于python判断传递时，一些变量类型是引用，不是值，因此需要做一步处理
        parentNode = copy.deepcopy(parentNode_outer)#对象拷贝，深拷贝
        candidatesSet = copy.copy(candidatesSet_outer)
        
        #以当前特征为起点，构建一个贪心策略生长的子树
        for i in range(len(candidatesSet)):  # @UnusedVariable
            #当前路径加入一个节点
            feature_current = candidatesSet.popleft()#队首弹出元素，并返回
            
            #判断新生成的路径之前是否已经生成过
            temp_path = copy.copy(currentPath)
            temp_path.append(feature_current)#当前正在生成的路径
            temp_set_path = set(temp_path)#路径转成集合
            if temp_set_path in self.list_allUsedPaths:
                continue
            
            #比较性能
            oldPerformance = parentNode.data
            newPerformance = self.Class_predictionModel.\
            FUNCTION_CaculatePredictionProbability_crossValidation_specifiedIndicator\
            (self.df_original,list(temp_set_path),self.specifiedIndicator,self.col_Label)#计算新性能
            
            if newPerformance > oldPerformance:
                currentPath.append(feature_current)#当前正在生成的路径
                z = list(currentPath)
                rot = 0
                str_z = z[rot:]+z[:rot]
                node = Node(tag=str_z, data=newPerformance)
                self.tree.add_node(node, parent=parentNode)
                self.list_allUsedPaths.append(temp_set_path)#保存已生成的当前路径
                self.list_allDatas.append(newPerformance)#保存已生成的当前路径对应的指标值
                #递归调用
                if len(candidatesSet) > 0:
                    parentNode_in = node#当前节点加入路径树后，变为下一个节点的父节点
                    self.level += 1
#                     print("Current level: %d, Current parent node: %s"%(self.level,parentNode_in.tag))
                    self.FUNCTION_bulidTree(parentNode_in,currentPath,candidatesSet);
                elif len(currentPath) != 1 and len(candidatesSet) == 0:
                    currentPath.pop()#移除当前路径新增的最后一个叶节点，返回上一个节点
                    currentPath.pop()#当前路径移除一个节点，返回上一个节点
                    self.level -= 1
                    return;
                elif len(currentPath) == 1 and len(candidatesSet) == 0:
                    currentPath.pop()#移除当前路径新增的最后一个节点，返回上一个节点
                    self.level -= 1
                    return;
            elif newPerformance < oldPerformance:
                self.list_allUsedPaths.append(temp_set_path)#保存已生成的当前路径
                self.list_allDatas.append(newPerformance)#保存已生成的当前路径对应的指标值
        #遍历完候选集，当前路径栈移除最后进入的节点，接着生成
        currentPath.pop()
        self.level -= 1
        return;
        
    def FUNCTION_featureSelection(self,df_original,specifiedIndicator,col_Label):
        self.df_original = df_original
        self.specifiedIndicator = specifiedIndicator
        self.col_Label = col_Label
        #获取特征名
        list_featureNames = list(df_original)
        list_featureNames.remove(col_Label)
#         list_featureNames.sort()
#         list_featureNames = ['a','b','c','d','e','f']
        
        self.tree = Tree()#存生成的路径树
        self.list_allUsedPaths = []#存所有已生成的路径
        self.list_allDatas = []#存所有已生成的路径的值
        self.level = 0#当前层
        
        newPerformance = self.Class_predictionModel.\
            FUNCTION_CaculatePredictionProbability_crossValidation_specifiedIndicator\
            (df_original,list_featureNames,specifiedIndicator,col_Label)#计算新性能
        self.list_allUsedPaths.append(set(list_featureNames))#保存已生成的当前路径
        self.list_allDatas.append(newPerformance)#保存已生成的当前路径对应的指标值
        
        #---start---#
        #构建特征选择树：启发式构建
        header = "NA"
        root = Node(tag=header,data=0)
        self.tree.add_node(root)# root node
        currentPath = deque([])#存当前生成的路径的栈
        candidatesSet = deque(list_featureNames)
#         print("Current level: %d, Current parent node: %s"%(self.level,"NA"))
        self.FUNCTION_bulidTree(root,currentPath,candidatesSet);
#         self.tree.show();
        
        #---rank---#
        #使性能最大的排在最前面
        paths_ranked = [x for _,x in sorted(zip(self.list_allDatas,self.list_allUsedPaths),reverse = True)]
        datas_ranked = [y for y,_ in sorted(zip(self.list_allDatas,self.list_allUsedPaths),reverse = True)]
        #如果最大值的路径有多大，使路径最短的排前面
        datas_ranked = np.array(datas_ranked)
        paths_ranked = np.array(paths_ranked)
        indexes_maxPerformance = np.where(datas_ranked == datas_ranked[0])[0]
        paths_maxPerformance = paths_ranked[indexes_maxPerformance]
        len_paths_maxPerformance = [len(x) for x in paths_maxPerformance]
        paths_maxPerformance_ranked = [x for _,x in sorted(zip(len_paths_maxPerformance,paths_maxPerformance),reverse=True)]
        features_selected = list(paths_maxPerformance_ranked[0])
        print(features_selected,datas_ranked[0])
        return features_selected,[datas_ranked[0]]
    
    def FUNCTION_bulidTree_test(self,parentNode_outer,currentPath,candidatesSet_outer):
        #由于python判断传递时，一些变量类型是引用，不是值，因此需要做一步处理
        parentNode = copy.deepcopy(parentNode_outer)#对象拷贝，深拷贝
        candidatesSet = copy.copy(candidatesSet_outer)
        
        #以当前特征为起点，构建一个贪心策略生长的子树
        for i in range(len(candidatesSet)):  # @UnusedVariable
            #当前路径加入一个节点
            feature_current = candidatesSet.popleft()#队首弹出元素，并返回
            
            #判断新生成的路径之前是否已经生成过
            temp_path = copy.copy(currentPath)
            temp_path.append(feature_current)#当前正在生成的路径
            temp_set_path = set(temp_path)#路径转成集合
            if temp_set_path in self.list_allUsedPaths:
                continue
            
            #比较性能
            oldPerformance = parentNode.data
            newPerformance = random.random()#计算新性能
            
            if newPerformance > oldPerformance:
                currentPath.append(feature_current)#当前正在生成的路径
                z = list(currentPath)
                rot = 0
                str_z = z[rot:]+z[:rot]
                node = Node(tag=str_z, data=newPerformance)
                self.tree.add_node(node, parent=parentNode)
                self.list_allUsedPaths.append(temp_set_path)#保存已生成的当前路径
                self.list_allDatas.append(newPerformance)#保存已生成的当前路径对应的指标值
                #递归调用
                if len(candidatesSet) > 0:
                    parentNode_in = node#当前节点加入路径树后，变为下一个节点的父节点
                    self.level += 1
#                     print("Current level: %d, Current parent node: %s"%(self.level,parentNode_in.tag))
                    self.FUNCTION_bulidTree_test(parentNode_in,currentPath,candidatesSet);
                elif len(currentPath) != 1 and len(candidatesSet) == 0:
                    currentPath.pop()#移除当前路径新增的最后一个叶节点，返回上一个节点
                    currentPath.pop()#当前路径移除一个节点，返回上一个节点
                    self.level -= 1
                    return;
                elif len(currentPath) == 1 and len(candidatesSet) == 0:
                    currentPath.pop()#移除当前路径新增的最后一个节点，返回上一个节点
                    self.level -= 1
                    return;
            elif newPerformance < oldPerformance:
                self.list_allUsedPaths.append(temp_set_path)#保存已生成的当前路径
                self.list_allDatas.append(newPerformance)#保存已生成的当前路径对应的指标值
        #遍历完候选集，当前路径栈移除最后进入的节点，接着生成
        currentPath.pop()
        self.level -= 1
        return;
    
    def FUNCTION_featureSelection_test(self, df_original,specifiedIndicator):
#         #标签
#         y_labels = df_original['mislabel'].values;
#         #获取特征名
#         list_featureNames = list(df_original)
#         list_featureNames.remove('mislabel')
#         list_featureNames.sort()
        list_featureNames = ['a','b','c','d','e','f']
        
        self.list_allUsedPaths = []#存所有已生成的路径
        self.list_allDatas = []#存所有已生成的路径的值
        self.level = 0#当前层
        
        #---start---#
        #构建特征选择树：启发式构建
        header = "NA"
        root = Node(tag=header,data=0)
        self.tree = Tree()
        self.tree.add_node(root)# root node
        currentPath = deque([])#存当前生成的路径的栈
        list_featureNames.sort()
        candidatesSet = deque(list_featureNames)
        print("Current level: %d, Current parent node: %s"%(self.level,"NA"))
        self.FUNCTION_bulidTree(root,currentPath,candidatesSet);
        self.tree.show();
        
        #---rank---#
        #使性能最大的排在最前面
        paths_ranked = [x for _,x in sorted(zip(self.list_allDatas,self.list_allUsedPaths),reverse = True)]
        datas_ranked = [y for y,_ in sorted(zip(self.list_allDatas,self.list_allUsedPaths),reverse = True)]
        #如果最大值的路径有多大，使路径最短的排前面
        datas_ranked = np.array(datas_ranked)
        paths_ranked = np.array(paths_ranked)
        indexes_maxPerformance = np.where(datas_ranked == datas_ranked[0])[0]
        paths_maxPerformance = paths_ranked[indexes_maxPerformance]
        len_paths_maxPerformance = [len(x) for x in paths_maxPerformance]
        paths_maxPerformance_ranked = [x for _,x in sorted(zip(len_paths_maxPerformance,paths_maxPerformance))]
        features_selected = list(paths_maxPerformance_ranked[0])
        print(features_selected)
        



# if __name__=='__main__':
#     random.seed(1);#不同随机种子生成的十折不一样
#     Class_FS = Class_FS();
#     Class_FS.FUNCTION_featureSelection([],'')
    
    
    
    
    
    