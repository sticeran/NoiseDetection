'''
Created on 2020年10月25日

@author: njdx
'''


import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import ADASYN
# from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek


#Rebalancing technique
class RebalancingClass(object):
    df_data = [];
    
#     def __init__(self,df_data):
#         self.df_data = df_data;
    
    #===This function gets prediction probability===#
    def FUNCTION_Rebalancing(self,df,col_Label):
        df_metrics_only = df.drop([col_Label],axis=1);#删除不需要的列
        X_array_metrics_only = df_metrics_only.values;#转成numpy数组
        y_labels = df[col_Label].values;
        
        y_labels_1 = y_labels[y_labels == 1]
        len_y_labels_1 = len(y_labels_1)
        y_labels_0 = y_labels[y_labels == 0]
        len_y_labels_0 = len(y_labels_0)
        #选取再平衡技术
        rebalanceTechnique = SMOTE(random_state=0)
#         rebalanceTechnique = ADASYN(random_state=0)
#         rebalanceTechnique = SMOTEENN(random_state=0)
#         if len_y_labels_1 < 6:
#             rebalanceTechnique1 = SMOTE(random_state=0, k_neighbors=len_y_labels_1-1,)
#             rebalanceTechnique = SMOTETomek(random_state=0, smote = rebalanceTechnique1)
#         else:
#             rebalanceTechnique = SMOTETomek(random_state=0)
        
        if len_y_labels_1 > 5 and len_y_labels_0 > 5:
            # 对训练数据集作平衡处理 
            over_samples_X,over_samples_y = rebalanceTechnique.fit_sample(X_array_metrics_only, y_labels)
        else:
            over_samples_X,over_samples_y = (X_array_metrics_only, y_labels)
        # 按照特征，标签顺序合并
        over_samples_y = over_samples_y.reshape(over_samples_y.shape[0],1)#对一维数组进行转置需要借助reshape来完成
        array_merge = np.concatenate((over_samples_X,over_samples_y),axis=1);
        list_colNames = df.columns.tolist();
        df_result = pd.DataFrame(array_merge, columns = list_colNames);
        
        return df_result;
    #===end===#
    
    #===This function gets prediction probability===#
    def FUNCTION_Rebalancing_SMOTETomek(self,X_array_metrics_only,y_labels):
#         df_metrics_only = df.drop([col_Label],axis=1);#删除不需要的列
#         X_array_metrics_only = df_metrics_only.values;#转成numpy数组
#         y_labels = df[col_Label].values;
        
        y_labels_1 = y_labels[y_labels == 1]
        len_y_labels_1 = len(y_labels_1)
        #选取再平衡技术
#         rebalanceTechnique = SMOTE(random_state=0)
#         rebalanceTechnique = ADASYN(random_state=0)
#         rebalanceTechnique = SMOTEENN(random_state=0)
        if len_y_labels_1 < 6:
            rebalanceTechnique1 = SMOTE(random_state=0, k_neighbors=len_y_labels_1-1,)
            rebalanceTechnique = SMOTETomek(random_state=0, smote = rebalanceTechnique1)
        else:
            rebalanceTechnique = SMOTETomek(random_state=0)
            
        # 对训练数据集作平衡处理 
        over_samples_X,over_samples_y = rebalanceTechnique.fit_sample(X_array_metrics_only, y_labels)
#         # 按照特征，标签顺序合并
#         over_samples_y = over_samples_y.reshape(over_samples_y.shape[0],1)#对一维数组进行转置需要借助reshape来完成
#         array_merge = np.concatenate((over_samples_X,over_samples_y),axis=1);
#         list_colNames = df.columns.tolist();
#         df_result = pd.DataFrame(array_merge, columns = list_colNames);
        
        return over_samples_X,over_samples_y;
    #===end===#
    
    
    