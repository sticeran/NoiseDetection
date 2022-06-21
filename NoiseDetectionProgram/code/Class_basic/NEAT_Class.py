'''
Created on 2020年10月25日

@author: njdx
'''


import pandas as pd
import numpy as np

class NEAT_Class(object):
    
    def __init__(self):
        self.list_mislabels_HR1 = []
        self.list_mislabels_HR2 = []
        self.list_mislabels_NEAT = []
    
    def FUNCTION_HeuristicRules(self,df_current_labels_n_low,df_current_labels_g_low,df_current_labels_n_high):
        self.list_mislabels_HR1 = []
        self.list_mislabels_HR2 = []
        self.list_mislabels_NEAT = []
        #低版本原始标签
#         df_current_labels_n_low = pd.read_csv(file_current_labels_n_low);
        #低版本真实误标签
#         df_current_labels_g_low = pd.read_csv(file_current_labels_g_low);
        #真实干净标签
        index_mis_low = df_current_labels_g_low.index.values;
        df_current_labels_clean_low = df_current_labels_n_low.loc[~df_current_labels_n_low.index.isin(index_mis_low)];
        
        #不保留特征列
        df_current_labels_n_low = df_current_labels_n_low[['instances','bug']];
        df_current_labels_n_high = df_current_labels_n_high[['instances','loc','bug']];
        df_current_labels_n_high['index'] = df_current_labels_n_high.index.values;
        df_current_labels_clean_low = df_current_labels_clean_low[['instances']];
        df_current_labels_g_low = df_current_labels_g_low[['instances']];
        
        #原始标签中，低版本和高版本实例的交集
        df_intersection_allInstances = pd.merge(df_current_labels_n_low, df_current_labels_n_high, how='inner', on='instances');
        #1B:和1A本质上相同。前一版本中不是误标签，下一版本中也不是误标签
        #2A:原始标签中，低版本和高版本实例的交集中，标签相同的实例
        temp_bool = df_intersection_allInstances['bug_x']==df_intersection_allInstances['bug_y']
        df_2A = df_intersection_allInstances.loc[temp_bool]
        #2B:原始标签中，低版本和高版本实例的交集中，标签不同的实例
        temp_bool = df_intersection_allInstances['bug_x']!=df_intersection_allInstances['bug_y']
        df_2B = df_intersection_allInstances.loc[temp_bool]
        #1A+2A(TP+FP)：原始标签中，低版本和高版本实例的交集中，标签相同的实例，且低版本中是误标签
        df_1A2A = pd.merge(df_2A, df_current_labels_g_low, how='inner', on='instances');
        #1B+2B(TP+FP)：原始标签中，低版本和高版本实例的交集中，标签不同的实例，且低版本中不是误标签
        df_1B2B = pd.merge(df_2B, df_current_labels_clean_low, how='inner', on='instances');
        #1A+2A(TP+FP)∪1B+2B(TP+FP)
        df_1A2A1B2B = df_1A2A.append(df_1B2B)
        
        #1A+2A
        df_predictedClean = df_1A2A.loc[df_1A2A['bug_y'] == 1];
        df_predictedClean = df_predictedClean.sort_values(by="loc", ascending=True);
        errors_idx_1_sort_ASC = df_predictedClean['index'].values;
        df_predictedBuggy = df_1A2A.loc[df_1A2A['bug_y'] == 0];
        df_predictedBuggy = df_predictedBuggy.sort_values(by="loc", ascending=False);
        errors_idx_0_sort_DES = df_predictedBuggy['index'].values;
        errors_idx_all_sort = np.concatenate((errors_idx_1_sort_ASC,errors_idx_0_sort_DES), axis=0);
        self.list_mislabels_HR1 = [errors_idx_all_sort,errors_idx_1_sort_ASC,errors_idx_0_sort_DES];
        
        #1B+2B
        df_predictedClean = df_1B2B.loc[df_1B2B['bug_y'] == 1];
        df_predictedClean = df_predictedClean.sort_values(by="loc", ascending=True);
        errors_idx_1_sort_ASC = df_predictedClean['index'].values;
        df_predictedBuggy = df_1B2B.loc[df_1B2B['bug_y'] == 0];
        df_predictedBuggy = df_predictedBuggy.sort_values(by="loc", ascending=False);
        errors_idx_0_sort_DES = df_predictedBuggy['index'].values;
        errors_idx_all_sort = np.concatenate((errors_idx_1_sort_ASC,errors_idx_0_sort_DES), axis=0);
        self.list_mislabels_HR2 = [errors_idx_all_sort,errors_idx_1_sort_ASC,errors_idx_0_sort_DES];
        
        #对识别的误标签按LOC值排序
        df_predictedClean = df_1A2A1B2B.loc[df_1A2A1B2B['bug_y'] == 1];
        df_predictedClean = df_predictedClean.sort_values(by="loc", ascending=True);
        errors_idx_1_sort_ASC = df_predictedClean['index'].values;
        df_predictedBuggy = df_1A2A1B2B.loc[df_1A2A1B2B['bug_y'] == 0];
        df_predictedBuggy = df_predictedBuggy.sort_values(by="loc", ascending=False);
        errors_idx_0_sort_DES = df_predictedBuggy['index'].values;
        errors_idx_all_sort = np.concatenate((errors_idx_1_sort_ASC,errors_idx_0_sort_DES), axis=0);
        #这个效果咋样？效果不好。代码行多的有无Bug尚有争议，但是代码行少的不容易出Bug。假如bug是随机出现的，那么代码行少的包含bug的概率自然比代码行多的要低。
#                 errors_idx_all_sort = np.concatenate((errors_idx_0_sort_DES,errors_idx_1_sort_ASC), axis=0);
        self.list_mislabels_NEAT = [errors_idx_all_sort,errors_idx_1_sort_ASC,errors_idx_0_sort_DES];
    
    
    
    
    
    