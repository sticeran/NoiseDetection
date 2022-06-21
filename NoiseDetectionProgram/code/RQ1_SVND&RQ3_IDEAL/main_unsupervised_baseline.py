
# coding: utf-8

from __future__ import print_function, absolute_import, division, with_statement
import pandas as pd
import numpy as np
# import math
import sys
import os
import random
import Class_basic.EvaluationIndicator as EI
import Class_basic.class_MyKNN as KNN
import Class_basic.class_MyCLNI as CLNI
import Class_basic.fileNameSorting as FNS
import itertools
import copy


#===根据相应的置信列，计算accuracy,precision,recall,F1,AP,RR,P@k===#
def FUNCTION_calculateIndicators_noScores(name_baseline,errors_idx,actual_label_errors,origin_index):
    errors_idx = np.array(errors_idx)
    actual_label_errors = np.array(actual_label_errors)
    
    list_result = [];
    if errors_idx.size!=0:
        list_indicators_allSeed = []
        #由于不区分顺序，所以应改成随机20次
        for i_randomSeed in list_randomSeed:
            random.seed(i_randomSeed);
            errors_idx_measure = errors_idx.copy();
            random.shuffle(errors_idx_measure)
            list_indicators = Class_Evaluation.calculateIndicators(errors_idx_measure,actual_label_errors,origin_index);
            list_indicators_allSeed.append(list_indicators)
        df_allSeed = pd.DataFrame(list_indicators_allSeed)
        mean = df_allSeed.mean()
        list_mean = mean.values;
        temp_oneRow = [name_baseline];
        temp_oneRow.extend(list_mean);
        list_result.append(temp_oneRow);
    else:
        list_indicators =  Class_Evaluation.theoreticalWorstValues();
        temp_oneRow = [name_baseline];
        temp_oneRow.extend(list_indicators);
        list_result.append(temp_oneRow);
    return list_result;
#===end===#

#===log(x+1)变换===#
def FUNCTION_doLogChange(df):
    columns = df.columns.tolist()
    for c in columns:
        # 对c列做log(x + 1)
        df[c] = df[c].apply(np.log1p) # np.log1p与np.expm1互为逆运算
#===end===#


if __name__=='__main__':
    #数据集名字
#     dataset_style = "IND-JLMIV+R-2020";
    dataset_style = "6M-SZZ-2020";
    
    #是否已过滤不一致标签
    read_type_IL = "filteredIL";
#     read_type_IL = "unfilteredIL";
    
    #数据集读取路径
    path_common = "D:/DataSets/"+read_type_IL+"/";#调整成你的数据集存储路径
    path_common_read_dataset = path_common + "(selected)dataSets/" + dataset_style + '/';
    #真实误标签读取路径
    path_common_mislabels_groundTruth = path_common + "(groundTruth)mislabels/" + dataset_style + "/";
    
    #存储类型文件夹名
    list_folderSaved = ["(all)mislabels","(buggy)mislabels","(non-buggy)mislabels"];
    #存储列名
    columns_indicators = ["name", "accuracy", "precision", "recall", "F1", "FAR", "D2H", 
                          "IFA", 'Inspect', 'LOC_Inspect', "NLPTLOC", "AP",
                          "P@1", "P@5", "P@10",];
    
    #实例化评价指标类
    Class_Evaluation = EI.EvaluationIndicator();
    
    # 随机种子列表
    list_specifiedSeed = [1];
    list_randomSeed = list(range(1, 21));
    
    #KNN方法选取最近邻
    Class_KNN = KNN.MyKNN();
    #CLNI方法选取误标签
    Class_CLNI = CLNI.MyCLNI();
    
    for specifiedSeed in list_specifiedSeed:
#         np.random.seed(specifiedSeed)
        random.seed(specifiedSeed)
        #获取需要计算的文件名列表从有监督模型计算的结果路径获取
        #存储路径
        path_common_saved_mislabel = path_common + "/(unsupervised)baseline/(result)mislabel/" + dataset_style + "/";#单个文件的误标签预测结果的存储路径
        path_saved_common = path_common + "/(unsupervised)baseline/evaluationIndicators/" + dataset_style + "/";
        
        #项目列表
        list_projectName = os.listdir(path_common_read_dataset);
        for i in range(len(list_projectName)):
            i_folder_project = list_projectName[i];
            
            current_folder_project = path_common_read_dataset + i_folder_project + "/";
            fileList = os.listdir(current_folder_project);
            fileList = FNS.sort_insert_filename(fileList);# 按文件名版本排序
            
            for i_file in fileList:
                print(i_file)
                #---读取噪音数据集原始标签---#
                file_dataset = path_common_read_dataset + i_folder_project + "/" + i_file;
                df_dataset = pd.read_csv(file_dataset);
                if dataset_style == "IND-JLMIV+R-2020" or dataset_style == "6M-SZZ-2020":
                    df_file_metrics_only = df_dataset.drop(['relName','bug'],axis=1);#删除不需要的列
                # 对数据集进行log(x+1)变换 
#                 FUNCTION_doLogChange(df_file_metrics_only);
                array_metrics_only = df_file_metrics_only.values;#转成numpy数组计算快
                df_original = copy.copy(df_dataset)#需要原始数据集中的代码行
                original_labels = df_dataset['bug'].values;
                origin_index_all = df_dataset['relName'].index.values;#ndarray
                origin_index_1 = df_dataset[df_dataset['bug']==1].index.values;#ndarray
                origin_index_0 = df_dataset[df_dataset['bug']==0].index.values;#ndarray
                #---end---#
                
                #---ground truth: 读取误标签的ground truth---#
                file_mislabels_groundTruth = path_common_mislabels_groundTruth + i_folder_project + "/" + i_file;
                df_mislabels_groundTruth = pd.read_csv(file_mislabels_groundTruth);
                actual_label_errors_all = df_mislabels_groundTruth['index'].values;#ndarray
                actual_label_errors_1 = df_mislabels_groundTruth[df_mislabels_groundTruth['bug']==1]['index'].values;#ndarray
                actual_label_errors_0 = df_mislabels_groundTruth[df_mislabels_groundTruth['bug']==0]['index'].values;#ndarray
                #---end---#
                
                list_orginIndex = [origin_index_all,origin_index_1,origin_index_0];
                list_index_realMislabels = [actual_label_errors_all,actual_label_errors_1,actual_label_errors_0];
                
                #baseline: IDI(inconsistent and duplicate instances based on metrics)
                set_errors_idx = set([]);
                for pair in itertools.combinations(range(len(array_metrics_only)),2):
                    if np.array_equal(array_metrics_only[pair[0]],array_metrics_only[pair[1]]): #compare rows
                        set_errors_idx.add(pair[0])
                        set_errors_idx.add(pair[1])
                if len(set_errors_idx)!=0:
                    errors_idx_all = np.array(list(set_errors_idx))
                    labels_errors = original_labels[errors_idx_all]
                    idx_temp = np.where(labels_errors == 1)[0];
                    errors_idx_1 = errors_idx_all[idx_temp];
                    idx_temp = np.where(labels_errors == 0)[0];
                    errors_idx_0 = errors_idx_all[idx_temp];
                    list_mislabels_IDI = [errors_idx_all,errors_idx_1,errors_idx_0];
                else:
                    list_mislabels_IDI = [np.array([]),np.array([]),np.array([])];
                
                #baseline: CLNI算法
                K = len(np.unique(original_labels))
                if K!=1:
                    errors_idx_all = Class_CLNI.FUNCTION_CLNI(array_metrics_only,original_labels);#origin_index_all是noise_labels
                    labels_errors = original_labels[errors_idx_all];
                    idx_temp = np.where(labels_errors == 1)[0];
                    errors_idx_1 = errors_idx_all[idx_temp];
                    idx_temp = np.where(labels_errors == 0)[0];
                    errors_idx_0 = errors_idx_all[idx_temp];
                else:
                    errors_idx_all = [];
                    errors_idx_1 = [];
                    errors_idx_0 = [];
                list_mislabels_CLNI = [errors_idx_all,errors_idx_1,errors_idx_0];
                
                #分别计算三种类型的误标签识别率
                for i_type_mislabels in range(len(list_folderSaved)):
                    i_type_mislabels = 1;
                    #存所有baseline结果
                    list_indicators_allBaselines = [];
                    
                    mislabels_currentType = list_folderSaved[i_type_mislabels];
                    originIndex_currentType = list_orginIndex[i_type_mislabels];
                    index_realMislabels_currentType = list_index_realMislabels[i_type_mislabels];
                    
                    #需要过滤buggy标签/clean标签无噪音标签的，否则画柱状图时，会拉低识别buggy标签/clean标签中的噪音标签的性能
                    if index_realMislabels_currentType.size!=0:
                        #需要原始数据集中的代码行
                        df_original_type = df_original.loc[originIndex_currentType,:]
                        Class_Evaluation.setDataset(df_original_type);
                        
                        #baseline: IDI算法
                        errors_idx = list_mislabels_IDI[i_type_mislabels];
                        list_result = FUNCTION_calculateIndicators_noScores("IDI",errors_idx,index_realMislabels_currentType,originIndex_currentType);
                        list_indicators_allBaselines.extend(list_result);
                        
                        #baseline: CLNI算法
                        errors_idx = list_mislabels_CLNI[i_type_mislabels];
                        list_result = FUNCTION_calculateIndicators_noScores("CLNI",errors_idx,index_realMislabels_currentType,originIndex_currentType);
                        list_indicators_allBaselines.extend(list_result);
                        
                        #存储所有无监督模型的评价指标计算结果
                        dir_path_saved = path_saved_common + mislabels_currentType + "/";
                        if not os.path.exists(dir_path_saved):
                            os.makedirs(dir_path_saved);
                        path_saved_fileName = dir_path_saved + i_file;
                        df_allMethods = pd.DataFrame(list_indicators_allBaselines, columns=columns_indicators);
                        df_allMethods.to_csv(path_saved_fileName,index=False);#不保存行索引
                        
                        
                        print("%s %s finish"%(i_folder_project,i_file));
    
    
    
    
    sys.exit()
