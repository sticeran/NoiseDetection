
# coding: utf-8

from __future__ import print_function, absolute_import, division, with_statement
import pandas as pd
import numpy as np
import math
import sys
import os
import random
import Class_basic.EvaluationIndicator as EI
import Class_basic.ConfidentLearningClass as CL
import Class_basic.class_MyKNN as KNN
import Class_basic.fileNameSorting as FNS
import Class_basic.EnsembleLearning_fromProbability as EL
import copy


#===两阶段CNDC方法识别噪音，返回下标===#
def FUNCTION_identifyNoise_CNDC(array_metrics_only,df_conﬁdenceMeasures_baseline,original_labels):
    array_votes = df_conﬁdenceMeasures_baseline["(MV)vote"].values;
    NS_idx_votes = np.where(array_votes > 0)[0];
    NSfree_idx_votes = np.where(array_votes == 0)[0];
    NSfree_original_labels = original_labels[NSfree_idx_votes];
    lebel_errors_bool = (NSfree_original_labels == 0);
    class0_idx_NSfree = NSfree_idx_votes[lebel_errors_bool];
    lebel_errors_bool = (NSfree_original_labels == 1);
    class1_idx_NSfree = NSfree_idx_votes[lebel_errors_bool];
    RealNoise_idx = [];
    for i_NS in NS_idx_votes:
        iRow = array_metrics_only[i_NS];
        iRow = np.expand_dims(iRow, axis=0);
        i_NS_label = original_labels[i_NS];
        array_metrics_otherInstances = array_metrics_only[class0_idx_NSfree];
        len_class0 = len(class0_idx_NSfree);
        if len_class0>10:
            N_nearest = 10;
        else:
            N_nearest = len_class0
        DC_class0 = Class_KNN.FUNCTION_KNN_avgD(iRow, array_metrics_otherInstances, k=N_nearest);#返回当前实例的前N个最近邻的实例的行号
        
        array_metrics_otherInstances = array_metrics_only[class1_idx_NSfree];
        len_class1 = len(class1_idx_NSfree);
        if len_class1>10:
            N_nearest = 10;
        else:
            N_nearest = len_class1
        DC_class1 = Class_KNN.FUNCTION_KNN_avgD(iRow, array_metrics_otherInstances, k=N_nearest);#返回当前实例的前N个最近邻的实例的行号
        
        if (DC_class0 > DC_class1) and (i_NS_label == 0):
            RealNoise_idx.append(i_NS);
        elif (DC_class0 < DC_class1) and (i_NS_label == 1):
            RealNoise_idx.append(i_NS);
    if RealNoise_idx:
        errors_idx_votes = np.array(RealNoise_idx);
    else:
        errors_idx_votes = np.array([]);
    return errors_idx_votes;
#===end===#

#===根据相应的置信列，计算accuracy,precision,recall,F1,AP,RR,P@k===#
def FUNCTION_calculateIndicators_baseline(name_baseline,list_measureNames,df_conﬁdenceMeasures,idx_errors,actual_label_errors,origin_index):
    list_result = [];
    if idx_errors.size != 0:
        idx_errors_current = df_conﬁdenceMeasures.iloc[idx_errors,:].index.values;
        for i_measureName in list_measureNames:
            list_indicators_allSeed = []
            for i_randomSeed in list_randomSeed:
                random.seed(i_randomSeed);
                array_measure = df_conﬁdenceMeasures[i_measureName].values;
                errors_idx_measure = idx_errors_current.copy();
                confident_measure = array_measure[errors_idx_measure];
                if i_measureName == "(MV)vote" or i_measureName == "(CV)vote":
    #                 errors_idx_allGroups = []
    #                 for i_group in range(num_classifiers, (vote_threshold-1), -1):
    #                     idx_temp = np.where(confident_measure == i_group)[0];
    # #                     idx_group = idx_errors[idx_temp]
    # #                     random.shuffle(idx_group)
    # #                     errors_idx_allGroups.extend(idx_group);
    #                     errors_idx_allGroups.extend(idx_temp);
    #                 errors_idx_measure = errors_idx_measure[np.array(errors_idx_allGroups)];
                    random.shuffle(errors_idx_measure)
                elif i_measureName == "std_MV" or i_measureName == "std_CV":
                    errors_idx_measure = errors_idx_measure[np.argsort(-confident_measure)];#按照置信值从大到小排序，返回索引
                else:
                    errors_idx_measure = errors_idx_measure[np.argsort(confident_measure)];#按照置信值从小到大排序，返回索引
            
                list_indicators = Class_Evaluation.calculateIndicators(errors_idx_measure,actual_label_errors,origin_index);
                list_indicators_allSeed.append(list_indicators)
            df_allSeed = pd.DataFrame(list_indicators_allSeed)
            mean = df_allSeed.mean()
            list_mean = mean.values;
            i_measureName_baseline = name_baseline + i_measureName;
            temp_oneRow = [i_measureName_baseline];
            temp_oneRow.extend(list_mean);
            list_result.append(temp_oneRow);
    else:
        for i_measureName in list_measureNames:
            errors_idx_measure = np.array(idx_errors);
            list_indicators = Class_Evaluation.theoreticalWorstValues();
#             list_indicators = Class_Evaluation.calculateIndicators(errors_idx_measure,actual_label_errors,origin_index);
            i_measureName_baseline = name_baseline + i_measureName;
            temp_oneRow = [i_measureName_baseline];
            temp_oneRow.extend(list_indicators);
            list_result.append(temp_oneRow);
    random.seed(specifiedSeed);
    return list_result;
#===end===#

#===对于多个分类器，对于投票数大于一半的按投票数分组，在每一组分别按10种置信度量排序，计算accuracy,precision,recall,F1,AP,RR,P@k===#
def FUNCTION_calculateIndicators_groupRanking(list_measureNames,df_conﬁdenceMeasures,idx_errors,actual_label_errors,origin_index):
    list_result = [];
    if idx_errors.size != 0:
        idx_errors_current = df_conﬁdenceMeasures.iloc[idx_errors,:].index.values;
#         confident_votes = array_votes[errors_idx_votes];
        for i_measureName in list_measureNames:
            if i_measureName == "(NR-OV)vote" or i_measureName == "(NR-OV)E_probability":
                vote_threshold = 1;
            else:
                vote_threshold = math.ceil(num_classifiers/2);
            list_indicators_allSeed = []
            for i_randomSeed in list_randomSeed:
                random.seed(i_randomSeed);
                array_measure = df_conﬁdenceMeasures[i_measureName].values;
                errors_idx_measure = idx_errors_current.copy();
                confident_measure = array_measure[idx_errors];
                
                #---分组排序---#
                if i_measureName == "(NR-OV)vote" or i_measureName == "(NR-MV)vote" or i_measureName == "(NR-CV)vote":
                    errors_idx_allGroups = []
                    for i_group in range(num_classifiers, (vote_threshold-1), -1):
                        idx_temp = np.where(confident_measure == i_group)[0];
                        idx_group = idx_errors[idx_temp]
                        random.shuffle(idx_group)
                        errors_idx_allGroups.extend(idx_group);
                    errors_idx_measure = np.array(errors_idx_allGroups);
                else:
                    array_votes = df_conﬁdenceMeasures[list_measureNames[0]].values;
                    confident_votes = array_votes[idx_errors];
                    errors_idx_allGroups = []
                    for i_group in range(num_classifiers, (vote_threshold-1), -1):
                        idx_temp = np.where(confident_votes == i_group)[0];
                        idx_group = idx_errors[idx_temp]
                        errors_idx_measure_group = idx_group
                        confident_group = confident_measure[idx_temp];
                        errors_idx_measure_group = errors_idx_measure_group[np.argsort(-confident_group)];#按照置信值从大到小排序，返回索引
                        errors_idx_allGroups.extend(errors_idx_measure_group);
                    errors_idx_measure = np.array(errors_idx_allGroups);
                #---end---#
            
                list_indicators = Class_Evaluation.calculateIndicators(errors_idx_measure,actual_label_errors,origin_index);
                list_indicators_allSeed.append(list_indicators)
            df_allSeed = pd.DataFrame(list_indicators_allSeed)
            mean = df_allSeed.mean()
            list_mean = mean.values;
            temp_oneRow = [i_measureName];
            temp_oneRow.extend(list_mean);
            list_result.append(temp_oneRow);
    else:
        for i_measureName in list_measureNames:
            errors_idx_measure = np.array(idx_errors);
            list_indicators = Class_Evaluation.theoreticalWorstValues();
#             list_indicators = Class_Evaluation.calculateIndicators(errors_idx_measure,actual_label_errors,origin_index);
            temp_oneRow = [i_measureName];
            temp_oneRow.extend(list_indicators);
            list_result.append(temp_oneRow);
    random.seed(specifiedSeed);
    return list_result;
#===end===#

#===根据相应的置信列，计算accuracy,precision,recall,F1,AP,RR,P@k===#
def FUNCTION_calculateIndicators(list_measureNames,df_conﬁdenceMeasures,idx_errors,actual_label_errors,origin_index):
    list_result = [];
    if idx_errors.size != 0:
        idx_errors_current = df_conﬁdenceMeasures.iloc[idx_errors,:].index.values;
        for i_measureName in list_measureNames:
            list_indicators_allSeed = []
            for i_randomSeed in list_randomSeed:
                random.seed(i_randomSeed);
                array_measure = df_conﬁdenceMeasures[i_measureName].values;
                errors_idx_measure = idx_errors_current.copy();
                confident_measure = array_measure[idx_errors];
                
                if i_measureName == "(MV)vote" or i_measureName == "(CV)vote":
    #                 errors_idx_allGroups = []
    #                 for i_group in range(num_classifiers, (vote_threshold-1), -1):
    #                     idx_temp = np.where(confident_measure == i_group)[0];
    #     #                 idx_group = idx_errors[idx_temp]
    #     #                 random.shuffle(idx_group)
    #     #                 errors_idx_allGroups.extend(idx_group);
    #                     errors_idx_allGroups.extend(idx_temp); 
    #                 errors_idx_measure = errors_idx_measure[np.array(errors_idx_allGroups)];
                    random.shuffle(errors_idx_measure)
                elif i_measureName == "(NR-MV)vote" or i_measureName == "(NR-CV)vote":
                    errors_idx_allGroups = []
                    for i_group in range(num_classifiers, (vote_threshold-1), -1):
                        idx_temp = np.where(confident_measure == i_group)[0];
                        idx_group = idx_errors[idx_temp]
                        random.shuffle(idx_group)
                        errors_idx_allGroups.extend(idx_group);
    #                     errors_idx_allGroups.extend(idx_temp);
    #                 errors_idx_measure = errors_idx_measure[np.array(errors_idx_allGroups)];
                    errors_idx_measure = np.array(errors_idx_allGroups);
                elif i_measureName == "std_MV" or i_measureName == "std_CV":
                    errors_idx_measure = errors_idx_measure[np.argsort(confident_measure)];#按照置信值从小到大排序，返回索引
                else:
                    errors_idx_measure = errors_idx_measure[np.argsort(-confident_measure)];#按照置信值从大到小排序，返回索引
                
                list_indicators = Class_Evaluation.calculateIndicators(errors_idx_measure,actual_label_errors,origin_index);
                list_indicators_allSeed.append(list_indicators)
            df_allSeed = pd.DataFrame(list_indicators_allSeed)
            mean = df_allSeed.mean()
            list_mean = mean.values;
            temp_oneRow = [i_measureName];
            temp_oneRow.extend(list_mean);
            list_result.append(temp_oneRow);
    else:
        for i_measureName in list_measureNames:
            errors_idx_measure = np.array(idx_errors);
            list_indicators = Class_Evaluation.theoreticalWorstValues();
#             list_indicators = Class_Evaluation.calculateIndicators(errors_idx_measure,actual_label_errors,origin_index);
            temp_oneRow = [i_measureName];
            temp_oneRow.extend(list_indicators);
            list_result.append(temp_oneRow);
    random.seed(specifiedSeed);
    return list_result;
#===end===#

#===majority vote to identify mislabels===#
def FUNCTION_majorityVote(array_probability_allClassfiers,s):
    Class_EL.FUNCTION_initialization(s);
    #获得psx，传给FUNCTION_EnsembleLearning_main
    for i_classifier in range(num_classifiers):#每个分类器包含预测为0的概率和1的概率的两列
        idx_col_currentClassifier = i_classifier * 2;#每次读当前分类器预测为0和1的列
        psx = array_probability_allClassfiers[:,[idx_col_currentClassifier,idx_col_currentClassifier+1]]
        
        #===传统集成学习Ensemble learning，即多分类器投票===#
        Class_EL.FUNCTION_EnsembleLearning_main(psx,i_classifier);
        #===end===#
    Class_EL.FUNCTION_CalculateVotingMeasures();
    return Class_EL.votingMatrix_probability,Class_EL.matrix_votingMeasures;
#===end===#

#===soft vote to identify mislabels===#
def FUNCTION_softVote(array_probability_allClassfiers,s):
#     Class_EL.FUNCTION_initialization(s);
    matrix_votingMeasures = Class_EL.FUNCTION_EnsembleLearning_softVoting(array_probability_allClassfiers,s);
#     Class_EL.FUNCTION_CalculateVotingMeasures();
    return matrix_votingMeasures;
#===end===#

#===置信学习，计算性能指标===#
def FUNCTION_calculateIndicators_confidentLearning(name_baseline,idx_errors,actual_label_errors,origin_index):
    list_result = [];
    idx_errors = np.array(idx_errors);
    if idx_errors.size != 0:
        list_indicators = Class_Evaluation.calculateIndicators(idx_errors,actual_label_errors,origin_index);
        temp_oneRow = [name_baseline];
        temp_oneRow.extend(list_indicators);
        list_result.append(temp_oneRow);
    else:
        list_indicators = Class_Evaluation.theoreticalWorstValues();
        temp_oneRow = [name_baseline];
        temp_oneRow.extend(list_indicators);
        list_result.append(temp_oneRow);
    return list_result;
#===end===#

#===对于单个分类器，计算性能指标===#
def FUNCTION_calculateIndicators_eachClassifier(df_votingMatrix,actual_label_errors,origin_index):
    list_result = [];
    list_classfierNames = list(df_votingMatrix)[1:];#获取分类器列名列表
    for i_classfierName in list_classfierNames:
        errors_idx_single = [];
        array_single = df_votingMatrix[i_classfierName].values;
#         errors_idx_single = np.where(array_single > 0)[0];
#         confident_single = array_single[errors_idx_single];
        idx_single = np.where(array_single > 0)[0];
        errors_idx_single = df_votingMatrix.iloc[idx_single,:].index.values;
        confident_single = array_single[idx_single];
        errors_idx_single = errors_idx_single[np.argsort(-confident_single)];#按照置信值从大到小排序，返回索引
        
        if errors_idx_single.size!=0:
            list_indicators = Class_Evaluation.calculateIndicators(errors_idx_single,actual_label_errors,origin_index);
            temp_oneRow = [i_classfierName];
            temp_oneRow.extend(list_indicators);
            list_result.append(temp_oneRow);
        else:
            list_indicators = Class_Evaluation.theoreticalWorstValues();
            temp_oneRow = [i_classfierName];
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
    
    folder_type_saved = "BugLabel";#不使用特征选择
#     folder_type_saved = "BugLabel_FS";#使用特征选择
    
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
    
    # 使用的分类器名字
    list_names_classifiers = ["RBF SVM", "Random Forest", "Naive Bayes",
                              "K-NN (K=5)", "Neural Net",]#CNDC文献所使用的5个分类器
    
    # 分类器数目
    num_classifiers = len(list_names_classifiers);
    #常量：分类器数大于一半的数量
    vote_threshold = math.ceil(num_classifiers/2);
    
    #集合学习类
    Class_EL = EL.EnsembleLearning(len(list_names_classifiers));
    # CL框架下， 共采用几种噪音识别方法
    strategy_name = "ConfidentJoint"
    #置信学习类
    Class_CL = CL.ConfidentLearning(strategy_name);
    #KNN方法选取最近邻
    Class_KNN = KNN.MyKNN();
    #实例化评价指标类
    Class_Evaluation = EI.EvaluationIndicator();
    
    # 共计算几个置信指标
    columns_voting = ["(OV)vote",
                      "(OV)E_probability",
                      "(MV)vote",
                      "(MV)E_probability",
                      "(CV)vote",
                      "(CV)E_probability"];
    list_measureNames_baseline_MV = ["(MV)vote","(MV)E_probability"];
    list_measureNames_baseline_CV = ["(CV)vote","(CV)E_probability"];
    list_measureNames_baseline_NR_OVMV = ["(NR-OV)vote","(NR-OV)E_probability","(NR-MV)vote","(NR-MV)E_probability"];
    
    # 随机种子列表
    list_specifiedSeed = [1];
    list_randomSeed = list(range(1, 21));
    
    #三种训练集
    list_trainType = ["Continuous"];
    
    # 读取所有分类器概率列名
    list_colNames_read = [];
    for i_classifierName in list_names_classifiers:
        i_classifierName_0 = i_classifierName + "_bug: 0";
        i_classifierName_1 = i_classifierName + "_bug: 1";
        list_colNames_read.extend([i_classifierName_0,i_classifierName_1]);
    # 为置信学习设置分类器
    classifier_CL = "Random Forest";
    # 读取读取特定分类器概率列名概率列名
    i_classifierName_0 = classifier_CL + "_bug: 0";
    i_classifierName_1 = classifier_CL + "_bug: 1";
    list_colNames_read_CL = [i_classifierName_0,i_classifierName_1];
    
    
    for i_trainType in list_trainType:
    
        for specifiedSeed in list_specifiedSeed:
            random.seed(specifiedSeed);
#             np.random.seed(specifiedSeed);
            
            #读取权重文件
            path_common_read =  path_common + i_trainType + "/" + folder_type_saved + "/";
            
            #获取需要计算的文件名列表从有监督模型计算的结果路径获取
            path_common_read_probability = path_common_read+"/(result)probability/" + "/classfiers_MAX/" + dataset_style + "/seed" + str(specifiedSeed) + "/";
            #存储路径
            path_common_saved_mislabel = path_common_read + "/(supervised)baseline/(result)mislabel/" + dataset_style + "/classfiers_" + str(num_classifiers) + "/seed" + str(specifiedSeed) + "/";#单个文件的误标签预测结果的存储路径
            path_saved_common = path_common_read+"/(supervised)baseline/evaluationIndicators/" + dataset_style + "/classfiers_" + str(num_classifiers) + "/seed" + str(specifiedSeed) + "/";
            
            # 获取数据集项目文件夹列表
            folderList_project = os.listdir(path_common_read_probability);
            for i in range(len(folderList_project)):
                i_folder_project = folderList_project[i];
                
                #获取文件列表
                current_folder_project = path_common_read_dataset + i_folder_project + "/";
                fileList = os.listdir(current_folder_project);
                fileList = FNS.sort_insert_filename(fileList);# 按文件名版本排序
                #获取概率文件列表
                current_folder_project = path_common_read_probability + i_folder_project + "/";
                fileList_probability = os.listdir(current_folder_project);
                fileList_probability = FNS.sort_insert_filename(fileList_probability);# 按文件名版本排序
                
                for i_num in range(len(fileList_probability)):
                    i_file_probability = fileList_probability[i_num];
                    # 获得对应场景的测试集的原始文件名
                    if i_trainType == "Noise":
                        i_file = fileList[i_num];
                    elif i_trainType == "Continuous" or i_trainType == "Single":
                        start_position = i_file_probability.rfind(",")+1;
                        end_position = i_file_probability.rfind(")");
                        version_test = i_file_probability[start_position:end_position];
                        i_file = i_folder_project + '-' + version_test + '.csv';
                    print(i_file_probability)
                    
#                     i_file_weights_classfiers = array_weights[i_num];#第i个文件上所有分类器权重
#                     i_file_weights_classfiers = []
                    
                    #---读取噪音数据集原始标签---#
                    file_dataset = path_common_read_dataset + i_folder_project + "/" + i_file;
                    df_dataset = pd.read_csv(file_dataset);
                    df_dataset_labels = df_dataset;
                    if dataset_style == "IND-JLMIV+R-2020" or dataset_style == "6M-SZZ-2020":
                        df_file_metrics_only = df_dataset.drop(['relName','bug'],axis=1);#删除不需要的列
                    # 对数据集进行log(x+1)变换
#                     FUNCTION_doLogChange(df_file_metrics_only);CNDC方法不需要进行log变换
                    array_metrics_only = df_file_metrics_only.values;#转成numpy数组计算快
#                     df_dataset.loc[df_dataset[df_dataset['bug']!=0].index,['bug']] = 1;#bug标签转成0，1二元标签
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
                    
                    #---读取计算的概率---#
                    file_dataset = path_common_read_probability + i_folder_project + "/" + i_file_probability;
                    df_dataset = pd.read_csv(file_dataset);
                    array_probability_allClassfiers = df_dataset[list_colNames_read].values;
                    array_probability_classfier_CL = df_dataset[list_colNames_read_CL].values;#读取一个分类器的预测概率
                    
                    #硬投票：根据所有分类器的预测概率，生成误标签的投票矩阵
                    matrix_probability, matrix_votingMeasures = FUNCTION_majorityVote(array_probability_allClassfiers,original_labels);
                    # 单个分类器的预测概率
                    df_result_probability = pd.DataFrame(matrix_probability,columns = list_names_classifiers);#共10列
                    df_result_probability["instances"] = df_dataset['instances'].values;
                    temp_col = ["instances"];
                    temp_col.extend(list_names_classifiers);     
                    df_result_probability = df_result_probability.reindex(columns=temp_col);
                    # 硬投票矩阵
                    df_result_votingMeasures_baseline = pd.DataFrame(matrix_votingMeasures,columns = columns_voting);
                    df_result_votingMeasures_baseline["instances"] = df_dataset['instances'].values;
                    temp_col = ["instances"];
                    temp_col.extend(columns_voting);
                    df_result_votingMeasures_baseline = df_result_votingMeasures_baseline.reindex(columns=temp_col);
                    #为NR方法，只保留取"(OV)vote","(MV)vote"两列
                    df_result_votingMeasures_baseline_NR = df_result_votingMeasures_baseline[["(OV)vote","(OV)E_probability",
                                                                                              "(MV)vote","(MV)E_probability",]];
                    df_result_votingMeasures_baseline_NR = df_result_votingMeasures_baseline_NR.rename(columns={"(OV)vote":"(NR-OV)vote","(OV)E_probability":"(NR-OV)E_probability",
                                                                                                                "(MV)vote":"(NR-MV)vote","(MV)E_probability":"(NR-MV)E_probability",})
                    #---end---#
                    
                    #baseline RF
                    array_probability = df_result_probability[classifier_CL].values;
                    errors_idx_all_RF = np.where(array_probability > 0)[0];#因为在之前的步骤中,仅对被投票的(被预测为噪音的)实例 存储了预测概率
                    
                    #baseline Confident Learning
                    errors_idx_all,array_lateralMargin = Class_CL.FUNCTION_confidentLearning_fromPredictionProbability(array_probability_classfier_CL, original_labels);
                    errors_idx_all = errors_idx_all[np.argsort(array_lateralMargin)];#置信值是负的，按照置信值从小到大排序，也即置信值是正的时，按照置信值从大到小排序
                    labels_errors = original_labels[errors_idx_all]
                    idx_temp = np.where(labels_errors == 1)[0];
                    errors_idx_1 = errors_idx_all[idx_temp];
                    idx_temp = np.where(labels_errors == 0)[0];
                    errors_idx_0 = errors_idx_all[idx_temp];
                    list_mislabels_CL = [errors_idx_all,errors_idx_1,errors_idx_0];
                    errors_idx_all_CL = errors_idx_all
                    
                    #baseline MV
                    array_votes = df_result_votingMeasures_baseline["(MV)vote"].values;
                    errors_idx_all = np.where(array_votes > 0)[0];#另一种写法，array_votes > classifier/2,小于classifier/2的也存入结果
                    labels_errors = original_labels[errors_idx_all]
                    idx_temp = np.where(labels_errors == 1)[0];
                    errors_idx_1 = errors_idx_all[idx_temp];
                    idx_temp = np.where(labels_errors == 0)[0];
                    errors_idx_0 = errors_idx_all[idx_temp];
                    list_mislabels_MV = [errors_idx_all,errors_idx_1,errors_idx_0];
                    
                    #baseline CV
                    array_votes = df_result_votingMeasures_baseline["(CV)vote"].values;
                    errors_idx_all = np.where(array_votes > 0)[0];#另一种写法，array_votes == classifier,不等于classifier的也存入结果
                    labels_errors = original_labels[errors_idx_all]
                    idx_temp = np.where(labels_errors == 1)[0];
                    errors_idx_1 = errors_idx_all[idx_temp];
                    idx_temp = np.where(labels_errors == 0)[0];
                    errors_idx_0 = errors_idx_all[idx_temp];        
                    list_mislabels_CV = [errors_idx_all,errors_idx_1,errors_idx_0];
                    
                    #baseline NR
                    array_votes = df_result_votingMeasures_baseline["(OV)vote"].values;
                    errors_idx_all = np.where(array_votes > 0)[0];#另一种写法，array_votes > classifier/2,小于classifier/2的也存入结果
                    labels_errors = original_labels[errors_idx_all]
                    idx_temp = np.where(labels_errors == 1)[0];
                    errors_idx_1 = errors_idx_all[idx_temp];
                    idx_temp = np.where(labels_errors == 0)[0];
                    errors_idx_0 = errors_idx_all[idx_temp];
                    list_mislabels_NR = [errors_idx_all,errors_idx_1,errors_idx_0];
                    
                    #baseline CNDC
                    errors_idx_all = FUNCTION_identifyNoise_CNDC(array_metrics_only,df_result_votingMeasures_baseline,original_labels);
                    if errors_idx_all.size != 0:
                        labels_errors = original_labels[errors_idx_all]
                        idx_temp = np.where(labels_errors == 1)[0];
                        errors_idx_1 = errors_idx_all[idx_temp];
                        idx_temp = np.where(labels_errors == 0)[0];
                        errors_idx_0 = errors_idx_all[idx_temp];        
                        list_mislabels_CNDC = [errors_idx_all,errors_idx_1,errors_idx_0];
                    else:
                        list_mislabels_CNDC = [np.array([]),np.array([]),np.array([])];
                    
                    #===存识别的误标签结果===#
                    #原始标签，留着观察用
                    df_dataset_labels = df_dataset_labels[['relName','bug']];
                    #真实误标签的结果
                    df_mislabels_groundTruth['realMislabel'] = 1;
                    df_mislabels_groundTruth = df_mislabels_groundTruth[['relName','realMislabel']];
                    df_result_votingMeasures_baseline = df_result_votingMeasures_baseline.rename(columns = {'instances':'relName'});
#                     df_result_softVoting_baseline = df_result_softVoting_baseline.rename(columns = {'instances':'relName'});
                    
                    saved_folder = "RF/";
                    #获取实例名和原始标签
                    df_result_mislabel = df_dataset_labels;
                    #记录预测误标签的结果
                    df_result_mislabel['predictedMislabel'] = 0;
                    df_result_mislabel.loc[errors_idx_all_RF,['predictedMislabel']] = 1;
                    #记录真实误标签的结果
                    df_result_mislabel = pd.merge(df_result_mislabel, df_mislabels_groundTruth, how='left', on='relName');
                    df_result_mislabel['realMislabel'].fillna(0, inplace=True);
                      
                    dir_path_saved = path_common_saved_mislabel + "/" + saved_folder + "/";
                    if not os.path.exists(dir_path_saved):
                        os.makedirs(dir_path_saved);
                    #度量和标签的交集的存储文件名
                    path_saved_fileName = dir_path_saved + i_file_probability;
                    df_result_mislabel.to_csv(path_saved_fileName,index=False);#不保存行索引
                    
                    saved_folder = "CL/";
                    #获取实例名和原始标签
                    df_result_mislabel = df_dataset_labels;
                    #记录预测误标签的结果
                    df_result_mislabel['predictedMislabel'] = 0;
                    df_result_mislabel.loc[errors_idx_all_CL,['predictedMislabel']] = 1;
                    #记录真实误标签的结果
                    df_result_mislabel = pd.merge(df_result_mislabel, df_mislabels_groundTruth, how='left', on='relName');
                    df_result_mislabel['realMislabel'].fillna(0, inplace=True);
                     
                    dir_path_saved = path_common_saved_mislabel + "/" + saved_folder + "/";
                    if not os.path.exists(dir_path_saved):
                        os.makedirs(dir_path_saved);
                    #度量和标签的交集的存储文件名
                    path_saved_fileName = dir_path_saved + i_file_probability;
                    df_result_mislabel.to_csv(path_saved_fileName,index=False);#不保存行索引
                    
                    saved_folder = "MV/";
                    #记录多数投票的票数（误标签预测结果），平均概率
                    df_result_mislabel = pd.merge(df_dataset_labels, df_result_votingMeasures_baseline[['relName',"(MV)vote","(MV)E_probability",]], how='left', on='relName');
                    #记录预测误标签的结果
                    df_result_mislabel['predictedMislabel'] = 0;
                    df_result_mislabel.loc[df_result_mislabel[df_result_mislabel['(MV)vote']!=0].index,['predictedMislabel']] = 1;#另一种写法，['(MV)vote'] > classifier/2,
                    #记录真实误标签的结果
                    df_result_mislabel = pd.merge(df_result_mislabel, df_mislabels_groundTruth, how='left', on='relName');
                    df_result_mislabel['realMislabel'].fillna(0, inplace=True);
                      
                    dir_path_saved = path_common_saved_mislabel + "/" + saved_folder + "/";
                    if not os.path.exists(dir_path_saved):
                        os.makedirs(dir_path_saved);
                    #度量和标签的交集的存储文件名
                    path_saved_fileName = dir_path_saved + i_file_probability;
                    df_result_mislabel.to_csv(path_saved_fileName,index=False);#不保存行索引
                    
                    saved_folder = "CV/";
                    #记录多数投票的票数（误标签预测结果），平均概率
                    df_result_mislabel = pd.merge(df_dataset_labels, df_result_votingMeasures_baseline[['relName',"(CV)vote","(CV)E_probability",]], how='left', on='relName');
                    #记录预测误标签的结果
                    df_result_mislabel['predictedMislabel'] = 0;
                    df_result_mislabel.loc[df_result_mislabel[df_result_mislabel['(CV)vote']!=0].index,['predictedMislabel']] = 1;#另一种写法，['(CV)vote'] == classifier,
                    #记录真实误标签的结果
                    df_result_mislabel = pd.merge(df_result_mislabel, df_mislabels_groundTruth, how='left', on='relName');
                    df_result_mislabel['realMislabel'].fillna(0, inplace=True);
                      
                    dir_path_saved = path_common_saved_mislabel + "/" + saved_folder + "/";
                    if not os.path.exists(dir_path_saved):
                        os.makedirs(dir_path_saved);
                    #度量和标签的交集的存储文件名
                    path_saved_fileName = dir_path_saved + i_file_probability;
                    df_result_mislabel.to_csv(path_saved_fileName,index=False);#不保存行索引
                      
                    saved_folder = "CNDC/";
                    #记录多数投票的票数（误标签预测结果），平均概率
                    df_result_mislabel = pd.merge(df_dataset_labels, df_result_votingMeasures_baseline[['relName',"(MV)vote","(MV)E_probability",]], how='left', on='relName');
                    #记录预测误标签的结果
                    df_result_mislabel['predictedMislabel'] = 0;
                    df_result_mislabel.loc[list_mislabels_CNDC[0],['predictedMislabel']] = 1;
                    #记录真实误标签的结果
                    df_result_mislabel = pd.merge(df_result_mislabel, df_mislabels_groundTruth, how='left', on='relName');
                    df_result_mislabel['realMislabel'].fillna(0, inplace=True);
                    
                    dir_path_saved = path_common_saved_mislabel + "/" + saved_folder + "/";
                    if not os.path.exists(dir_path_saved):
                        os.makedirs(dir_path_saved);
                    #度量和标签的交集的存储文件名
                    path_saved_fileName = dir_path_saved + i_file_probability;
                    df_result_mislabel.to_csv(path_saved_fileName,index=False);#不保存行索引
#                     #===end===#
                    
                    #分别计算三种类型的误标签识别率
                    for i_type_mislabels in range(len(list_folderSaved)):
    #                     i_type_mislabels = 2;#调试用，1："(buggy)mislabels"
                        #存所有baseline结果
                        list_indicators_allBaselines = [];
                        
                        mislabels_currentType = list_folderSaved[i_type_mislabels];
                        originIndex_currentType = list_orginIndex[i_type_mislabels];#标签是1，0，1和0，的行的下标
                        index_realMislabels_currentType = list_index_realMislabels[i_type_mislabels];
                        
                        #需要过滤buggy标签/clean标签无噪音标签的，否则画柱状图时，会拉低识别buggy标签/clean标签中的噪音标签的性能
                        if index_realMislabels_currentType.size!=0:
                            #需要原始数据集中的代码行
                            df_original_type = df_original.loc[originIndex_currentType,:]
                            Class_Evaluation.setDataset(df_original_type);
                            
                            #对于单个分类器，计算性能指标
                            df_result_probability_current = df_result_probability.iloc[originIndex_currentType,:]
                            list_result = FUNCTION_calculateIndicators_eachClassifier(df_result_probability_current,index_realMislabels_currentType,originIndex_currentType);
                            list_indicators_allBaselines.extend(list_result);
                            
                            #baseline Confident Learning
                            errors_idx = list_mislabels_CL[i_type_mislabels];
                            list_result = FUNCTION_calculateIndicators_confidentLearning("ConfidentLearning",errors_idx,index_realMislabels_currentType,originIndex_currentType)
                            list_indicators_allBaselines.extend(list_result);
                            
                            #baseline MV
                            errors_idx = list_mislabels_MV[i_type_mislabels];
                            list_result = FUNCTION_calculateIndicators(list_measureNames_baseline_MV,df_result_votingMeasures_baseline,errors_idx,index_realMislabels_currentType,originIndex_currentType);
                            list_indicators_allBaselines.extend(list_result);
                            
                            #baseline CV
                            errors_idx = list_mislabels_CV[i_type_mislabels];
                            list_result = FUNCTION_calculateIndicators(list_measureNames_baseline_CV,df_result_votingMeasures_baseline,errors_idx,index_realMislabels_currentType,originIndex_currentType);
                            list_indicators_allBaselines.extend(list_result);
                            
                            #baseline NR-OV, NR-MV
                            errors_idx = list_mislabels_NR[i_type_mislabels];#需要加几个排序指标
                            list_result = FUNCTION_calculateIndicators_groupRanking(list_measureNames_baseline_NR_OVMV,df_result_votingMeasures_baseline_NR,errors_idx,index_realMislabels_currentType,originIndex_currentType);
                            list_indicators_allBaselines.extend(list_result);
                            
                            #baseline CNDC
                            errors_idx = list_mislabels_CNDC[i_type_mislabels];
                            list_result = FUNCTION_calculateIndicators_baseline("(CNDC)",list_measureNames_baseline_MV,df_result_votingMeasures_baseline,errors_idx,index_realMislabels_currentType,originIndex_currentType);
                            list_indicators_allBaselines.extend(list_result);
                        
                            #存储单分类器和多分类器的投票的评价指标计算结果
                            dir_path_saved = path_saved_common + mislabels_currentType + "/";
                            if not os.path.exists(dir_path_saved):
                                os.makedirs(dir_path_saved);
                            path_saved_fileName = dir_path_saved + i_file_probability;
                            df_allMethods = pd.DataFrame(list_indicators_allBaselines, columns=columns_indicators);
                            df_allMethods.to_csv(path_saved_fileName,index=False);#不保存行索引
                            
                            print("%s %s finish"%(i_folder_project,i_file_probability));
    
    
    
    
    
    sys.exit()
