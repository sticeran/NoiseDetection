
# coding: utf-8

from __future__ import print_function, absolute_import, division, with_statement
import pandas as pd
import numpy as np
import math
import sys
import os
import time
import random
# import copy
import Class_basic.PredictionModelClass as PMC
import Class_basic.RebalancingClass as RTC
import Class_basic.NEAT_Class as NC
import TOOLS.fileNameSorting as FNS
import Class_basic.EnsembleLearning_fromProbability as EL
import Class_basic.EvaluationIndicator as EI
import Class_basic.ConfidentLearningClass as CL
import Class_basic.class_MyKNN as KNN
from numpy import float64
# import rpy2.robjects as robjects#更改环境变量后，需要重启一下eclipse
# from rpy2.robjects import pandas2ri


#===log(x+1)变换===#
def FUNCTION_doLogChange(df):
    set_temp = set(df.columns.tolist()) - set(['bug'])
    columns = list(set_temp)
    for c in columns:
        # 对c列做log(x + 1)
        df[c] = df[c].apply(np.log1p) # np.log1p与np.expm1互为逆运算
#===end===#

#===训练集和测试集预处理===#
def FUNCTION_Preprocessing(df_train,df_test,df_test_groundTruth):
    #bug标签转成0，1二元标签
    df_train = df_train.drop(['instances','realMislabel'],axis=1);#删除不需要的列
    df_test = df_test.drop(['instances'],axis=1);#删除不需要的列
    df_test_groundTruth = df_test_groundTruth.drop(['instances'],axis=1);#删除不需要的列
    
    # 对数据集进行log(x+1)变换
    FUNCTION_doLogChange(df_train);
    FUNCTION_doLogChange(df_test);
    FUNCTION_doLogChange(df_test_groundTruth);
    df_train = df_train.fillna(value=0)
    df_test = df_test.fillna(value=0)
    df_test_groundTruth = df_test_groundTruth.fillna(value=0)
    
    # SMOTETomek再平衡技术。对于使用噪音训练集训练时无法用smote的就用简单的过采样。
    df_training_rebalance = Class_Rebalancing.FUNCTION_Rebalancing(df_train,CategoryLabelName)
    
    # 是否使用特征选择方法
    if folder_type_saved == "BugLabel":
        df_training_fs = df_training_rebalance
    else:
        pass;
        # GainRatio特征选择方法
#                         df_training_fs = Class_FS.GainRatio_from_R(df_training_rebalance, col_Label)
    features_label = list(df_training_fs)
    df_test_fs = df_test[features_label]#测试集根据特征选择后的训练集保留列
    df_test_groundTruth_fs = df_test_groundTruth[features_label]#测试集根据特征选择后的训练集保留列
    return df_training_fs,df_test_fs,df_test_groundTruth_fs
#===end===#

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

#===根据相应的置信列，计算性能指标===#
def FUNCTION_calculateIndicators_NEAT(name_baseline,errors_idx,actual_label_errors,origin_index):
    errors_idx = np.array(errors_idx)
    actual_label_errors = np.array(actual_label_errors)
    
    list_result = [];
    if errors_idx.size!=0:
        #LOC的顺序已排
        list_indicators = Class_Evaluation.calculateIndicators(errors_idx,actual_label_errors,origin_index);
#         TP = Class_Evaluation.getTP();
        temp_oneRow = [name_baseline];
        temp_oneRow.extend(list_indicators);
        list_result.append(temp_oneRow);
    else:
        list_indicators =  Class_Evaluation.theoreticalWorstValues();
#         TP = 0;
        temp_oneRow = [name_baseline];
        temp_oneRow.extend(list_indicators);
        list_result.append(temp_oneRow);
    return list_result;
#===end===#

# #===训练集和测试集预处理===#
# def FUNCTION_CalculatePredictionResultALLMethod():
#     
#     return
# #===end===#

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为200，默认为50
pd.set_option('max_colwidth',200)



#SCVND-ALL：实际的情况。第一对儿训练-测试对儿，训练集使用噪音版本，后续使用部分干净的前一个版本数据集作训练集

if __name__=='__main__':
    #数据集名字
    dataset_style_noise = "6M-SZZ-2020";
    dataset_style_groundTruth = "IND-JLMIV+R-2020"#groundTruth是clean
    
    #是否已过滤不一致标签
    read_type_IL = "filteredIL";
#     read_type_IL = "unfilteredIL";
    
    folder_type_saved = "BugLabel"#不使用特征选择
    #     list_folder_type_saved = ["BugLabel_FS"]#使用特征选择
    
    #噪音检测场景
    scenario_type = "SCVND-ALL"
    
    #读取路径
    path_common = "D:/DataSets/"+read_type_IL+"/";#调整成你的数据集存储路径
    path_common_read_dataset = path_common+"(selected)dataSets/";
    #真实误标签读取路径
    path_common_mislabels_groundTruth = path_common + "(groundTruth)mislabels/" + dataset_style_noise + "/";
    
    #存储列名
    columns_indicators = ["name", "accuracy", "precision", "recall", "F1", "FAR", "D2H", 
                          "IFA", 'Inspect', 'LOC_Inspect', "NLPTLOC", "AP",
                          "P@1", "P@5", "P@10",];
                          
    #项目列表
    folder_read = path_common_read_dataset + dataset_style_noise + "/";
    list_projectName = os.listdir(folder_read);
    
    # 使用的分类器名字
    list_names_classifiers = ["RBF SVM", "Random Forest", "Naive Bayes",
                              "K-NN (K=5)", "Neural Net",]#CNDC文献所使用的5个分类器
    
    # 分类器数目
    num_classifiers = len(list_names_classifiers);
    #常量：分类器数大于一半的数量
    vote_threshold = math.ceil(num_classifiers/2);
    
    # 创建模型预测类
    Class_predictionModel = PMC.PredictionModelClass(list_names_classifiers);
    # 创建数据再平衡类
    Class_Rebalancing = RTC.RebalancingClass();
    #集合学习类
    Class_EL = EL.EnsembleLearning(len(list_names_classifiers));
    # CL框架下， 共采用几种噪音识别方法
    strategy_name = "ConfidentJoint"
    #置信学习类
    Class_CL = CL.ConfidentLearning(strategy_name);
    #KNN方法选取最近邻
    Class_KNN = KNN.MyKNN();
    #NEAT方法类
    Class_NEAT = NC.NEAT_Class();
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
    list_randomSeed = list(range(1, 2));
    
    # 生成存储列名
    list_colNames = [];
    for i_classifierName in list_names_classifiers:
        i_classifierName_0 = i_classifierName + "_bug: 0";
        i_classifierName_1 = i_classifierName + "_bug: 1";
        list_colNames.extend([i_classifierName_0,i_classifierName_1]);
    # 为置信学习设置分类器
    classifier_CL = "Random Forest";
    # 读取读取特定分类器概率列名概率列名
    i_classifierName_0 = classifier_CL + "_bug: 0";
    i_classifierName_1 = classifier_CL + "_bug: 1";
    list_colNames_read_CL = [i_classifierName_0,i_classifierName_1];
    
    #存储类型文件夹名
    list_folderSaved = ["(all)mislabels","(buggy)mislabels","(non-buggy)mislabels"];
    
    
    saved_methodFolder = "CNDC";
    #存储路径
    path_common_saved_mislabel =  path_common + scenario_type + "/" + folder_type_saved + "/(supervised)baseline/"+saved_methodFolder+"/(result)mislabel/";#单个文件的误标签预测结果的存储路径
    path_saved_common = path_common + scenario_type + "/" + folder_type_saved + "/(supervised)baseline/"+saved_methodFolder+"/evaluationIndicators/";
    
    
    
    
    for specifiedSeed in list_specifiedSeed:
        # 设置随机种子
        np.random.seed(specifiedSeed);#不同随机种子生成的十折不一样
        
        #获取需要计算的文件名列表从有监督模型计算的结果路径获取
        path_common_read_probability = path_common + "Noise/" + folder_type_saved + "/(result)probability/" + "/classfiers_MAX/" + dataset_style_noise + "/seed" + str(specifiedSeed) + "/";
        
        #===开始===#
        for i_projectName in list_projectName:
#             i_projectName='commons-collections';
            # 获取数据集的文件列表
            folderName_dataset_noise = path_common_read_dataset + dataset_style_noise + '/' + i_projectName + '/';
            folderName_dataset_clean_groundTruth = path_common_read_dataset + dataset_style_groundTruth + '/' + i_projectName + '/';#作为ground truth的干净数据集
            fileList = os.listdir(folderName_dataset_noise);
            fileList = FNS.sort_insert_filename(fileList);# 按文件名版本排序
            
            #由于每个文件都太小，因此把文件合起来，变成一个大文件。不行，太浪费时间，也不合理。
            #由于单个文件高度不平衡，不利于模型的训练，因此使用全部的历史版本来增强数据。不行，耗时太长。
            df_allFiles = pd.DataFrame();
            df_groundTruth_part = pd.DataFrame();
            for i_num in range(0,len(fileList)):
                begin_time = time.perf_counter();
                array_psx_allClassifiers = [];
                
                i_file_test = fileList[i_num];
                file_test = folderName_dataset_noise + i_file_test;#测试集使用噪音数据。（干净数据的区别只在于标签不同，度量相同。因此预测时，测试集使用噪音或干净数据没区别。噪音标签只用来判断其中的误标签）
                df_test = pd.read_csv(file_test);
                df_test = df_test.rename(columns = {'relName':'instances'});
                df_train_original = pd.DataFrame();
                df_known = pd.DataFrame();#未使用
                df_unknown = pd.DataFrame();#未使用
                df_train_original_known = pd.DataFrame();#未使用
                df_train_original_unknown = pd.DataFrame();#未使用
                #---STEP 1 - 第一个版本直接读取在原始噪音数据上，5折交叉验证获得的预测结果（概率和标签），根据预测为噪音的结果，
                #          - 人工审查后获得部分人工纠正的标签数据，即获得部分ground truth
                if i_num == 0:
                    # 读取计算的概率
                    file_dataset = path_common_read_probability + i_projectName + "/" + i_file_test;
                    df_result = pd.read_csv(file_dataset);
                    array_psx_allClassifiers = df_result[list_colNames].values;
                #---STEP 2 - 后续版本在前一次获得的标签质量改进的训练集上，执行后续的噪音检测
                else:
                    file_test_groundTruth = folderName_dataset_clean_groundTruth + i_file_test;#使用基础事实来观察分类器准确性
                    df_test_groundTruth = pd.read_csv(file_test_groundTruth);
                    df_test_groundTruth = df_test_groundTruth.rename(columns = {'relName':'instances'});
                    i_file_train = fileList[i_num-1];
                    file_train = folderName_dataset_noise + i_file_train;#第一对儿的训练集使用原始噪音数据
                    df_train = pd.read_csv(file_train);
                    df_train = df_train.rename(columns = {'relName':'instances'});
                    df_train_original = df_train;
                    index_known = df_known.index.values;
                    index_unknown = df_unknown.index.values;
                    df_train_original_known = df_train_original.loc[index_known]
                    df_train_original_unknown = df_train_original.loc[index_unknown]
                    # 对于方法识别的“噪音标签”，获知其中真实的噪音标签，纠正这些标签，获得标签质量改进的缺陷数据集
                    df_train = pd.merge(df_train, df_groundTruth_part, how='left', on='instances');
                    index_realMislabels = df_train[df_train['realMislabel'] == 1].index.values;
                    df_train.loc[index_realMislabels,['bug']] = abs(df_train.loc[index_realMislabels,['bug']] - 1);
                    #训练集和测试集预处理
                    CategoryLabelName = 'bug'
                    df_train_p,df_test_p,df_test_groundTruth_p = FUNCTION_Preprocessing(df_train,df_test,df_test_groundTruth);
                    #---end---
                    
                    #---STEP 2 - Get each classifier prediction probability
    #                     features_only = copy.copy(features_label);
                    features_only = list(df_train_p)
                    features_only.remove(CategoryLabelName);
                    
                    # 特征向量和标签
                    X_array_train_metrics_only = df_train_p[features_only].values;#转成numpy数组计算快
                    X_array_test_metrics_only = df_test_p[features_only].values;#转成numpy数组计算快
                    y_labels_train = df_train_p[CategoryLabelName].values;
                    y_labels_test = df_test_p[CategoryLabelName].values;
                    y_labels_test_groundTruth = df_test_groundTruth_p[CategoryLabelName].values;
                    
                    # 在预测概率前，需要对于无法作为训练集的情况进行处理
                    #如果y全是0，1，则预测概率全是0，1
                    
                    # CVDP：连续版本。记录对于测试集每个实例的预测概率
                    str_array_psx_allClassifiers,array_labels \
                    = Class_predictionModel.FUNCTION_CaculatePredictionProbability_main(X_array_train_metrics_only,y_labels_train,X_array_test_metrics_only,y_labels_test_groundTruth);
                    array_psx_allClassifiers = str_array_psx_allClassifiers.astype(dtype=float64)
                # 根据预测概率生成dataframe
                df_result = pd.DataFrame(array_psx_allClassifiers,columns = list_colNames);#共19列
                #测试集实例名
                series_instanceNames = df_test['instances'];#实例名
                df_result["instances"] = series_instanceNames;
                temp_col = ["instances"];
                temp_col.extend(list_colNames);
                df_result = df_result.reindex(columns=temp_col);
                     
                end_time = time.perf_counter()
                run_time = end_time-begin_time
                print (i_file_test+' running time: ',run_time)
                #---end---
                
                # 在获得的预测结果上，计算各方法识别的噪音标签，通过人工检测，获取部分ground truth，然后进行下一轮
                #---STEP 3 - 计算每种方法的预测结果
#                 FUNCTION_CalculatePredictionResultALLMethod();
                
                #---读取噪音测试集原始标签---#
                df_original = df_test#需要原始测试集中的代码行
                original_labels = df_original['bug'].values;
                origin_index_all = df_original['instances'].index.values;#ndarray
                origin_index_1 = df_original[df_original['bug']==1].index.values;#ndarray
                origin_index_0 = df_original[df_original['bug']==0].index.values;#ndarray
                #---end---#
                
                #---ground truth: 读取误标签的ground truth---#
                file_mislabels_groundTruth = path_common_mislabels_groundTruth + i_projectName + "/" + i_file_test;
                df_mislabels_groundTruth = pd.read_csv(file_mislabels_groundTruth);
                actual_label_errors_all = df_mislabels_groundTruth['index'].values;#ndarray
                actual_label_errors_1 = df_mislabels_groundTruth[df_mislabels_groundTruth['bug']==1]['index'].values;#ndarray
                actual_label_errors_0 = df_mislabels_groundTruth[df_mislabels_groundTruth['bug']==0]['index'].values;#ndarray
                #---end---#
                
                list_orginIndex = [origin_index_all,origin_index_1,origin_index_0];
                list_index_realMislabels = [actual_label_errors_all,actual_label_errors_1,actual_label_errors_0];
                
                #---读取计算的概率---#
                array_probability_allClassfiers = df_result[list_colNames].values;
                
                #硬投票：根据所有分类器的预测概率，生成误标签的投票矩阵
                matrix_probability, matrix_votingMeasures = FUNCTION_majorityVote(array_probability_allClassfiers,original_labels);
                # 单个分类器的预测概率
                df_result_probability = pd.DataFrame(matrix_probability,columns = list_names_classifiers);#共10列
                df_result_probability["instances"] = df_result['instances'].values;
                temp_col = ["instances"];
                temp_col.extend(list_names_classifiers);
                df_result_probability = df_result_probability.reindex(columns=temp_col);
                # 硬投票矩阵
                df_result_votingMeasures_baseline = pd.DataFrame(matrix_votingMeasures,columns = columns_voting);
                df_result_votingMeasures_baseline["instances"] = df_result['instances'].values;
                temp_col = ["instances"];
                temp_col.extend(columns_voting);
                df_result_votingMeasures_baseline = df_result_votingMeasures_baseline.reindex(columns=temp_col);
                #为NR方法，只保留取"(OV)vote","(MV)vote"两列
                df_result_votingMeasures_baseline_NR = df_result_votingMeasures_baseline[["(OV)vote","(OV)E_probability"]];
                df_result_votingMeasures_baseline_NR = df_result_votingMeasures_baseline_NR.rename(columns={"(OV)vote":"(NR-OV)vote","(OV)E_probability":"(NR-OV)E_probability"})
                #---end---#
                
                #---方法识别的误标签---#
                #baseline CNDC
                # 由于CNDC方法其中一部分不需要进行log变换，为该方法单独读取
                df_file_metrics_only = df_test.drop(['instances','bug'],axis=1);#删除不需要的列
                array_metrics_only = df_file_metrics_only.values;#转成numpy数组计算快
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
                #---end---#
                
                #---识别的误标签结果---#
                #原始标签，留着观察用
                df_dataset_labels = df_test[['instances','bug']];
                #真实误标签的结果
                df_mislabels_groundTruth['realMislabel'] = 1;
                df_mislabels_groundTruth = df_mislabels_groundTruth[['relName','realMislabel']];
                df_mislabels_groundTruth = df_mislabels_groundTruth.rename(columns = {'relName':'instances'});
                
                #记录多数投票的票数（误标签预测结果），平均概率
                df_result_mislabel = pd.merge(df_dataset_labels, df_result_votingMeasures_baseline[['instances',"(MV)vote","(MV)E_probability",]], how='left', on='instances');
                #记录预测误标签的结果
                df_result_mislabel['predictedMislabel'] = 0;
                df_result_mislabel.loc[list_mislabels_CNDC[0],['predictedMislabel']] = 1;
                #记录真实误标签的结果
                df_result_mislabel = pd.merge(df_result_mislabel, df_mislabels_groundTruth, how='left', on='instances');
                df_result_mislabel['realMislabel'].fillna(0, inplace=True);
                
                if i_num != 0:
                    dir_path_saved = path_common_saved_mislabel;
                    if not os.path.exists(dir_path_saved):
                        os.makedirs(dir_path_saved);
                    #度量和标签的交集的存储文件名
                    path_saved_fileName = dir_path_saved + i_file_test;
                    df_result_mislabel.to_csv(path_saved_fileName,index=False);#不保存行索引
                #---end---#
                
                #---STEP 4 - NEAT
                if i_num != 0:
                    list_mislabels_HR1 = [];
                    list_mislabels_HR2 = [];
                    list_mislabels_NEAT = [];
                    
#                     df_current_labels_n_low = df_train_original_known
                    df_current_labels_n_low = df_train_original;
                    df_current_labels_g_low = df_groundTruth_part;
                    Class_NEAT.FUNCTION_HeuristicRules(df_current_labels_n_low, df_current_labels_g_low, df_test);
                    list_mislabels_HR1 = Class_NEAT.list_mislabels_HR1;
                    list_mislabels_HR2 = Class_NEAT.list_mislabels_HR2;
                    list_mislabels_NEAT = Class_NEAT.list_mislabels_NEAT;
                    #===根据预测标签和测试集实际标签，分别计算识别三种类型误标签的准确率===#
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
                            
                            #baseline CNDC
                            errors_idx = list_mislabels_CNDC[i_type_mislabels];
                            list_result = FUNCTION_calculateIndicators_baseline("(CNDC)",list_measureNames_baseline_MV,df_result_votingMeasures_baseline,errors_idx,index_realMislabels_currentType,originIndex_currentType);
                            list_indicators_allBaselines.extend(list_result);
                            
                            #baseline: 1A+2A模型
                            errors_idx = list_mislabels_HR1[i_type_mislabels];
                            list_result = FUNCTION_calculateIndicators_NEAT("HR1",errors_idx,index_realMislabels_currentType,originIndex_currentType);
                            list_indicators_allBaselines.extend(list_result);
                            
                            #baseline: 1B+2B模型
                            errors_idx = list_mislabels_HR2[i_type_mislabels];
                            list_result = FUNCTION_calculateIndicators_NEAT("HR2",errors_idx,index_realMislabels_currentType,originIndex_currentType);
                            list_indicators_allBaselines.extend(list_result);
                            
                            #baseline: NEAT模型
                            errors_idx = list_mislabels_NEAT[i_type_mislabels];
                            list_result = FUNCTION_calculateIndicators_NEAT("NEAT",errors_idx,index_realMislabels_currentType,originIndex_currentType);
                            list_indicators_allBaselines.extend(list_result);
                            
                            #存储单分类器和多分类器的投票的评价指标计算结果
                            dir_path_saved = path_saved_common + mislabels_currentType + "/";
                            if not os.path.exists(dir_path_saved):
                                os.makedirs(dir_path_saved);
                            path_saved_fileName = dir_path_saved + i_file_test;
                            df_allMethods = pd.DataFrame(list_indicators_allBaselines, columns=columns_indicators);
                            df_allMethods.to_csv(path_saved_fileName,index=False);#不保存行索引
                            
                            print("%s %s finish"%(i_projectName,i_file_test));
                    #===end===#
                
                df_result_mislabel = df_result_mislabel[['instances',"predictedMislabel","realMislabel"]];
                df_groundTruth_part = df_result_mislabel[(df_result_mislabel['predictedMislabel'] == 1) & (df_result_mislabel['realMislabel'] == 1)];
                df_groundTruth_part = df_groundTruth_part[['instances',"realMislabel"]];
                
                #根据低版本上已知标签部分，和未知标签部分，来预测噪音标签
                df_known = df_result_mislabel[df_result_mislabel["predictedMislabel"] == 1];
                df_unknown = df_result_mislabel[df_result_mislabel["predictedMislabel"] != 1];
                
        print("all finish")
        
        
    sys.exit()
