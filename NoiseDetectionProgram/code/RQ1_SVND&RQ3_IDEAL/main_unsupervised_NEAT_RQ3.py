import numpy as np
import pandas as pd
import copy
import os
# from collections import deque
# import copy
import Class_basic.EvaluationIndicator as EI
import Class_basic.fileNameSorting as FNS


#===根据相应的置信列，计算性能指标===#
def FUNCTION_calculateIndicators_NEAT(name_baseline,errors_idx,actual_label_errors,origin_index):
    errors_idx = np.array(errors_idx)
    actual_label_errors = np.array(actual_label_errors)
    
    list_result = [];
    if errors_idx.size!=0:
        #LOC的顺序已排
        list_indicators = Class_Evaluation.calculateIndicators(errors_idx,actual_label_errors,origin_index);
        TP = Class_Evaluation.getTP();
        temp_oneRow = [name_baseline,TP];
        temp_oneRow.extend(list_indicators);
        list_result.append(temp_oneRow);
    else:
        list_indicators =  Class_Evaluation.theoreticalWorstValues();
        TP = 0;
        temp_oneRow = [name_baseline,TP];
        temp_oneRow.extend(list_indicators);
        list_result.append(temp_oneRow);
    return list_result;
#===end===#

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为200，默认为50
pd.set_option('max_colwidth',2000)

#对于个别项目的跨版本实例要加模糊匹配处理，因为路径变化极大。
#一个实例，在相邻的前一个历史版本是/不是误标签，在下一版本是/不是误标签的概率高不高
if __name__ == "__main__":
#     dataset_style_g = "IND-JLMIV+R-2020";#这个是ground truth，没有对应的误标签
    dataset_style_n = "6M-SZZ-2020";
    
    data_style = "filteredIL";
#     data_style = "unfilteredIL";
    
    path_common = "D:/DataSets/"+data_style+"/";#调整成你的数据集存储路径
    #数据集读取路径
    path_common_read_dataset = path_common + "(selected)dataSets/" + dataset_style_n + "/";
    #真实误标签读取路径
    path_common_mislabels_groundTruth = path_common + "(groundTruth)mislabels/" + dataset_style_n + "/";
    
    #存储列名
    columns_indicators = ["name", "TP", "accuracy", "precision", "recall", "F1", "FAR", "D2H", 
                          "IFA", "Inspect", "LOC_Inspect", "NLPTLOC", "AP",
                          "P@1", "P@5", "P@10",];
    #存储列名
    list_colNames = ["file","low","high","intersection_lowHigh","mis_low","mis_high",
                     "misLowRealInHigh","misLowRealInmisHighReal",
                     "precision_1A","recall_1A","F1_1A",
                     "precision_1A2A","recall_1A2A","F1_1A2A",
                     "precision_1B2B","recall_1B2B","F1_1B2B",
                     "precision_1A2A1B2B","recall_1A2A1B2B","F1_1A2A1B2B",];
    
    #两种最低版本
    list_lowType = ["Continuous"]
    list_filename_saved = ["indicators.csv","indicators.csv"]
    #三种误标签文件夹名
    list_folderRead = ["(all)mislabels","(buggy)mislabels","(non-buggy)mislabels"];
    
    
    #实例化评价指标类
    Class_Evaluation = EI.EvaluationIndicator();
    for i_lowType,i_filename_saved in zip(list_lowType,list_filename_saved):
        
        #存储路径
        path_common_saved_mislabel = path_common + "(model)NEAT/" + i_lowType + "/(result)mislabel/" + dataset_style_n + "/";#单个文件的误标签预测结果的存储路径
        path_common_saved = path_common + "(model)NEAT/" + i_lowType + "/evaluationIndicators/" + dataset_style_n + "/";#所有文件的F1等性能值的存储路径
        path_common_saved_all = path_common + "(model)NEAT/" + i_lowType + "/evaluationIndicators_all/" + dataset_style_n + "/";#所有文件的F1等性能值的存储路径
            
        list_allRows = [];
        #项目列表
        list_projectName = os.listdir(path_common_read_dataset);
        for i_projectName in list_projectName:
            folderName_project_g = path_common_mislabels_groundTruth + i_projectName + "/";
            folderName_project_n = path_common_read_dataset + i_projectName + "/";
            fileList = os.listdir(folderName_project_n);#文件
            fileList = FNS.sort_insert_filename(fileList);# 按文件名版本排序
            
            for i_num in range(0,len(fileList)-1):
                if i_lowType == "Continuous":
                    i_file_low = fileList[i_num];
                elif i_lowType == "Single":
                    i_file_low = fileList[0];
                i_file_high = fileList[i_num+1];
                #读取文件名
                file_current_labels_g_low = folderName_project_g + i_file_low;
                file_current_labels_g_high = folderName_project_g + i_file_high;
                file_current_labels_n_low = folderName_project_n + i_file_low;
                file_current_labels_n_high = folderName_project_n + i_file_high;
                
                #高版本原始标签
                df_current_labels_n_high = pd.read_csv(file_current_labels_n_high);
#                     origin_index = df_current_labels_n_high['relName'].index.values;#ndarray
                original_labels = df_current_labels_n_high['bug'].values;
                origin_index_all = df_current_labels_n_high['relName'].index.values;#ndarray
                origin_index_1 = df_current_labels_n_high[df_current_labels_n_high['bug']==1].index.values;#ndarray
                origin_index_0 = df_current_labels_n_high[df_current_labels_n_high['bug']==0].index.values;#ndarray
                
                #高版本真实误标签
                df_current_labels_g_high = pd.read_csv(file_current_labels_g_high);
                actual_label_errors_all = df_current_labels_g_high['index'].values;#ndarray
                actual_label_errors_1 = df_current_labels_g_high[df_current_labels_g_high['bug']==1]['index'].values;#ndarray
                actual_label_errors_0 = df_current_labels_g_high[df_current_labels_g_high['bug']==0]['index'].values;#ndarray
                
                list_orginIndex = [origin_index_all,origin_index_1,origin_index_0];
                list_index_realMislabels = [actual_label_errors_all,actual_label_errors_1,actual_label_errors_0];
                
                #低版本原始标签
                df_current_labels_n_low = pd.read_csv(file_current_labels_n_low);
                #低版本真实误标签
                df_current_labels_g_low = pd.read_csv(file_current_labels_g_low);
                #真实干净标签
                index_mis_low = df_current_labels_g_low['index'].values;
                df_current_labels_clean_low = df_current_labels_n_low.loc[~df_current_labels_n_low.index.isin(index_mis_low)];
                
                #不保留特征列
                df_current_labels_n_low = df_current_labels_n_low[['relName','bug']];
                df_current_labels_n_high = df_current_labels_n_high[['relName','loc','bug']];
                df_current_labels_n_high['index'] = df_current_labels_n_high.index.values;
                df_current_labels_clean_low = df_current_labels_clean_low[['relName']];
    #             df_current_labels_clean_high = df_current_labels_clean_high[['relName','bug']];
                df_current_labels_g_low = df_current_labels_g_low[['relName']];
                df_current_labels_g_high = df_current_labels_g_high[['relName']];
                
                #原始标签中，低版本和高版本实例的交集
                df_intersection_allInstances = pd.merge(df_current_labels_n_low, df_current_labels_n_high, how='inner', on='relName');
#                 #1A(TP+FP)：低版本中带误标签的实例和高版本实例的交集
#                 df_1A = pd.merge(df_current_labels_g_low, df_current_labels_n_high, how='inner', on='relName');
#                 #1A(TP)：低版本中带误标签的实例和高版本真实误标签实例的交集
#                 df_TP_1A = pd.merge(df_current_labels_g_low, df_current_labels_g_high, how='inner', on='relName');
                
                #交集之外的预测为误标签还是非误标签？按理说不知道，实际预测为非误标签
                
                #1B:和1A本质上相同。前一版本中不是误标签，下一版本中也不是误标签
                #2A:原始标签中，低版本和高版本实例的交集中，标签相同的实例
                temp_bool = df_intersection_allInstances['bug_x']==df_intersection_allInstances['bug_y']
                df_2A = df_intersection_allInstances.loc[temp_bool]
                #2B:原始标签中，低版本和高版本实例的交集中，标签不同的实例
                temp_bool = df_intersection_allInstances['bug_x']!=df_intersection_allInstances['bug_y']
                df_2B = df_intersection_allInstances.loc[temp_bool]
                #1A+2A(TP+FP)：原始标签中，低版本和高版本实例的交集中，标签相同的实例，且低版本中是误标签
                df_1A2A = pd.merge(df_2A, df_current_labels_g_low, how='inner', on='relName');
#                 #1A+2A(TP)
#                 df_TP_1A2A = pd.merge(df_1A2A, df_current_labels_g_high, how='inner', on='relName');
                #1B+2B(TP+FP)：原始标签中，低版本和高版本实例的交集中，标签不同的实例，且低版本中不是误标签
                df_1B2B = pd.merge(df_2B, df_current_labels_clean_low, how='inner', on='relName');
#                 #1B+2B(TP)
#                 df_TP_1B2B = pd.merge(df_1B2B, df_current_labels_g_high, how='inner', on='relName');
                #1A+2A(TP+FP)∪1B+2B(TP+FP)
                df_1A2A1B2B = df_1A2A.append(df_1B2B)
#                 #1A+2A(TP)∪1B+2B(TP)
#                 df_TP_1A2A1B2B = df_TP_1A2A.append(df_TP_1B2B)
                
                #===存识别的误标签结果===#
                #记录原始噪音，低版本和高版本的bug标签
                df_current_labels_n_low = df_current_labels_n_low.rename(columns = {'bug':'bug_low'})
                df_current_labels_n_high = df_current_labels_n_high.rename(columns = {'bug':'bug_high'})
                df_result_mislabel = pd.merge(df_current_labels_n_low, df_current_labels_n_high, how='right', on='relName');
                #记录HMIL方法的误标签识别结果
                df = copy.deepcopy(df_1A2A1B2B)
                df['predictedMislabel'] = 1;#增加误标签列，表示是否被预测为误标签
                df = df[['relName','index','predictedMislabel']];
                df_result_mislabel = pd.merge(df_result_mislabel, df, how='left', on='relName');
                df_result_mislabel['predictedMislabel'].fillna(0, inplace=True);
                #记录低版本上真实误标签的结果
                df_current_labels_g_low['realMislabel_low'] = 1;
                df_current_labels_g_low = df_current_labels_g_low[['relName','realMislabel_low']];
                df_result_mislabel = pd.merge(df_result_mislabel, df_current_labels_g_low, how='left', on='relName');
                #记录高版本上真实误标签的结果
                df_current_labels_g_high['realMislabel_high'] = 1;
                df_current_labels_g_high = df_current_labels_g_high[['relName','realMislabel_high']];
                df_result_mislabel = pd.merge(df_result_mislabel, df_current_labels_g_high, how='left', on='relName');
#                     df_result_mislabel['realMislabel'].fillna(0, inplace=True);
                
                #记录训练集和测试集对儿
                start_position = i_file_low.rfind("-")+1;
                end_position = i_file_low.rfind(".csv");
                versino_i = i_file_low[start_position:end_position];
                start_position = i_file_high.rfind("-")+1;
                end_position = i_file_high.rfind(".csv");
                versino_iadd1 = i_file_high[start_position:end_position];
                str_pair = '('+versino_i+','+versino_iadd1+')';
                
                dir_path_saved = path_common_saved_mislabel + "/";
                if not os.path.exists(dir_path_saved):
                    os.makedirs(dir_path_saved);
                #度量和标签的交集的存储文件名
                path_saved_fileName = dir_path_saved + i_projectName + '-' + str_pair + '.csv';
                df_result_mislabel.to_csv(path_saved_fileName,index=False);#不保存行索引
                #===end===#
                
                #1A+2A
                df_predictedClean = df_1A2A.loc[df_1A2A['bug_y'] == 1];
                df_predictedClean = df_predictedClean.sort_values(by="loc", ascending=True);
                errors_idx_1_sort_ASC = df_predictedClean['index'].values;
                df_predictedBuggy = df_1A2A.loc[df_1A2A['bug_y'] == 0];
                df_predictedBuggy = df_predictedBuggy.sort_values(by="loc", ascending=False);
                errors_idx_0_sort_DES = df_predictedBuggy['index'].values;
                errors_idx_all_sort = np.concatenate((errors_idx_1_sort_ASC,errors_idx_0_sort_DES), axis=0);
                list_mislabels_1A2A = [errors_idx_all_sort,errors_idx_1_sort_ASC,errors_idx_0_sort_DES];
                
                #1B+2B
                df_predictedClean = df_1B2B.loc[df_1B2B['bug_y'] == 1];
                df_predictedClean = df_predictedClean.sort_values(by="loc", ascending=True);
                errors_idx_1_sort_ASC = df_predictedClean['index'].values;
                df_predictedBuggy = df_1B2B.loc[df_1B2B['bug_y'] == 0];
                df_predictedBuggy = df_predictedBuggy.sort_values(by="loc", ascending=False);
                errors_idx_0_sort_DES = df_predictedBuggy['index'].values;
                errors_idx_all_sort = np.concatenate((errors_idx_1_sort_ASC,errors_idx_0_sort_DES), axis=0);
                list_mislabels_1B2B = [errors_idx_all_sort,errors_idx_1_sort_ASC,errors_idx_0_sort_DES];
                
                #对识别的误标签按LOC值排序
                df_predictedClean = df_1A2A1B2B.loc[df_1A2A1B2B['bug_y'] == 1];
                df_predictedClean = df_predictedClean.sort_values(by="loc", ascending=True);
                errors_idx_1_sort_ASC = df_predictedClean['index'].values;
                df_predictedBuggy = df_1A2A1B2B.loc[df_1A2A1B2B['bug_y'] == 0];
                df_predictedBuggy = df_predictedBuggy.sort_values(by="loc", ascending=False);
                errors_idx_0_sort_DES = df_predictedBuggy['index'].values;
                errors_idx_all_sort = np.concatenate((errors_idx_1_sort_ASC,errors_idx_0_sort_DES), axis=0);
                list_mislabels = [errors_idx_all_sort,errors_idx_1_sort_ASC,errors_idx_0_sort_DES];
                
                #分别计算三种类型的误标签识别率
                for i_type_mislabels in range(len(list_folderRead)):
                    #存所有baseline结果
                    list_indicators_current = [];
                    
                    mislabels_currentType = list_folderRead[i_type_mislabels];
                    originIndex_currentType = list_orginIndex[i_type_mislabels];
                    index_realMislabels_currentType = list_index_realMislabels[i_type_mislabels];
                    
                    #需要过滤buggy标签/clean标签无噪音标签的，否则画柱状图时，会拉低识别buggy标签/clean标签中的噪音标签的性能
                    if index_realMislabels_currentType.size!=0:
                        #需要原始数据集中的代码行
                        df_original_type = df_current_labels_n_high.loc[originIndex_currentType,:];
                        Class_Evaluation.setDataset(df_original_type);
                        
                        #baseline: 1A+2A模型
                        errors_idx = list_mislabels_1A2A[i_type_mislabels];
                        list_result = FUNCTION_calculateIndicators_NEAT("HR1",errors_idx,index_realMislabels_currentType,originIndex_currentType);
                        list_indicators_current.extend(list_result);
                        
                        #baseline: 1B+2B模型
                        errors_idx = list_mislabels_1B2B[i_type_mislabels];
                        list_result = FUNCTION_calculateIndicators_NEAT("HR2",errors_idx,index_realMislabels_currentType,originIndex_currentType);
                        list_indicators_current.extend(list_result);
                        
                        #baseline: NEAT模型
                        errors_idx = list_mislabels[i_type_mislabels];
                        list_result = FUNCTION_calculateIndicators_NEAT("NEAT",errors_idx,index_realMislabels_currentType,originIndex_currentType);
                        list_indicators_current.extend(list_result);
                        
                        #存储NEAT模型的评价指标计算结果
                        dir_path_saved = path_common_saved + mislabels_currentType + "/";
                        if not os.path.exists(dir_path_saved):
                            os.makedirs(dir_path_saved);
                        path_saved_fileName = dir_path_saved + i_projectName + '-' + str_pair + '.csv';
                        df_allMethods = pd.DataFrame(list_indicators_current, columns=columns_indicators);
                        df_allMethods.to_csv(path_saved_fileName,index=False);#不保存行索引
                        
                        print("%s %s finish"%(i_projectName,i_projectName + '-' + str_pair + '.csv'))
                        
    print("all finish")







