'''
Created on 2020年9月24日

@author: njdx
'''
# import cleanlab
import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.neural_network import MLPClassifier


class EnsembleLearning(object):
    
    s = [];
    psx = [];
    num_classifiers = 0;
    votingMatrix_margin = [];
    votingMatrix_probability = [];
    #根据投票矩阵中存储的margin，存储margin的几个指标
    matrix_votingMeasures = [];
    
    def __init__(self,num_classifiers):
        self.s = [];
        self.psx = [];
        self.num_classifiers = num_classifiers;
        self.votingMatrix_margin = [];#margin，横向置信
        self.votingMatrix_probability = [];#概率
        self.matrix_votingMeasures = [];#多个分类器的margin的几个指标的计算
    
    def FUNCTION_initialization(self,s):
        self.s = [];
        self.votingMatrix_margin = [];#margin，横向置信
        self.votingMatrix_probability = [];#概率
        self.matrix_votingMeasures = [];#多个分类器的margin的几个指标的计算
        
        self.s = s;
        num_instances = len(s);#实例数
        # Initiate: voting matrix
        self.votingMatrix_margin = np.zeros((num_instances,self.num_classifiers), dtype=np.float);
        self.votingMatrix_probability = np.zeros((num_instances,self.num_classifiers), dtype=np.float);
    
#     def weighted_avg(self, values, weights):
#         """
#         Return the weighted average and standard deviation.
#     
#         values, weights -- Numpy ndarrays with the same shape.
#         """
#         average = np.average(values, axis=1, weights=weights)
#         average = average.reshape(average.shape[0],1)#对一维数组进行转置需要借助reshape来完成
#         return average
    
    #===Calculate n measures of vote===#
    def FUNCTION_CalculateVotingMeasures(self):
        rows = self.votingMatrix_margin.shape[0];#行数
        result_matrix_votingMeasures = np.zeros((rows,6), dtype=np.float);#每个实例共有6个结果
        
        votingMatrix_margin_abs = np.fabs(self.votingMatrix_margin)
        idx_nonzeroRow_margin = np.where(votingMatrix_margin_abs.any(axis=1))[0]#获取非零行的下标
        
        for i in idx_nonzeroRow_margin:
            oneRow = votingMatrix_margin_abs[i];
            oneRow_probability = self.votingMatrix_probability[i];
            # 置信投票矩阵中行是实例，列是分类器，单元格是横向信心值
            vote = np.count_nonzero(oneRow)
            vote_OV = 0;
            
            vote_MV = 0;
#             mean_MV = 0;
#             std_MV = 0; # 计算总体标准差
#             ED_MV = 0;
            vote_CV = 0;
#             mean_CV = 0;
#             std_CV = 0; # 计算总体标准差
#             ED_CV = 0;
            mean_probability_OV = 0;
            mean_probability_MV = 0;
#             std_probability_MV = 0;
#             ED_probability_MV = 0;
#             mean_weight_MV = 0;
            mean_probability_CV = 0;
#             std_probability_CV = 0;
#             ED_probability_CV = 0;
#             mean_weight_CV = 0;
            # One (favor) Voting, i.e. Noise Ranking
            if vote > 0:
                vote_OV = vote;
                mean_probability_OV = np.mean(oneRow_probability);
            
            # Majority Voting (MV)
            if vote >= (self.num_classifiers/2):
                vote_MV = vote;
#                 mean_MV = np.mean(oneRow);
#                 std_MV = np.std(oneRow); # 计算总体标准差
#                 ED_MV = mean_MV/std_MV;
                mean_probability_MV = np.mean(oneRow_probability);
#                 std_probability_MV = np.std(oneRow_probability); # 计算总体标准差
#                 ED_probability_MV = mean_probability_MV/std_probability_MV;
#                 mean_weight_MV = np.average(oneRow_probability, weights=list_confidenceWeights)
                
            #Consensus Voting (CV) 
            if vote == self.num_classifiers:
                vote_CV = vote;
#                 mean_CV = np.mean(oneRow);
#                 std_CV = np.std(oneRow); # 计算总体标准差
#                 ED_CV = mean_CV/std_CV;
                mean_probability_CV = np.mean(oneRow_probability);
#                 std_probability_CV = np.std(oneRow_probability); # 计算总体标准差
#                 ED_probability_CV = mean_probability_CV/std_probability_CV;
#                 mean_weight_CV = np.average(oneRow_probability, weights=list_confidenceWeights)
            
            result_matrix_votingMeasures[i] = [vote_OV,mean_probability_OV,
                                               vote_MV,mean_probability_MV,
                                               vote_CV,mean_probability_CV];
        
        self.matrix_votingMeasures = result_matrix_votingMeasures;
    #===end===#
    
    #===Calculate n measures of vote===#
    def FUNCTION_CalculateVotingMeasures_weight(self,i_file_weights_classfiers):
        array_confidenceWeights = np.array(i_file_weights_classfiers);
        rank = np.argsort(array_confidenceWeights);# 正序输出索引，从小到大
        w = np.array(range(1,len(array_confidenceWeights)+1))
        w_arr = w[np.argsort(rank)];#将数组按照F1值大小转换为权重
        list_confidenceWeights = w_arr
        
        rows = self.votingMatrix_margin.shape[0];#行数
        result_matrix_votingMeasures = np.zeros((rows,16), dtype=np.float);#每个实例共有16个结果
        
        votingMatrix_margin_abs = np.fabs(self.votingMatrix_margin)
        idx_nonzeroRow_margin = np.where(votingMatrix_margin_abs.any(axis=1))[0]#获取非零行的下标
        
        for i in idx_nonzeroRow_margin:
            oneRow = votingMatrix_margin_abs[i];
            oneRow_probability = self.votingMatrix_probability[i];
#                 oneRow_lo = votingMatrix_longitudinalMargin_abs[i];
            # 置信投票矩阵中行是实例，列是分类器，单元格是横向信心值
            vote = np.count_nonzero(oneRow)
#             mean_each = np.mean(oneRow)
#             std_each = np.std(oneRow) # 计算总体标准差
#             ED_each = mean_each/std_each
            vote_MV = 0;
            mean_MV = 0;
            std_MV = 0; # 计算总体标准差
            ED_MV = 0;
#             sum_MV = 0;
#             SUMD_MV = 0;
            vote_CV = 0;
            mean_CV = 0;
            std_CV = 0; # 计算总体标准差
            ED_CV = 0;
#             sum_CV = 0;
#             SUMD_CV = 0;
            mean_probability_MV = 0;
            std_probability_MV = 0;
            ED_probability_MV = 0;
            mean_weight_MV = 0;
            mean_probability_CV = 0;
            std_probability_CV = 0;
            ED_probability_CV = 0;
            mean_weight_CV = 0;
            # Majority Voting (MV)
            if vote >= (self.num_classifiers/2):
                vote_MV = vote;
                mean_MV = np.mean(oneRow);
                std_MV = np.std(oneRow); # 计算总体标准差
                ED_MV = mean_MV/std_MV;
                mean_probability_MV = np.mean(oneRow_probability);
                std_probability_MV = np.std(oneRow_probability); # 计算总体标准差
                ED_probability_MV = mean_probability_MV/std_probability_MV;
                mean_weight_MV = np.average(oneRow_probability, weights=list_confidenceWeights)
                
            #Consensus Voting (CV) 
            if vote == self.num_classifiers:
                vote_CV = vote;
                mean_CV = np.mean(oneRow);
                std_CV = np.std(oneRow); # 计算总体标准差
                ED_CV = mean_CV/std_CV;
                mean_probability_CV = np.mean(oneRow_probability);
                std_probability_CV = np.std(oneRow_probability); # 计算总体标准差
                ED_probability_CV = mean_probability_CV/std_probability_CV;
                mean_weight_CV = np.average(oneRow_probability, weights=list_confidenceWeights)
            
            result_matrix_votingMeasures[i] = [vote_MV,mean_MV,std_MV,ED_MV,
                                               mean_probability_MV,std_probability_MV,ED_probability_MV,mean_weight_MV,
                                               vote_CV,mean_CV,std_CV,ED_CV,
                                               mean_probability_CV,std_probability_CV,ED_probability_CV,mean_weight_CV];
        
        self.matrix_votingMeasures = result_matrix_votingMeasures;
    #===end===#
    
    #===Vote===#
    def FUNCTION_Vote(self,array_lateralMargin,array_probability,
                      label_errors_idx,i_classifier):
        # 给置信投票矩阵赋值
        for i in range(len(label_errors_idx)):
            i_row = label_errors_idx[i]
            self.votingMatrix_margin[i_row][i_classifier] = array_lateralMargin[i];
            self.votingMatrix_probability[i_row][i_classifier] = array_probability[i];
    #===end===#
    
    #===Get probability: prediction probability===#
    def FUNCTION_GetProbability(self,label_errors_bool,label_errors_idx):#传入真实标签列表，返回生成的误标签列表和误标签的下标
        # self confidence is the holdout probability that an example
        # belongs to its given class label
        self_confidence = np.array(
            [np.mean(self.psx[i][self.s[i]]) for i in label_errors_idx]
        )
        margin = self_confidence - self.psx[label_errors_bool].max(axis=1)
        return margin
    #===end===#    
    
    #===Get self confidence: lateral margin===#
    def FUNCTION_GetLateralMargin(self,label_errors_bool,label_errors_idx):#传入真实标签列表，返回生成的误标签列表和误标签的下标
        # self confidence is the holdout probability that an example
        # belongs to its given class label
        self_confidence = np.array(
            [np.mean(self.psx[i][self.s[i]]) for i in label_errors_idx]
        )
        margin = self_confidence - self.psx[label_errors_bool].max(axis=1)
        return margin
    #===end===#
    
    #===Remove label errors if given label == model prediction===#
    def FUNCTION_RemoveLabelErrors(self,label_errors_bool):#传入真实标签列表，返回生成的误标签列表和误标签的下标
        # Remove label errors if given label == model prediction
        for i, pred_label in enumerate(self.psx.argmax(axis=1)):
            # np.all let's this work for multi_label and single label
            if label_errors_bool[i] and np.all(pred_label == self.s[i]):
                label_errors_bool[i] = False 
        return label_errors_bool
    #===end===#
    
    #===Ensemble learning===#
    def FUNCTION_EnsembleLearning_main(self,psx,i_classifier):
        self.psx = [];
        self.psx = psx;
        len_instances = len(self.s);
        labels_predicted = np.zeros((len_instances), dtype=int);
        # psx大于等于0.5
        idx_predicted_1 = np.where(self.psx[:, 1] > 0.5)[0];
        labels_predicted[idx_predicted_1] = 1;
        # 和带噪音的s标签比，挑出预测错误的标签，预测错误的标签被认为是类噪音
        label_errors_bool = ~(labels_predicted == self.s);
        # Remove label errors if given label == model prediction
#         label_errors_bool = self.FUNCTION_RemoveLabelErrors(label_errors_bool);
        label_errors_idx = np.arange(len(label_errors_bool))[label_errors_bool];
        array_lateralMargin = self.FUNCTION_GetLateralMargin(label_errors_bool,label_errors_idx)
        array_probability = self.psx[label_errors_bool].max(axis=1)
        self.FUNCTION_Vote(array_lateralMargin,array_probability,label_errors_idx,i_classifier)
    #===end===#
    
    #===softVoting===#
    def FUNCTION_EnsembleLearning_softVoting(self,array_probability_allClassfiers,s):
        len_instances = len(s);#实例数
#         # 记录结果文件
#         result_matrix_votingMeasures = np.zeros((len_instances,6), dtype=np.float);#每个实例共有6个结果
#         
#         # Initiate: voting matrix
#         votingMatrix_margin = np.zeros((len_instances,1), dtype=np.float);
#         votingMatrix_probability = np.zeros((len_instances,1), dtype=np.float);
        
        psx_all_0 = [];
        psx_all_1 = [];
        for i_classifier in range(0, self.num_classifiers*2, 2):#每次读当前分类器预测为0和1的列
            col_label_0 = i_classifier;
            col_label_1 = i_classifier+1;
            array_label_0 = array_probability_allClassfiers[:,col_label_0]
            array_label_1 = array_probability_allClassfiers[:,col_label_1]
#             array_margin = array_label_1 - array_label_0
            array_label_0 = array_label_0.reshape(array_label_0.shape[0],1)#对一维数组进行转置需要借助reshape来完成
            array_label_1 = array_label_1.reshape(array_label_1.shape[0],1)#对一维数组进行转置需要借助reshape来完成
#             array_margin = array_margin.reshape(array_margin.shape[0],1)#对一维数组进行转置需要借助reshape来完成
            if i_classifier == 0:
                psx_all_0 = array_label_0
                psx_all_1 = array_label_1
#                 margin_all = array_margin
            else:
                psx_all_0 = np.concatenate((psx_all_0,array_label_0),axis=1)
                psx_all_1 = np.concatenate((psx_all_1,array_label_1),axis=1)
#                 margin_all = np.concatenate((margin_all,array_margin),axis=1)
        psx_all_0_avg = np.mean(psx_all_0, axis=1) # 计算每一行的均值
        psx_all_1_avg = np.mean(psx_all_1, axis=1) # 计算每一行的均值
#         margin_all_avg = np.mean(margin_all, axis=1) # 计算每一行的均值
        psx_all_0_avg = psx_all_0_avg.reshape(psx_all_0_avg.shape[0],1)#对一维数组进行转置需要借助reshape来完成
        psx_all_1_avg = psx_all_1_avg.reshape(psx_all_1_avg.shape[0],1)#对一维数组进行转置需要借助reshape来完成
#         margin_all_avg = margin_all_avg.reshape(margin_all_avg.shape[0],1)#对一维数组进行转置需要借助reshape来完成
        psx = np.concatenate((psx_all_0_avg,psx_all_1_avg),axis=1)
        
        #计算方差，和ED
#         std_0 = np.std(psx_all_0, axis=1); # 计算每一行的标准差
#         std_1 = np.std(psx_all_1, axis=1); # 计算每一行的标准差
        
#         std_margin = np.std(margin_all, axis=1); # 计算每一行的标准差
#         std_0_temp = std_0.reshape(std_0.shape[0],1)
#         std_1_temp = std_1.reshape(std_1.shape[0],1)
#         std_margin_temp = std_margin.reshape(std_margin.shape[0],1)
#         ED_0 = psx_all_0_avg/std_0_temp;
#         ED_1 = psx_all_1_avg/std_1_temp;
#         ED_margin = margin_all_avg/std_margin_temp;
        
        #识别误标签
        labels_predicted = np.zeros((len_instances), dtype=int);
        # psx_all_1_avg大于等于0.5
        idx_predicted_1 = np.where(psx[:, 1] > psx[:, 0])[0];
        labels_predicted[idx_predicted_1] = 1;
        # 和带噪音的s标签比，挑出预测错误的标签，预测错误的标签被认为是类噪音
        label_errors_bool = ~(labels_predicted == s);
        label_errors_idx = np.arange(len(label_errors_bool))[label_errors_bool];
#         array_lateralMargin = self.FUNCTION_GetLateralMargin(label_errors_bool,label_errors_idx)
#         array_probability_errors = psx[label_errors_bool].max(axis=1)
#         self.FUNCTION_Vote(array_lateralMargin,array_probability,label_errors_idx,i_classifier)
        
        #获得误标签的概率和对应的方差
        array_probability_errors = np.zeros((len_instances), dtype=float);
#         array_probability_errors_std = np.zeros((len_instances), dtype=float);
#         array_margin_errors = np.zeros((len_instances), dtype=float);
#         array_margin_errors_std = np.zeros((len_instances), dtype=float);
        
        array_probability_errors[label_errors_idx] = psx[label_errors_idx].max(axis=1);
#         array_margin_errors[label_errors_idx] = margin_all_avg[label_errors_idx];
#         array_margin_errors_std[label_errors_idx] = std_margin[label_errors_idx];
        
#         for i_temp in range(len(label_errors_idx)):
#             i_errors = label_errors_idx[i_temp];
#             if labels_predicted[i_errors] == 1:
#                 array_probability_errors_std[i_errors] = std_1[i_errors];
#             else:#elif labels_predicted[i_errors] == 0:
#                 array_probability_errors_std[i_errors] = std_0[i_errors];
        
#         ED_probability_errors = array_probability_errors/array_probability_errors_std;
#         ED_margin_errors = array_margin_errors/array_margin_errors_std;
        
#         array_margin_errors = array_margin_errors.reshape(array_margin_errors.shape[0],1)
#         array_margin_errors_std = array_margin_errors_std.reshape(array_margin_errors_std.shape[0],1)
#         ED_margin_errors = ED_margin_errors.reshape(ED_margin_errors.shape[0],1)
        array_probability_errors = array_probability_errors.reshape(array_probability_errors.shape[0],1)
#         array_probability_errors_std = array_probability_errors_std.reshape(array_probability_errors_std.shape[0],1)
#         ED_probability_errors = ED_probability_errors.reshape(ED_probability_errors.shape[0],1)
        
#         result_matrix_votingMeasures = np.concatenate((array_probability_errors,array_probability_errors_std,ED_probability_errors),axis=1);
#         return result_matrix_votingMeasures;
        return array_probability_errors;
    #===end===#
    
#     #===Ensemble learning===#
#     def FUNCTION_EnsembleLearning_fromPredictionProbability(self,array_probability_allClassfiers,s):
#         print()
#     #===end===#
    
    
    
    
    