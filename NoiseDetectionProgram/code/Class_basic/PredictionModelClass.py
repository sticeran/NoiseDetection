'''
Created on 2020年10月24日

@author: njdx
'''

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier#利用邻近点方式训练数据
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
# from sklearn.metrics import f1_score, matthews_corrcoef, precision_score
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, classification_report
from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import ADASYN
# from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
import xgboost

# from MyFeatureSelection import CFS
# from sklearn.feature_selection import RFECV

# from collections import Counter
import Class_basic.RebalancingClass as RTC
import Class_basic.FeatureSelectionClass as FSC



class PredictionModelClass(object):
    
    classifiers = [];
    # 创建数据再平衡类
    Class_Rebalancing = RTC.RebalancingClass();
    # 创建特征选择类
    Class_FS = FSC.FeatureSelectionClass();
#     class_CFS = [];
    
    def __init__(self,list_names_classifiers):
        self.classifiers = [];
#         self.class_CFS = CFS.CFS();
        for i_name in list_names_classifiers:
            if "K-NN" in i_name:
                startIdx = i_name.find('=')
                endIdx = i_name.find(')')
                intNum = int(i_name[startIdx+1:endIdx])
                self.classifiers.append(KNeighborsClassifier(n_neighbors=intNum));
            elif i_name == "Naive Bayes":
                self.classifiers.append(GaussianNB());
            elif i_name == "Gaussian Process":
                self.classifiers.append(GaussianProcessClassifier(random_state=0, kernel = 1.0 * RBF(1.0)));
            elif i_name == "Logistic":
                self.classifiers.append(LogisticRegression(random_state=0));#就用默认
            elif i_name == "RBF SVM":
                self.classifiers.append(SVC(random_state=0, gamma=2, C=1, probability=True));
            elif i_name == "Decision Tree":
                self.classifiers.append(DecisionTreeClassifier(random_state=0));
            elif i_name == "Random Forest":
#                 self.classifiers.append(RandomForestClassifier(random_state=0,n_estimators=500));
                self.classifiers.append(RandomForestClassifier(random_state=0));
            elif i_name == "AdaBoost":
#                 self.classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=0),n_estimators=500,random_state=0));
                self.classifiers.append(AdaBoostClassifier(random_state=0));
            elif i_name == "Neural Net":
                self.classifiers.append(MLPClassifier(random_state=0,learning_rate_init=0.01,max_iter=1000));#alpha=1,
            elif i_name == "GBDT":
                self.classifiers.append(GradientBoostingClassifier(random_state=0));
            elif i_name == "xgboost":
                self.classifiers.append(xgboost.XGBClassifier(random_state=0));
            elif i_name == "QDA":
                self.classifiers.append(QuadraticDiscriminantAnalysis());
            elif i_name == "Random":
                self.classifiers.append("Random");
    
    #===This function gets prediction probability===#
    def FUNCTION_CaculatePredictionProbability_main(self,X_train,y_train,X_test,y_test):
        #存所有分类器的性能
#         list_weights_precision = []
#         list_weights_recall = []
#         list_weights_f1 = []
        #判断少数类的数量是否小于n_fold
        y_0 = y_train[y_train == 0]
        y_1 = y_train[y_train == 1]
        len_y_0 = len(y_0)
        len_y_1 = len(y_1)
        
        for i_classifier in range(len(self.classifiers)):
            proba_result = 0;
            pred_y = 0;
            if len_y_1 == 0:
                #生成n*2的表
                ndarray1 = np.array([1, 0])
                proba_result = np.resize(ndarray1, (X_test.shape[0], *ndarray1.shape))
                pred_y = np.zeros(X_test.shape[0])
            elif len_y_0 == 0:
                #生成n*2的表
                ndarray1 = np.array([0, 1])
                proba_result = np.resize(ndarray1, (X_test.shape[0], *ndarray1.shape))
                pred_y = np.ones(X_test.shape[0])
            else:
                clf = self.classifiers[i_classifier];#引入训练方法
                print(clf)
    #             if clf == "Random":
    #                 indicator_precision,indicator_recall,indicator_f1 = self.FUNCTION_randomModelExpectation(y_test);
    #                 proba_result = np.zeros(shape=(y_test.shape[0],2))
    #             else:
    #             # 对训练数据集作平衡处理 
    #             over_samples_X,over_samples_y = rebalanceTechnique.fit_sample(X_train, y_train)
                
                ###训练数据###
                clf.fit(X_train,y_train)#进行填充测试数据进行训练
    #             clf.fit(over_samples_X,over_samples_y)
                
                ###预测数据###
                # 返回预测属于某标签的概率
                proba_result = clf.predict_proba(X_test)
                pred_y = clf.predict(X_test)
                
    #             #验证阈值是不是0.5使用
    #             labels_predicted = np.zeros((len(y_test)), dtype=int);
    #             idx_predicted_1 = np.where(proba_result[:,1] > 0.5)[0];
    #             labels_predicted[idx_predicted_1] = 1;
    #             predictedY_allFolds = labels_predicted
                
                # 模型评估报告
    #             print(classification_report(y_test, pred_y))
    #             print(f1_score(y_test, pred_y))
    #             print("MCC: ",matthews_corrcoef(y, predictedY_allFolds))
    #             indicator_MCC = matthews_corrcoef(y, predictedY_allFolds);
    
    #             indicator_precision = precision_score(y_test, pred_y);
    #             indicator_recall = recall_score(y_test, pred_y);
    #             indicator_f1 = f1_score(y_test, pred_y);
    #             indicator_precision = round(indicator_precision,3);
    #             indicator_recall = round(indicator_recall,3);
    #             indicator_f1 = round(indicator_f1,3);
                
    #             list_weights_precision.append(indicator_precision);
    #             list_weights_recall.append(indicator_recall);
    #             list_weights_f1.append(indicator_f1);
        
            psx = proba_result.copy()
            str_array_psx_oneClassifier = psx.astype(str)
            labels = pred_y.reshape(-1,1)
            
            if i_classifier == 0:
                str_array_psx_allClassifiers = str_array_psx_oneClassifier;
                array_labels = labels;
                
            else:
                str_array_psx_allClassifiers = np.concatenate((str_array_psx_allClassifiers,str_array_psx_oneClassifier),axis=1)
                array_labels = np.concatenate((array_labels,labels),axis=1)
        
#         return str_array_psx_allClassifiers,list_weights_precision+list_weights_recall+list_weights_f1;
        return str_array_psx_allClassifiers,array_labels;
    #===end===#
    
#     #===This function gets prediction probability===#
#     def FUNCTION_CaculatePredictionProbability_crossValidation(self,X,y,n_fold,specifiedSeed):
#         #存所有分类器的权重
#         list_weights = []
#         #判断少数类的数量是否大于等于10
#         y_0 = y[y == 0]
#         y_1 = y[y == 1]
#         len_y_0 = len(y_0)
#         len_y_1 = len(y_1)
#         if len_y_1 < len_y_0:
#             minority_class = len_y_1;
#         else:
#             minority_class = len_y_0;
#         if minority_class < n_fold:
#             n_fold = minority_class;
#         
#         #标签类别数量
# #         num_categories = len(np.unique(y))
#         #十折分层抽样
#         sfolder = StratifiedKFold(n_splits=n_fold,shuffle=True,random_state=specifiedSeed)
#         #选取再平衡技术
# #         rebalanceTechnique = SMOTE(random_state=0)
# #         rebalanceTechnique = ADASYN(random_state=0)
# #         rebalanceTechnique = SMOTEENN(random_state=0)
# 
#         for i_classifier in range(len(self.classifiers)):
#             clf = self.classifiers[i_classifier];#引入训练方法
#             print(clf)
#             
#             # out-of-sample predicted probabilities using cross-validation
# #             proba_allFolds = np.empty([0,num_categories]);
#             test_index_allFolds = np.empty([0,0]);
#             predictedY_allFolds = np.empty([0,0]);
#             for train_index, test_index in sfolder.split(X,y):
#                 #print('Train: %s | test: %s' % (train_index, test_index))
#                 test_index_allFolds = np.append(test_index_allFolds,test_index)#记录每一折测试集的下标
#                 
#                 X_train, X_test = X[train_index], X[test_index]#训练集和测试集的特征
#                 y_train, y_test = y[train_index], y[test_index]#训练集的标签
#                 
#                 y_train_1 = y_train[y_train == 1]
#                 len_y_train_1 = len(y_train_1)
#                 # 对训练数据集作平衡处理 
#                 if len_y_train_1 < 6:
#                     rebalanceTechnique1 = SMOTE(random_state=0, k_neighbors = len_y_train_1-1,)
#                     rebalanceTechnique = SMOTETomek(random_state=0, smote = rebalanceTechnique1)
#                 else:
#                     rebalanceTechnique = SMOTETomek(random_state=0)
#                 over_samples_X,over_samples_y = rebalanceTechnique.fit_sample(X_train, y_train)
#                 
# #                 # CFS特征选择
# #                 col_CFS = self.class_CFS.fit_transform_col(over_samples_X, over_samples_y);
# #                 over_samples_X = over_samples_X[:, col_CFS]
# #                 X_test = X_test[:, col_CFS]#测试集和训练集保留相同的特征
#                 
#                 ###训练数据###
# #                 clf.fit(X_train,y_train)#进行填充测试数据进行训练
#                 clf.fit(over_samples_X,over_samples_y)#进行填充测试数据进行训练
#                 
#                 ###预测数据###
#                 # 返回预测属于某标签的概率
# #                 result = clf.predict_proba(X_test)
# #                 proba_allFolds = np.append(proba_allFolds,result,axis=0)
#                 predY = clf.predict(X_test)
#                 predictedY_allFolds = np.append(predictedY_allFolds,predY)
#                 
#                 ###评估###
#                 print(precision_score(y_test, predY))
#                 print(recall_score(y_test, predY))
#                 print(f1_score(y_test, predY))
#                 print(matthews_corrcoef(y_test, predY))
#                 
#             
#             #按照测试集原始下标排序
#             idx_originalOrder = test_index_allFolds.argsort()
# #             y_originalOrder = test_index_allFolds[idx_originalOrder]
# #             proba_allFolds = proba_allFolds[idx_originalOrder]
#             predictedY_allFolds = predictedY_allFolds[idx_originalOrder]
#             
# #             #验证用
# #             labels_predicted = np.zeros((len(y)), dtype=int);
# #             idx_predicted_1 = np.where(proba_allFolds[:,1] > 0.5)[0];
# #             labels_predicted[idx_predicted_1] = 1;
# #             predictedY_allFolds = labels_predicted
#             
# #             # 模型的预测准确率
# #             print(metrics.accuracy_score(y, predictedY_allFolds))
#             # 模型评估报告
#             print(classification_report(y, predictedY_allFolds))
#             print(f1_score(y, predictedY_allFolds))
# #             indicator = f1_score(y, predictedY_allFolds);            
#             print("MCC: ",matthews_corrcoef(y, predictedY_allFolds))
#             indicator = matthews_corrcoef(y, predictedY_allFolds);
# 
#             indicator = round(indicator,2);
#             list_weights.append(indicator);
#             
#             
#         return list_weights;
#     #===end===#
    
    #===This function gets prediction probability from random model===#
    def FUNCTION_randomModelExpectation(self,y):
#         y_0 = y[y == 0]
        y_1 = y[y == 1]
#         len_y_0 = len(y_0)
        len_y_1 = len(y_1)
        len_y = len(y)
#         E_precision_0 = len_y_0/len_y
        if len_y != 0:
            E_precision_1 = len_y_1/len_y
        else:
            E_precision_1 = 0
        E_recall = 0.5#对于阈值是0.5的随机模型而言
        E_f1 = 2*E_precision_1*E_recall/(E_precision_1+E_recall)
        
#         TP=0.5*len_y_1
#         FP=0.5*len_y-TP
#         TN=0.5*len_y_0
#         FN=0.5*len_y-TN
        
#         if ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) == 0:
#             E_MCC = np.nan
#         else:
#             E_MCC = (TP*TN-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5);
        
        
        return E_precision_1,E_recall,E_f1;
    #===end===#
    
    #===This function gets prediction probability===#
    def FUNCTION_CaculatePredictionProbability_crossValidation_allIndicators(self,df,features,colLabel,n_fold=5,specifiedSeed=1):
        #特征值
        X = df[features].values;
        #标签
        y = df[colLabel].values;
        
        #存所有分类器的权重
        list_weights_precision = []
        list_weights_recall = []
        list_weights_f1 = []
        #判断少数类的数量是否大于等于10
        y_0 = y[y == 0]
        y_1 = y[y == 1]
        len_y_0 = len(y_0)
        len_y_1 = len(y_1)
        if len_y_1 < len_y_0:
            minority_class = len_y_1;
        else:
            minority_class = len_y_0;
        if minority_class < n_fold:
            n_fold = minority_class;
        
        #标签类别数量
#         num_categories = len(np.unique(y))
        #十折分层抽样
        sfolder = StratifiedKFold(n_splits=n_fold,shuffle=True,random_state=specifiedSeed)
        #选取再平衡技术
#         rebalanceTechnique = SMOTE(random_state=0)
#         rebalanceTechnique = ADASYN(random_state=0)
#         rebalanceTechnique = SMOTEENN(random_state=0)

        for i_classifier in range(len(self.classifiers)):
            clf = self.classifiers[i_classifier];#引入训练方法
#             print(clf)
            
            if clf == "Random":
                indicator_precision,indicator_recall,indicator_f1 = self.FUNCTION_randomModelExpectation(y);
            else:
                # out-of-sample predicted probabilities using cross-validation
    #             proba_allFolds = np.empty([0,num_categories]);
                test_index_allFolds = np.empty([0,0]);
                predictedY_allFolds = np.empty([0,0]);
                for train_index, test_index in sfolder.split(X,y):
                    #print('Train: %s | test: %s' % (train_index, test_index))
                    test_index_allFolds = np.append(test_index_allFolds,test_index)#记录每一折测试集的下标
                    
                    X_train, X_test = X[train_index], X[test_index]#训练集和测试集的特征
                    y_train = y[train_index]#训练集的标签
                    
                    ###训练数据###
                    clf.fit(X_train,y_train)#进行填充测试数据进行训练
    #                 clf.fit(over_samples_X,over_samples_y)#进行填充测试数据进行训练
                    
                    ###预测数据###
                    # 返回预测属于某标签的概率
    #                 result = clf.predict_proba(X_test)
    #                 proba_allFolds = np.append(proba_allFolds,result,axis=0)
                    predY = clf.predict(X_test)
                    predictedY_allFolds = np.append(predictedY_allFolds,predY)
                
                #按照测试集原始下标排序
                idx_originalOrder = test_index_allFolds.argsort()
    #             y_originalOrder = test_index_allFolds[idx_originalOrder]
    #             proba_allFolds = proba_allFolds[idx_originalOrder]
                predictedY_allFolds = predictedY_allFolds[idx_originalOrder]
                
    #             #验证用
    #             labels_predicted = np.zeros((len(y)), dtype=int);
    #             idx_predicted_1 = np.where(proba_allFolds[:,1] > 0.5)[0];
    #             labels_predicted[idx_predicted_1] = 1;
    #             predictedY_allFolds = labels_predicted
                
    #             # 模型的预测准确率
    #             print(metrics.accuracy_score(y, predictedY_allFolds))
                # 模型评估报告
    #             print(classification_report(y, predictedY_allFolds))
    #             print(f1_score(y, predictedY_allFolds))
                indicator_f1 = f1_score(y, predictedY_allFolds);
    #             print("MCC: ",matthews_corrcoef(y, predictedY_allFolds))
#                 indicator_MCC = matthews_corrcoef(y, predictedY_allFolds);
                indicator_precision = precision_score(y, predictedY_allFolds);
                indicator_recall = recall_score(y, predictedY_allFolds);
            
            indicator_precision = round(indicator_precision,3);
            indicator_recall = round(indicator_recall,3);
            indicator_f1 = round(indicator_f1,3);
#             indicator_MCC = round(indicator_MCC,3);
            
            list_weights_precision.append(indicator_precision);
            list_weights_recall.append(indicator_recall);
            list_weights_f1.append(indicator_f1);
#             list_weights_MCC.append(indicator_MCC);
        
        return list_weights_precision+list_weights_recall+list_weights_f1
    #===end===#
    
#     #===This function gets prediction probability===#
#     def FUNCTION_CaculatePredictionProbability_crossValidation_specifiedIndicator(self,df,features,specifiedIndicator,col_Label,n_fold=5,specifiedSeed=1):
#         #特征值
#         X = df[features].values;
#         #标签
#         y = df[col_Label].values;
#         
#         #存所有分类器的权重
#         list_weights_f1 = []
# #         list_weights_MCC = []
#         list_weights_precision = []
#         #判断少数类的数量是否大于等于10
#         y_0 = y[y == 0]
#         y_1 = y[y == 1]
#         len_y_0 = len(y_0)
#         len_y_1 = len(y_1)
#         if len_y_1 < len_y_0:
#             minority_class = len_y_1;
#         else:
#             minority_class = len_y_0;
#         if minority_class < n_fold:
#             n_fold = minority_class;
#         
#         #标签类别数量
# #         num_categories = len(np.unique(y))
#         #十折分层抽样
#         sfolder = StratifiedKFold(n_splits=n_fold,shuffle=True,random_state=specifiedSeed)
#         #选取再平衡技术
# #         rebalanceTechnique = SMOTE(random_state=0)
# #         rebalanceTechnique = ADASYN(random_state=0)
# #         rebalanceTechnique = SMOTEENN(random_state=0)
# 
#         for i_classifier in range(len(self.classifiers)):
#             clf = self.classifiers[i_classifier];#引入训练方法
# #             print(clf)
#             
#             if clf == "Random":
#                 indicator_precision,indicator_recall,indicator_f1 = self.FUNCTION_randomModelExpectation(y);
#             else:
#                 # out-of-sample predicted probabilities using cross-validation
#     #             proba_allFolds = np.empty([0,num_categories]);
#                 test_index_allFolds = np.empty([0,0]);
#                 predictedY_allFolds = np.empty([0,0]);
#                 for train_index, test_index in sfolder.split(X,y):
#                     #print('Train: %s | test: %s' % (train_index, test_index))
#                     test_index_allFolds = np.append(test_index_allFolds,test_index)#记录每一折测试集的下标
#                     
#                     X_train, X_test = X[train_index], X[test_index]#训练集和测试集的特征
#                     y_train = y[train_index]#训练集的标签
#                     
#                     ###训练数据###
#                     clf.fit(X_train,y_train)#进行填充测试数据进行训练
#     #                 clf.fit(over_samples_X,over_samples_y)#进行填充测试数据进行训练
#                     
#                     ###预测数据###
#                     # 返回预测属于某标签的概率
#     #                 result = clf.predict_proba(X_test)
#     #                 proba_allFolds = np.append(proba_allFolds,result,axis=0)
#                     predY = clf.predict(X_test)
#                     predictedY_allFolds = np.append(predictedY_allFolds,predY)
#                 
#                 #按照测试集原始下标排序
#                 idx_originalOrder = test_index_allFolds.argsort()
#     #             y_originalOrder = test_index_allFolds[idx_originalOrder]
#     #             proba_allFolds = proba_allFolds[idx_originalOrder]
#                 predictedY_allFolds = predictedY_allFolds[idx_originalOrder]
#                 
#     #             #验证用
#     #             labels_predicted = np.zeros((len(y)), dtype=int);
#     #             idx_predicted_1 = np.where(proba_allFolds[:,1] > 0.5)[0];
#     #             labels_predicted[idx_predicted_1] = 1;
#     #             predictedY_allFolds = labels_predicted
#                 
#     #             # 模型的预测准确率
#     #             print(metrics.accuracy_score(y, predictedY_allFolds))
#                 # 模型评估报告
#     #             print(classification_report(y, predictedY_allFolds))
#     #             print(f1_score(y, predictedY_allFolds))
#                 indicator_f1 = f1_score(y, predictedY_allFolds);
#     #             print("MCC: ",matthews_corrcoef(y, predictedY_allFolds))
# #                 indicator_MCC = matthews_corrcoef(y, predictedY_allFolds);
#                 indicator_precision = precision_score(y, predictedY_allFolds);
#             
#             indicator_precision = round(indicator_precision,3);
#             indicator_f1 = round(indicator_f1,3);
# #             indicator_MCC = round(indicator_MCC,3);
#             
#             list_weights_f1.append(indicator_f1);
# #             list_weights_MCC.append(indicator_MCC);
#             list_weights_precision.append(indicator_precision);
#         
#         if len(self.classifiers) > 1:
#             if specifiedIndicator == "F1":
#                 return list_weights_f1
#             elif specifiedIndicator == "precision":
#                 return list_weights_precision
#             else:
#                 return ;
#         else:
#             if specifiedIndicator == "F1":
#                 return list_weights_f1[0]#只有一个分类器，只返回一个值
#             elif specifiedIndicator == "precision":
#                 return list_weights_precision[0]
#             else:
#                 return ;
#     #===end===#
    
    #===This function gets prediction probability===#
    def FUNCTION_CaculatePredictionProbability_crossValidation(self,X,y,n_fold,specifiedSeed):
        #存所有分类器的权重
        list_weights_precision = []
        list_weights_recall = []
        list_weights_f1 = []
        list_weights_f1_ED = []
        #判断少数类的数量是否大于等于10
        y_0 = y[y == 0]
        y_1 = y[y == 1]
        len_y_0 = len(y_0)
        len_y_1 = len(y_1)
        if len_y_1 < len_y_0:
            minority_class = len_y_1;
        else:
            minority_class = len_y_0;
        if minority_class < n_fold:
            n_fold = minority_class;
        
        #标签类别数量
#         num_categories = len(np.unique(y))
        #十折分层抽样
        sfolder = StratifiedKFold(n_splits=n_fold,shuffle=True,random_state=specifiedSeed)
        #选取再平衡技术
#         rebalanceTechnique = SMOTE(random_state=0)
#         rebalanceTechnique = ADASYN(random_state=0)
#         rebalanceTechnique = SMOTEENN(random_state=0)

        for i_classifier in range(len(self.classifiers)):
            clf = self.classifiers[i_classifier];#引入训练方法
#             print(clf)
            
            if clf == "Random":
                indicator_precision,indicator_recall,indicator_f1 = self.FUNCTION_randomModelExpectation(y);
            else:
                # out-of-sample predicted probabilities using cross-validation
    #             proba_allFolds = np.empty([0,num_categories]);
                test_index_allFolds = np.empty([0,0]);
                f1_tenFolds = [];
                precision_tenFolds = [];
                recall_tenFolds = [];
                for train_index, test_index in sfolder.split(X,y):
                    #print('Train: %s | test: %s' % (train_index, test_index))
                    test_index_allFolds = np.append(test_index_allFolds,test_index)#记录每一折测试集的下标
                    
                    X_train, X_test = X[train_index], X[test_index]#训练集和测试集的特征
                    y_train, y_test = y[train_index], y[test_index]#训练集的标签
                    
    #                 y_train_1 = y_train[y_train == 1]
    #                 len_y_train_1 = len(y_train_1)
                    
                    # 对训练数据集作平衡处理 
                    over_samples_X,over_samples_y = self.Class_Rebalancing.FUNCTION_Rebalancing_SMOTETomek(X_train, y_train)
    #                 if len_y_train_1 < 6:
    #                     rebalanceTechnique1 = SMOTE(random_state=0, k_neighbors = len_y_train_1-1,)
    #                     rebalanceTechnique = SMOTETomek(random_state=0, smote = rebalanceTechnique1)
    #                 else:
    #                     rebalanceTechnique = SMOTETomek(random_state=0)
    #                 over_samples_X,over_samples_y = rebalanceTechnique.fit_sample(X_train, y_train)
                    
    #                 # CFS特征选择
    #                 col_CFS = self.class_CFS.fit_transform_col(over_samples_X, over_samples_y);
    #                 over_samples_X = over_samples_X[:, col_CFS]
    #                 X_test = X_test[:, col_CFS]#测试集和训练集保留相同的特征
                    
                    ###训练数据###
#                     clf.fit(X_train,y_train)#进行填充测试数据进行训练
                    clf.fit(over_samples_X,over_samples_y)#进行填充测试数据进行训练
                    
                    ###预测数据###
                    # 返回预测属于某标签的概率
    #                 result = clf.predict_proba(X_test)
    #                 proba_allFolds = np.append(proba_allFolds,result,axis=0)
                    predY = clf.predict(X_test)
                    
                    # 模型评估报告
                    f1_temp = f1_score(y_test, predY);
                    precision_temp = precision_score(y_test, predY);
                    recall_temp = recall_score(y_test, predY);
                    f1_tenFolds.append(f1_temp)
                    precision_tenFolds.append(precision_temp)
                    recall_tenFolds.append(recall_temp)
                
                #求10折指标平均
                indicator_f1 = np.mean(f1_tenFolds)
                indicator_f1_std = np.std(f1_tenFolds,ddof=1)
                indicator_precision = np.mean(precision_tenFolds)
                indicator_recall = np.mean(recall_tenFolds)
                if indicator_f1_std:
                    indicator_f1_ED = indicator_f1/indicator_f1_std
                else:
                    indicator_f1_ED = 5 * indicator_f1
            
            indicator_precision = round(indicator_precision,3);
            indicator_recall = round(indicator_recall,3);
            indicator_f1 = round(indicator_f1,3);
            indicator_f1_ED = round(indicator_f1_ED,3);
            
            list_weights_precision.append(indicator_precision);
            list_weights_recall.append(indicator_recall);
            list_weights_f1.append(indicator_f1);
            list_weights_f1_ED.append(indicator_f1_ED);
            
        return list_weights_precision+list_weights_recall+list_weights_f1+list_weights_f1_ED
    #===end===#
    
    #===This function gets prediction probability===#
    def FUNCTION_CaculatePredictionProbability_crossValidation_noise(self,df,col_Label,n_fold=5,specifiedSeed=1):
        #存所有分类器的性能
        list_weights_precision = []
        list_weights_recall = []
        list_weights_f1 = []
        
        colNames = list(df)
        colNames.remove(col_Label)
        X = df[colNames].values;
        y = df[col_Label].values;
        
        #判断少数类的数量是否小于n_fold
        y_0 = y[y == 0]
        y_1 = y[y == 1]
        len_y_0 = len(y_0)
        len_y_1 = len(y_1)
        
        if len_y_1 < len_y_0:
            minority_class = len_y_1;
        else:
            minority_class = len_y_0;
        if minority_class < n_fold:
            n_fold = minority_class;
        
        #标签类别数量
        num_categories = len(np.unique(y))
        #n折分层抽样
        if n_fold <= 1:
            sfolder = np.nan;
        else:
            sfolder = StratifiedKFold(n_splits=n_fold,shuffle=True,random_state=specifiedSeed)
        #选取再平衡技术
#         rebalanceTechnique = SMOTE(random_state=0)
#         rebalanceTechnique = ADASYN(random_state=0)
#         rebalanceTechnique = SMOTEENN(random_state=0)

        for i_classifier in range(len(self.classifiers)):
            clf = self.classifiers[i_classifier];#引入训练方法
            print(clf)
            if clf == "Random":
                indicator_precision,indicator_recall,indicator_f1 = self.FUNCTION_randomModelExpectation(y);
                proba_allFolds = np.zeros(shape=(y.shape[0],2))
            else:
                if pd.isnull(sfolder):
                    indicator_precision,indicator_recall,indicator_f1 = 0,0,0
                    #生成n*2的表
                    ndarray1 = np.array([1, 0])
                    proba_allFolds = np.resize(ndarray1, (y.shape[0], *ndarray1.shape))
                else:
                    # out-of-sample predicted probabilities using cross-validation
                    proba_allFolds = np.empty([0,num_categories]);
                    test_index_allFolds = np.empty([0,0]);
                    y_test_allFolds = np.empty([0,0]);
                    predictedY_allFolds = np.empty([0,0]);
                    for train_index, test_index in sfolder.split(X,y):
                        #print('Train: %s | test: %s' % (train_index, test_index))
                        test_index_allFolds = np.append(test_index_allFolds,test_index)#记录每一折测试集的下标
                        
                        X_train, X_test = X[train_index], X[test_index]#训练集和测试集的特征
                        y_train, y_test = y[train_index], y[test_index]#训练集的标签
                        
                        # 按照特征，标签顺序合并
                        list_colNames = df.columns.tolist();
                        y_train = y_train.reshape(y_train.shape[0],1)#对一维数组进行转置需要借助reshape来完成
                        array_merge = np.concatenate((X_train,y_train),axis=1);
                        df_train = pd.DataFrame(array_merge, columns = list_colNames);
                        y_test = y_test.reshape(y_test.shape[0],1)#对一维数组进行转置需要借助reshape来完成
                        array_merge = np.concatenate((X_test,y_test),axis=1);
                        df_test = pd.DataFrame(array_merge, columns = list_colNames);
                        
                        # 判断是否使用SMOTE再平衡技术
                        df_train_rebalance = self.Class_Rebalancing.FUNCTION_Rebalancing(df_train,col_Label)
    #                     # GainRatio特征选择方法
    #                     df_train_fs = self.Class_FS.GainRatio_from_R(df_train_rebalance, col_Label)
    #                     colNames = list(df_train_fs)
    #                     df_test_fs = df_test[colNames]#测试集根据特征选择后的训练集保留列
                        
                        colNames = list(df_train_rebalance)
                        colNames.remove(col_Label)
                        X_train = df_train_rebalance[colNames].values;
                        y_train = df_train_rebalance[col_Label].values;
                        X_test = df_test[colNames].values;
                        y_test = df_test[col_Label].values;
                        
                        ###训练数据###
                        clf.fit(X_train,y_train)#进行填充测试数据进行训练
        #                 clf.fit(over_samples_X,over_samples_y)#进行填充测试数据进行训练
                        
                        ###预测数据###
                        # 返回预测属于某标签的概率
                        result = clf.predict_proba(X_test)
                        proba_allFolds = np.append(proba_allFolds,result,axis=0)
                        predY = clf.predict(X_test)
                        y_test_allFolds = np.append(y_test_allFolds,y_test)
                        predictedY_allFolds = np.append(predictedY_allFolds,predY)
                
                    # 模型评估报告
        #             print(classification_report(y_test_allFolds, predictedY_allFolds))
                    indicator_precision = precision_score(y_test_allFolds, predictedY_allFolds);
                    indicator_recall = recall_score(y_test_allFolds, predictedY_allFolds);
                    indicator_f1 = f1_score(y_test_allFolds, predictedY_allFolds);
                    #按照测试集原始下标排序
                    idx_originalOrder = test_index_allFolds.argsort()
        #             y_originalOrder = test_index_allFolds[idx_originalOrder]
                    proba_allFolds = proba_allFolds[idx_originalOrder]
        #             predictedY_allFolds = predictedY_allFolds[idx_originalOrder]
#             indicator_precision = round(indicator_precision,3);
#             indicator_recall = round(indicator_recall,3);
#             indicator_f1 = round(indicator_f1,3);
            
            list_weights_precision.append(indicator_precision);
            list_weights_recall.append(indicator_recall);
            list_weights_f1.append(indicator_f1);
            

            
            psx = proba_allFolds.copy()
            
            str_array_psx_oneClassifier = psx.astype(str)
            
            if i_classifier == 0:
                str_array_psx_allClassifiers = str_array_psx_oneClassifier;
            else:
                str_array_psx_allClassifiers = np.concatenate((str_array_psx_allClassifiers,str_array_psx_oneClassifier),axis=1)
        
        return str_array_psx_allClassifiers,list_weights_precision+list_weights_recall+list_weights_f1;
    #===end===#
    
    #===This function gets prediction probability===#
    def FUNCTION_CaculatePredictionProbability_crossValidation_noise_fs(self,df,col_Label,n_fold=5,specifiedSeed=1):
        #存所有分类器的性能
        list_weights_precision = []
        list_weights_recall = []
        list_weights_f1 = []
        
        colNames = list(df)
        colNames.remove(col_Label)
        X = df[colNames].values;
        y = df[col_Label].values;
        
        #判断少数类的数量是否大于等于10
        y_0 = y[y == 0]
        y_1 = y[y == 1]
        len_y_0 = len(y_0)
        len_y_1 = len(y_1)
        
        if len_y_1 < len_y_0:
            minority_class = len_y_1;
        else:
            minority_class = len_y_0;
        if minority_class < n_fold:
            n_fold = minority_class;
        
        #标签类别数量
        num_categories = len(np.unique(y))
        #十折分层抽样
        sfolder = StratifiedKFold(n_splits=n_fold,shuffle=True,random_state=specifiedSeed)
        #选取再平衡技术
#         rebalanceTechnique = SMOTE(random_state=0)
#         rebalanceTechnique = ADASYN(random_state=0)
#         rebalanceTechnique = SMOTEENN(random_state=0)

        for i_classifier in range(len(self.classifiers)):
            clf = self.classifiers[i_classifier];#引入训练方法
            print(clf)
            if clf == "Random":
                indicator_precision,indicator_recall,indicator_f1 = self.FUNCTION_randomModelExpectation(y);
                test_index_allFolds = np.zeros(y.shape[0])
                proba_allFolds = np.zeros(shape=(y.shape[0],2))
            else:
                # out-of-sample predicted probabilities using cross-validation
                proba_allFolds = np.empty([0,num_categories]);
                test_index_allFolds = np.empty([0,0]);
                y_test_allFolds = np.empty([0,0]);
                predictedY_allFolds = np.empty([0,0]);
                for train_index, test_index in sfolder.split(X,y):
                    #print('Train: %s | test: %s' % (train_index, test_index))
                    test_index_allFolds = np.append(test_index_allFolds,test_index)#记录每一折测试集的下标
                    
                    X_train, X_test = X[train_index], X[test_index]#训练集和测试集的特征
                    y_train, y_test = y[train_index], y[test_index]#训练集的标签
                    
    #                 y_train_0 = y_train[y_train == 0]
    #                 y_train_1 = y_train[y_train == 1]
    #                 len_y_train_0 = len(y_train_0)
    #                 len_y_train_1 = len(y_train_1)
    #                 if len_y_train_1 < len_y_train_0:
    #                     minority_class_train = len_y_train_1;
    #                 else:
    #                     minority_class_train = len_y_train_0;
    #                 # 对训练数据集作平衡处理 
    #                 if minority_class_train < 6:
    #                     rebalanceTechnique1 = SMOTE(random_state=0, k_neighbors = minority_class_train-1,)
    #                     rebalanceTechnique = SMOTETomek(random_state=0, smote = rebalanceTechnique1)
    #                 else:
    #                     rebalanceTechnique = SMOTETomek(random_state=0)
    #                 over_samples_X,over_samples_y = rebalanceTechnique.fit_sample(X_train, y_train)
                    
                    # 按照特征，标签顺序合并
                    list_colNames = df.columns.tolist();
                    y_train = y_train.reshape(y_train.shape[0],1)#对一维数组进行转置需要借助reshape来完成
                    array_merge = np.concatenate((X_train,y_train),axis=1);
                    df_train = pd.DataFrame(array_merge, columns = list_colNames);
                    y_test = y_test.reshape(y_test.shape[0],1)#对一维数组进行转置需要借助reshape来完成
                    array_merge = np.concatenate((X_test,y_test),axis=1);
                    df_test = pd.DataFrame(array_merge, columns = list_colNames);
                    
                    # SMOTETomek再平衡技术
                    df_train_rebalance = self.Class_Rebalancing.FUNCTION_Rebalancing(df_train,col_Label)
                    # GainRatio特征选择方法
                    df_train_fs = self.Class_FS.GainRatio_from_R(df_train_rebalance, col_Label)
                    colNames = list(df_train_fs)
                    df_test_fs = df_test[colNames]#测试集根据特征选择后的训练集保留列
                    
                    colNames.remove(col_Label)
                    X_train = df_train_fs[colNames].values;
                    y_train = df_train_fs[col_Label].values;
                    X_test = df_test_fs[colNames].values;
                    y_test = df_test_fs[col_Label].values;
                    
                    ###训练数据###
                    clf.fit(X_train,y_train)#进行填充测试数据进行训练
    #                 clf.fit(over_samples_X,over_samples_y)#进行填充测试数据进行训练
                    
                    ###预测数据###
                    # 返回预测属于某标签的概率
                    result = clf.predict_proba(X_test)
                    proba_allFolds = np.append(proba_allFolds,result,axis=0)
                    predY = clf.predict(X_test)
                    y_test_allFolds = np.append(y_test_allFolds,y_test)
                    predictedY_allFolds = np.append(predictedY_allFolds,predY)
            
                # 模型评估报告
    #             print(classification_report(y_test_allFolds, predictedY_allFolds))
                indicator_precision = precision_score(y_test_allFolds, predictedY_allFolds);
                indicator_recall = recall_score(y_test_allFolds, predictedY_allFolds);
                indicator_f1 = f1_score(y_test_allFolds, predictedY_allFolds);
            indicator_precision = round(indicator_precision,3);
            indicator_recall = round(indicator_recall,3);
            indicator_f1 = round(indicator_f1,3);
            
            list_weights_precision.append(indicator_precision);
            list_weights_recall.append(indicator_recall);
            list_weights_f1.append(indicator_f1);
            
            #按照测试集原始下标排序
            idx_originalOrder = test_index_allFolds.argsort()
#             y_originalOrder = test_index_allFolds[idx_originalOrder]
            proba_allFolds = proba_allFolds[idx_originalOrder]
#             predictedY_allFolds = predictedY_allFolds[idx_originalOrder]
            
            psx = proba_allFolds.copy()
            
            str_array_psx_oneClassifier = psx.astype(str)
            
            if i_classifier == 0:
                str_array_psx_allClassifiers = str_array_psx_oneClassifier;
            else:
                str_array_psx_allClassifiers = np.concatenate((str_array_psx_allClassifiers,str_array_psx_oneClassifier),axis=1)
        
        return str_array_psx_allClassifiers,list_weights_precision+list_weights_recall+list_weights_f1;
    #===end===#
    
    #===This function gets prediction probability===#
    def FUNCTION_CaculatePredictionProbability_crossValidation_self(self,X,y,n_fold=5,specifiedSeed=1):
        #判断少数类的数量是否大于等于10
        y_0 = y[y == 0]
        y_1 = y[y == 1]
        len_y_0 = len(y_0)
        len_y_1 = len(y_1)
        
        if len_y_1 < len_y_0:
            minority_class = len_y_1;
        else:
            minority_class = len_y_0;
        if minority_class < n_fold:
            n_fold = minority_class;
        
        #标签类别数量
        num_categories = len(np.unique(y))
        #十折分层抽样
        sfolder = StratifiedKFold(n_splits=n_fold,shuffle=True,random_state=specifiedSeed)
        #选取再平衡技术
#         rebalanceTechnique = SMOTE(random_state=0)
#         rebalanceTechnique = ADASYN(random_state=0)
#         rebalanceTechnique = SMOTEENN(random_state=0)

        for i_classifier in range(len(self.classifiers)):
            clf = self.classifiers[i_classifier];#引入训练方法
            print(clf)
            
            # out-of-sample predicted probabilities using cross-validation
            proba_allFolds = np.empty([0,num_categories]);
            test_index_allFolds = np.empty([0,0]);
            y_test_allFolds = np.empty([0,0]);
            predictedY_allFolds = np.empty([0,0]);
            for train_index, test_index in sfolder.split(X,y):
                #print('Train: %s | test: %s' % (train_index, test_index))
                test_index_allFolds = np.append(test_index_allFolds,test_index)#记录每一折测试集的下标
                
                X_train, X_test = X[train_index], X[test_index]#训练集和测试集的特征
                y_train, y_test = y[train_index], y[test_index]#训练集的标签
                
                y_train_0 = y_train[y_train == 0]
                y_train_1 = y_train[y_train == 1]
                len_y_train_0 = len(y_train_0)
                len_y_train_1 = len(y_train_1)
                if len_y_train_1 < len_y_train_0:
                    minority_class_train = len_y_train_1;
                else:
                    minority_class_train = len_y_train_0;
                # 对训练数据集作平衡处理 
                if minority_class_train < 6:
                    rebalanceTechnique1 = SMOTE(random_state=0, k_neighbors = minority_class_train-1,)
                    rebalanceTechnique = SMOTETomek(random_state=0, smote = rebalanceTechnique1)
                else:
                    rebalanceTechnique = SMOTETomek(random_state=0)
                over_samples_X,over_samples_y = rebalanceTechnique.fit_sample(X_train, y_train)
                
                #特征选择
                
                
                ###训练数据###
#                 clf.fit(X_train,y_train)#进行填充测试数据进行训练
                clf.fit(over_samples_X,over_samples_y)#进行填充测试数据进行训练
                
                ###预测数据###
                # 返回预测属于某标签的概率
                result = clf.predict_proba(X_test)
                proba_allFolds = np.append(proba_allFolds,result,axis=0)
                predY = clf.predict(X_test)
                y_test_allFolds = np.append(y_test_allFolds,y_test)
                predictedY_allFolds = np.append(predictedY_allFolds,predY)
            
            # 模型评估报告
            print(classification_report(y_test_allFolds, predictedY_allFolds))
            
            #按照测试集原始下标排序
            idx_originalOrder = test_index_allFolds.argsort()
#             y_originalOrder = test_index_allFolds[idx_originalOrder]
            proba_allFolds = proba_allFolds[idx_originalOrder]
#             predictedY_allFolds = predictedY_allFolds[idx_originalOrder]
            
            psx = proba_allFolds.copy()
            
            str_array_psx_oneClassifier = psx.astype(str)
            
            if i_classifier == 0:
                str_array_psx_allClassifiers = str_array_psx_oneClassifier;
            else:
                str_array_psx_allClassifiers = np.concatenate((str_array_psx_allClassifiers,str_array_psx_oneClassifier),axis=1)
        
        return str_array_psx_allClassifiers;
    #===end===#
    
    #===This function gets prediction probability===#
    def FUNCTION_CaculatePredictionProbability_bootstrap(self,X,y,specifiedSeed):
        # 设置随机种子
        np.random.seed(specifiedSeed);
        #存所有分类器的权重
        list_weights = []
        len_y = len(y)
        iter = 100  # @ReservedAssignment
        idx = np.arange(len_y)
        #选取再平衡技术
#         rebalanceTechnique = SMOTE(random_state=0)
#         rebalanceTechnique = ADASYN(random_state=0)
#         rebalanceTechnique = SMOTEENN(random_state=0)
        rebalanceTechnique = SMOTETomek(random_state=0)
        for i_classifier in range(len(self.classifiers)):
            clf = self.classifiers[i_classifier];#引入训练方法
            print(clf)
            
#             y_all = np.empty([0,0]);
#             predictedY_all = np.empty([0,0]);
            array_result = np.array([]);
            for i in range(iter):  # @UnusedVariable
                idx_train = resample(idx,n_samples=len_y,replace=1)
                idx_train = np.unique(idx_train)
                X_train = X[idx_train]
                y_train = y[idx_train]
                idx_diff = np.setdiff1d(idx, idx_train)
                X_test = X[idx_diff]
                y_test = y[idx_diff]
                
                # 对训练数据集作平衡处理 
                over_samples_X,over_samples_y = rebalanceTechnique.fit_sample(X_train, y_train)
                
                ###训练数据###
    #                 clf.fit(X_train,y_train)#进行填充测试数据进行训练
                clf.fit(over_samples_X,over_samples_y)#进行填充测试数据进行训练
                
                ###预测数据###
                # 返回预测属于某标签的概率
    #                 result = clf.predict_proba(X_test)
    #                 proba_allFolds = np.append(proba_allFolds,result,axis=0)
                predY = clf.predict(X_test)
#                 predictedY_all = np.append(predictedY_all,predY)
#                 y_all = np.append(y_all,y_test)
                
                result = matthews_corrcoef(y_test, predY);
                array_result = np.append(array_result,result)
            
            indicator = np.mean(array_result)
            indicator = round(indicator,2);
            list_weights.append(indicator);
        
        
        return list_weights;
    #===end===#
    
    #===This function gets prediction probability===#
    def FUNCTION_TrainClassifier(self,X_train,y_train):
        list_clf = [];
        y_train_1 = y_train[y_train == 1]
        len_y_train_1 = len(y_train_1)
        #选取再平衡技术
#         rebalanceTechnique = SMOTE(random_state=0)
#         rebalanceTechnique = ADASYN(random_state=0)
#         rebalanceTechnique = SMOTEENN(random_state=0)
        if len_y_train_1 < 6:
            rebalanceTechnique1 = SMOTE(random_state=0, k_neighbors=len_y_train_1-1,)
            rebalanceTechnique = SMOTETomek(random_state=0, smote = rebalanceTechnique1)
        else:
            rebalanceTechnique = SMOTETomek(random_state=0)
        for i_classifier in range(len(self.classifiers)):
            clf = self.classifiers[i_classifier];#引入训练方法
#             print(clf)
            
            # 对训练数据集作平衡处理 
            over_samples_X,over_samples_y = rebalanceTechnique.fit_sample(X_train, y_train)
            
            ###训练数据###
#             clf.fit(X_train,y_train)#进行填充测试数据进行训练
            clf.fit(over_samples_X,over_samples_y)
            
            list_clf.append(clf);
        
        return list_clf;
    #===end===#
    
    #===This function gets prediction probability===#
    def FUNCTION_Predict(self,list_clf,X_test):
        
        for i in range(len(list_clf)):
            clf = list_clf[i];
            print(clf)
            
            ###预测数据###
            # 返回预测属于某标签的概率
            proba_result = clf.predict_proba(X_test)
            
            psx = proba_result.copy()
            
            str_array_psx_oneClassifier = psx.astype(str)
            
            if i == 0:
                str_array_psx_allClassifiers = str_array_psx_oneClassifier;
            else:
                str_array_psx_allClassifiers = np.concatenate((str_array_psx_allClassifiers,str_array_psx_oneClassifier),axis=1)
        
        return str_array_psx_allClassifiers;
    #===end===#
    
    
    
    
    
    
    
    
    
    
    
    

