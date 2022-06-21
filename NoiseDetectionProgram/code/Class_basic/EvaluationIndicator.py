import numpy as np
# from sklearn import metrics

class EvaluationIndicator(object):
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
#     list_precision = [];
#     list_recall = [];
#     list_F1 = [];
    totalLen_benchmark = 0;
    
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
#         self.list_precision = [];
#         self.list_recall = [];
#         self.list_F1 = [];
        self.totalLen_benchmark = 0;
        
    ### 获得数据集中每个实例的代码行
    def setDataset(self,df):
        self.df_original = df;
    
    ### 设置计算P@k%时的总长度基准，为了公平对比
    def setTotalLenBenchmark(self,totalLen_benchmark):
        self.totalLen_benchmark = totalLen_benchmark;
    
    ### compute P@k%SLOC
    def ComputeP_k_percent_SLOC(self,list_k,label_errors_idx,actual_label_errors):
        #list_k = [0.05,0.10];
        list_result = [];
        SLOC_total = self.df_original['loc'].sum()
        mean_SLOC = self.df_original['loc'].mean();
        for i_k in list_k:
            SLOC_k = round(i_k * SLOC_total);
            #判断：按预测概率排序的误标签，加起来的行数是否大于SLOC_k
            sum_SLOC = 0;
            if len(label_errors_idx):
                for j in range(len(label_errors_idx)):
                    if sum_SLOC >= SLOC_k:
                        break;
                    index = label_errors_idx[j];
                    sum_SLOC += self.df_original.loc[index,"loc"];
                numIns = j;
                if sum_SLOC < SLOC_k:
                    while sum_SLOC < SLOC_k:
                        sum_SLOC += mean_SLOC;
                        numIns += 1;
                label_errors_idx_atK = label_errors_idx[:j];
                intersection = np.intersect1d(label_errors_idx_atK,actual_label_errors);
                len_intersection = len(intersection);
                if len_intersection:
                    P_k = len_intersection/numIns;
                else:
                    P_k = 0;
            else:
                P_k = 0;
            list_result.append(P_k);
        return list_result;
    
    ### compute P@k%
    def ComputeP_k_percent(self,list_k,label_errors_idx,actual_label_errors):
        #list_k = [0.10,0.20,0.30,0.40,0.50];
        list_result = [];
#         len_predict = len(actual_label_errors);
#         len_predict = len(label_errors_idx);
        len_predict = self.totalLen_benchmark;#以MV为基准
        for i_k in list_k:
            num_topk = round(i_k * len_predict);
            label_errors_idx_atK = label_errors_idx[:num_topk];
            intersection = np.intersect1d(label_errors_idx_atK,actual_label_errors);
            len_intersection = len(intersection);
            if len_intersection:
                P_k = len_intersection/num_topk;
            else:
                P_k = 0;
            list_result.append(P_k);
        return list_result;
    
    ### compute P@k
    def ComputeP_k(self,list_k,label_errors_idx,actual_label_errors):
#         list_k = [10,20,30,40,50];
#         list_k = [1,5,10];
        list_result = [];
        for i_k in list_k:
            if len(label_errors_idx)>=i_k:
                label_errors_idx_atK = label_errors_idx[:i_k];
                intersection = np.intersect1d(label_errors_idx_atK,actual_label_errors);
                len_intersection = len(intersection);
                P_k = len_intersection/i_k;
            else:
                P_k = np.nan;#如果预测误标签数量不大于k，则赋值为nan
            list_result.append(P_k);
        return list_result;
    
    ### compute IFA
    #RR只考虑排序后遇到的第一个实际buggy实例，是它的位置的倒数
    def ComputeIFA(self,label_errors_idx,actual_label_errors):
        len_errors_idx = len(label_errors_idx);
        IFA = 0
        for i in range(len_errors_idx):
            if label_errors_idx[i] in actual_label_errors:
                break;
            IFA += 1;
        if IFA == len_errors_idx:
            IFA = len(self.df_original);#如果预测的中没有TP，则赋值为最大值，数据集的实例数量。意味着还是需要审查整个数据集。
#             IFA = float('inf')
        return(IFA)
    
    ### compute SLOC(Inspect)
    def ComputeSLOC_Inspect(self,label_errors_idx):
        sumLOC = 0;
        for i_idx in label_errors_idx:
            sumLOC += self.df_original.loc[i_idx,'loc'];
        return sumLOC;
    
    ### compute NLPTLOC
    def ComputeNLPTLOC(self,TP,label_errors_idx):
        sumLOC = 0;
        if label_errors_idx.size != 0:
            for i_idx in label_errors_idx:
                sumLOC += self.df_original.loc[i_idx,'loc'];
            NLPTLOC = TP/sumLOC*1000;
        else:
            NLPTLOC = 0;
        return NLPTLOC;
    
    ### compute RR
    #RR只考虑排序后遇到的第一个实际buggy实例，是它的位置的倒数
    def ComputeRR(self,label_errors_idx,actual_label_errors):
        len_errors_idx = len(label_errors_idx);
        RR = 0
        for i in range(len_errors_idx):
            if label_errors_idx[i] in actual_label_errors:
                RR = 1/(i+1);
                break;
        return(RR)
    
    ### compute AP
    def ComputeAP(self,label_errors_idx,actual_label_errors):
        len_errors_idx = len(label_errors_idx);
        num_true = 0
        AP = 0
        for i in range(len_errors_idx):
            if label_errors_idx[i] in actual_label_errors:
                num_true = num_true + 1;
                AP = AP + num_true/(i+1);
        if len_errors_idx:
            AP = AP / len_errors_idx;
        else:
            AP = 0;
        return(AP)
    
    ### compute all evaluation indicators
    def calculateIndicators(self,label_errors_idx,actual_label_errors,origin_index):
        if origin_index.size!=0:
            list_label_errors_idx = label_errors_idx.tolist()
            set_label_errors_idx = set(list_label_errors_idx)#方法预测的误标签下标
            set_actual_label_errors = set(actual_label_errors)#实际的误标签下标
            set_origin_index = set(origin_index)#所有实例下标
            
            TP_label_errors = set_label_errors_idx & set_actual_label_errors;
            FP_label_errors = set_label_errors_idx - TP_label_errors;
            TN_label_errors = (set_origin_index - set_label_errors_idx) & (set_origin_index - set_actual_label_errors);
            FN_label_errors = set_actual_label_errors - TP_label_errors;
                
            TP = len(TP_label_errors)
            FP = len(FP_label_errors)
            TN = len(TN_label_errors)
            FN = len(FN_label_errors)
            self.TP=TP
            self.FP=FP
            self.TN=TN
            self.FN=FN
            
            if (TP+FP+TN+FN):
                accuracy = (TP + TN) / (TP+FP+TN+FN)
            else:
                accuracy = 0
            if (TP+FP):
                precision = TP / (TP+FP);
            else:
                precision = 0;
            if (TP+FN):
                recall = TP / (TP+FN);
            else:
                recall = 0;
            if (precision+recall):
                F1 = 2*precision*recall / (precision+recall);
            else:
                F1 = 0;
#             x = TP + FP
#             y = TP
#             n=TP+FN
#             N=TP+FP+FN+TN
#             if (y*N == 0):
#                 ER = np.nan
#             else:
#                 ER = (y*N-x*n)/(y*N)
#             if (x*n == 0):
#                 RI = np.nan
#             else:
#                 RI = (y*N-x*n)/(x*n)
            if (FP+TN) == 0:
                FAR = 1;#np.nan不合理，在统计中位数等会被忽略
            else:
                FAR = FP/(FP+TN);
            D2H = (((1-recall)**2 + (0-FAR)**2)/2)**0.5;
#             if np.isnan(recall) == False and np.isnan(FAR) == False:
#                 D2H = (((1-recall)**2 + (0-FAR)**2)/2)**0.5;
#             else:
#                 D2H = np.nan
#             if ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) == 0:
#                 MCC = np.nan
#             else:
#                 MCC = (TP*TN-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5);
            IFA = self.ComputeIFA(label_errors_idx,actual_label_errors);
            Inspect = len(label_errors_idx);
            LOC_Inspect = self.ComputeSLOC_Inspect(label_errors_idx);
            NLPTLOC = self.ComputeNLPTLOC(TP,label_errors_idx);
            AP = self.ComputeAP(label_errors_idx,actual_label_errors);
#             RR = self.ComputeRR(label_errors_idx,actual_label_errors);
            #P@K，代表前 K个预测值中有多少的精确率 (Precision)
    #         list_P_k = self.ComputeP_k(label_errors_idx,actual_label_errors);
            
            list_k = [1,5,10];
            list_P_k = self.ComputeP_k(list_k,label_errors_idx,actual_label_errors);
#             list_k = [0.05,0.10];
#             list_P_k_SLOC = self.ComputeP_k_percent_SLOC(list_k,label_errors_idx,actual_label_errors);
#             #P%K，代表前 K%个预测值中有多少的精确率 (Precision)
#             list_k = [0.10,0.20,0.30,0.40,0.50];#20%
#             list_P_k_percent = self.ComputeP_k_percent(list_k,label_errors_idx,actual_label_errors);
            
            
    #             print('% accuracy: {:.0%}'.format(accuracy))
    #             print('% precision: {:.0%}'.format(precision))
    #             print('% recall: {:.0%}'.format(recall))
    #             print('% F1: {:.0%}'.format(F1))
            
            list_indicators = [accuracy,precision,recall,F1,FAR,D2H,IFA,Inspect,LOC_Inspect,NLPTLOC,AP]
            list_indicators.extend(list_P_k);
#             list_indicators.extend(list_P_k_SLOC);
#             list_indicators.extend(list_P_k_percent);
            
        else:
            list_indicators = [np.nan,np.nan,np.nan,np.nan,
                               np.nan,np.nan,np.nan,np.nan,
                               np.nan,np.nan,np.nan,np.nan,
                               np.nan,np.nan,]
        return list_indicators;
    
    ### Returns theoretical worst values of all indicators
    def theoreticalWorstValues(self):
        accuracy = 0;
        precision = 0;
        recall = 0;
        F1 = 0;
        FAR = 1;
        D2H = 1;
        IFA = len(self.df_original);
        Inspect = len(self.df_original);
        LOC_Inspect = self.df_original['loc'].sum();
        NLPTLOC = 0;
        AP = 0;
        list_P_k = [0,0,0];
        list_indicators = [accuracy,precision,recall,F1,FAR,D2H,IFA,Inspect,LOC_Inspect,NLPTLOC,AP]
        list_indicators.extend(list_P_k);
        return list_indicators;
    
    ### 在现有的带有噪音的标签下，计算性能指标
    def calculatePerformanceIndicators(self,predicted_labels,origin_labels):
        TP, FP, TN, FN = 0, 0, 0, 0
        
        y_true = origin_labels
        y_pred = predicted_labels
        
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                TP += 1
            if y_true[i] == 0 and y_pred[i] == 1:
                FP += 1
            if y_true[i] == 0 and y_pred[i] == 0:
                TN += 1
            if y_true[i] == 1 and y_pred[i] == 0:
                FN += 1
        
        if (TP+FP):
            precision = TP / (TP+FP);
        else:
            precision = 0;
        if (TP+FN):
            recall = TP / (TP+FN);
        else:
            recall = 0;
        if (precision+recall):
            F1 = 2*precision*recall / (precision+recall);
        else:
            F1 = 0;
        
        return F1
        
    def getTP(self):
        return self.TP
        
        
        
        
        
        
        
        
        
        
        
        
