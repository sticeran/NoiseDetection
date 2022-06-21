'''
Created on 2020年9月24日

@author: njdx
'''
import cleanlab
import numpy as np


class ConfidentLearning(object):
    
    s = [];
    psx = [];
    strategy_name = "ConfidentJoint";#默认ConfidentJoint

    
    def __init__(self,strategy_name):
        self.s = [];
        self.psx = [];
        self.strategy_name = strategy_name;
    
    #===Remove label errors if given label == model prediction===#
    def FUNCTION_RemoveLabelErrors(self,label_errors_bool):#传入真实标签列表，返回生成的误标签列表和误标签的下标
        # Remove label errors if given label == model prediction
        for i, pred_label in enumerate(self.psx.argmax(axis=1)):
            # np.all let's this work for multi_label and single label
            if label_errors_bool[i] and np.all(pred_label == self.s[i]):
                label_errors_bool[i] = False
        return label_errors_bool
    #===end===#
    
    #===identify noise===#
    def FUNCTION_IdentifyNoise_bool(self,i_method):
        
        # Find the number of unique classes if K is not given
        K = len(np.unique(self.s))
        
        label_errors_bool = [];
        if i_method == "Confusion":
            # Method: C_confusion
            label_errors_bool = cleanlab.baseline_methods.baseline_argmax(self.psx, self.s)
        #     label_errors_idx = np.arange(len(s))[baseline_argmax_bool]
        elif i_method == "ConfidentJoint":
            # Method: confident joint only
            label_error_mask = np.zeros(len(self.s), dtype=bool)
            label_error_indices = cleanlab.latent_estimation.compute_confident_joint(
                self.s, self.psx, return_indices_of_off_diagonals=True
            )[1]
            for idx in label_error_indices:
                label_error_mask[idx] = True
            label_errors_bool = label_error_mask
        #     label_errors_idx = np.arange(len(s))[baseline_conf_joint_only_bool]
        elif i_method == "PBC" and K != 1:
            # Method: CL: PBC
            label_errors_bool = cleanlab.pruning.get_noise_indices(
                        self.s, self.psx, prune_method='prune_by_class')
        #     label_errors_idx = np.arange(len(s))[baseline_cl_pbc_bool]
        elif i_method == "PBC" and K == 1:
            # Method: CL: PBC
            label_errors_bool = np.array([])
        else:
            print("error")
        return label_errors_bool;
    #===end===#
    
    #===Get self confidence: lateral margin===#
    def FUNCTION_GetLateralMargin(self,label_errors_bool,label_errors_idx):
        # self confidence is the holdout probability that an example
        # belongs to its given class label
        self_confidence = np.array(
            [np.mean(self.psx[i][self.s[i]]) for i in label_errors_idx]
        )
        margin = self_confidence - self.psx[label_errors_bool].max(axis=1)
        return margin
    #===end===#
    
    #===This function gets the confidence value that each instance belongs to the noise===#
    def FUNCTION_confidentLearning_fromPredictionProbability(self,psx,s):
        self.s = s;
        self.psx = psx;
        # 根据选取的噪音识别策略识别噪音bool数组
        method_lebel_errors_bool = self.FUNCTION_IdentifyNoise_bool(self.strategy_name);
        
        if method_lebel_errors_bool.size != 0:
            # Remove label errors if given label == model prediction
            method_lebel_errors_bool = self.FUNCTION_RemoveLabelErrors(method_lebel_errors_bool);
            # Convert bool mask to index mask
            label_errors_idx = np.arange(len(method_lebel_errors_bool))[method_lebel_errors_bool];
#                 print(self.list_methods_noiseIdentification[i_method]," len: ",len(label_errors_idx))
            
            # get lateral margin: self confidence is the holdout probability that an example belongs to its given class label
            array_lateralMargin = self.FUNCTION_GetLateralMargin(method_lebel_errors_bool,label_errors_idx)
            
        return label_errors_idx,array_lateralMargin;    
    
    
    