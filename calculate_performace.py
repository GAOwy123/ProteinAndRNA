# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score


def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
    
    acc = 'na'
    precision = 'na'
    sensitivity = 'na'
    specificity = 'na'
    MCC = 'na'
    AUC = 'na'        
    acc = float(tp + tn)/test_num
    if tp + fp != 0:
        precision = float(tp)/(tp+ fp)
    
    if tp+fn !=0:
        sensitivity = float(tp)/ (tp+fn)
    
    if tn+fp !=0:
        specificity = float(tn)/(tn + fp)
    
    if np.sqrt(np.float64((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))!=0:
        MCC = float(tp*tn-fp*fn)/(np.sqrt(np.float64((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))
    
    AUC = roc_auc_score(labels, pred_y)    

    return acc, precision, sensitivity, specificity, MCC , AUC
