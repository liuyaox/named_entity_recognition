# -*- coding: utf-8 -*-
"""
Created:    2019-11-26 15:24:02
Author:     liuyao8
Descritipn: 
"""

import numpy as np
from keras import backend as K


# Custom Metric
def eval_helper(y_true, y_pred):
    '''模型评估辅助工具：统计各label在ytrue和ypred中的数量，以及各label在ytrue和ypred相应位置取值相同的数量'''
    # ypred's shape: (, maxlen, n_labels)  ytrue's shape: (, maxlen, 1)
    y_pred = np.argmax(y_pred, -1)
    y_true = y_true[:, :, 0]
    pred_dic, true_dic, corr_dic = {}, {}, {}
    for ypreds, ytrues in zip(y_pred, y_true):
        for ypred, ytrue in zip(ypreds, ytrues):
            ypred, ytrue = int(ypred), int(ytrue)
            pred_dic[ypred] = pred_dic.get(ypred, 0) + 1
            true_dic[ytrue] = true_dic.get(ytrue, 0) + 1
            if ytrue == ypred:
                corr_dic[ytrue] = corr_dic.get(ytrue, 0) + 1
    return (pred_dic, true_dic, corr_dic)


def pre_rec(y_true, y_pred, mask_less=2, n_tags=8):
    '''模型评估时使用'''
    pred_dic, true_dic, corr_dic = eval_helper(y_true, y_pred)
    corrs, preds, trues = 0, 0,  0
    for i in range(mask_less, n_tags):
        corrs += corr_dic[i]
        preds += pred_dic[i]
        trues += true_dic[i]
    pre = round(corrs / preds, 4)
    rec = round(corrs / trues, 4)
    return (pre, rec)


def pre_rec2(y_true, y_pred, mask_less=2):
    '''模型评估时使用  结果数值同pre_rec'''
    y_pred = np.argmax(y_pred, -1)
    y_true = y_true[:, :, 0]
    judge = np.equal(y_pred, y_true)
    
    mask1 = np.greater_equal(y_pred, mask_less)
    mask2 = np.greater_equal(y_true, mask_less)
    
    pre = round(np.sum(judge * mask1) / np.sum(mask1), 4)
    rec = round(np.sum(judge * mask2) / np.sum(mask2), 4)
    return (pre, rec)


def print_metrics(y_true, y_pred):
    '''模型评估时输出结果'''
    pred_dic, true_dic, corr_dic = eval_helper(y_true, y_pred)
    print('pred_dic: ')
    print(sorted(pred_dic.items(), key=lambda x: x[0]))
    print('true_dic: ')
    print(sorted(true_dic.items(), key=lambda x: x[0]))
    print('corr_dic: ')
    print(sorted(corr_dic.items(), key=lambda x: x[0]))
    print('-----------Precision-----------')
    for i in range(8):
        print(str(i) + ': ' + str(round(corr_dic[i] / pred_dic[i], 4)))
    print('-----------Recall-----------')
    for i in range(8):
        print(str(i) + ': ' + str(round(corr_dic[i] / true_dic[i], 4)))
    print('# 有mask，忽略1  最合理的metric')
    print(pre_rec(y_true, y_pred, mask_less=2))
    print('# 有mask，不忽略1  accuracy与输出一致，说明网络里的mask的确生效了')
    print(pre_rec(y_true, y_pred, mask_less=1))
    print('# 没有mask(不忽略0))')
    print(pre_rec(y_true, y_pred, mask_less=0))

    
def acc_pre_rec(y_true, y_pred, mask_less=2):
    '''模型训练时使用  与pre_rec2一样的逻辑，但结果数值却不一样？？？'''
    y_pred = K.argmax(y_pred, -1)
    y_true = K.cast(y_true[:, :, 0], K.dtype(y_pred))
    judge = K.cast(K.equal(y_pred, y_true), K.floatx())

    mask = K.cast(K.not_equal(y_pred, 0), K.floatx())
    mask1 = K.cast(K.greater_equal(y_pred, mask_less), K.floatx())  # 以y_pred为基准，用于计算precision
    mask2 = K.cast(K.greater_equal(y_true, mask_less), K.floatx())  # 以y_true为基准，用于计算recall
    
    acc = K.sum(judge * mask) / (K.sum(mask) + K.epsilon())    # 与crf_accuracy一模一样！
    pre = K.sum(judge * mask1) / (K.sum(mask1) + K.epsilon())
    rec = K.sum(judge * mask2) / (K.sum(mask2) + K.epsilon())
    return (acc, pre, rec)


def acc(y_true, y_pred):
    return acc_pre_rec(y_true, y_pred)[0]

def pre(y_true, y_pred):
    return acc_pre_rec(y_true, y_pred)[1]

def rec(y_true, y_pred):
    return acc_pre_rec(y_true, y_pred)[2]

