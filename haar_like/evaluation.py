#Accuracy(正解率) : (TP + TN) / (TP + FP + FN + TN)
#Recall(再現率) : TP / (TP + FN)
#Precision(適合率) : TP / (TP + FP)
#F-value : 2 * Recall * Precision / (Recall + Precision)
#　　 　実際
#　　　 O  X
#予　O TP FP
#測　X FN TN

import numpy as np
from excel import *

def evaluate(predict_path, answer_path, result_path):
    predict = load_excel_np(predict_path)
    answer = load_excel_np(answer_path)
    cross_validation = predict.shape[0]
    split_sample = predict.shape[1]
    TP = np.zeros(cross_validation)
    FP = np.zeros(cross_validation)
    FN = np.zeros(cross_validation)
    TN = np.zeros(cross_validation)
    for i in range(cross_validation):
        for j in range(split_sample):
            if predict[i][j] == 1 and answer[i][j] == 1:
                TP[i]+=1
            elif predict[i][j] == 1 and answer[i][j] == 0:
                FP[i]+=1
            elif predict[i][j] == 0 and answer[i][j] == 1:
                FN[i]+=1
            else:
                TN[i]+=1
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    Accuracy_mean = np.mean(Accuracy)
    Recall = TP / (TP + FN)
    Recall_mean = np.mean(Recall)
    Precision = TP / (TP + FP)
    Precision_mean = np.mean(Precision)
    F_value = 2 * Recall * Precision / (Recall + Precision)
    F_value_mean = np.mean(F_value)
    predict_list = [['TP', 'FP', 'FN', 'TN', 'Accuracy', 'Recall', 'Precision', 'F-Value'], [np.sum(TP), np.sum(FP), np.sum(FN), np.sum(TN), Accuracy_mean, Recall_mean, Precision_mean, F_value_mean]]
    np_excel(predict_list, result_path)
    print('finished:evaluate')


if __name__ == '__main__':
    predict_path = 'select_filters/predict.xlsx'
    answer_path = 'evaluation/frog_sheet_0.5s.xlsx'
    result_path = 'select_filters/result.xlsx'
    evaluate(predict_path, answer_path, result_path)
