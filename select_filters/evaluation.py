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

def evaluate(predict_path, answer_path):
    predict = load_excel_np(predict_path)
    answer = load_excel_np(answer_path)
    test_sample = predict.shape[0]
    split_sample = predict.shape[1]
    TP = np.zeros(test_sample)
    FP = np.zeros(test_sample)
    FN = np.zeros(test_sample)
    TN = np.zeros(test_sample)
    for i in range(test_sample):
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
    return Accuracy_mean, Recall_mean, Precision_mean, F_value_mean

def frog_evaluation(section, section_name, nfft, hoplength, para1, para2, way):
    predict_path = 'mel_filter/frog_data/predict.xlsx'
    Accuracy, Recall, Precision, F_value = evaluate(predict_path, 'evaluation/frog_sheet_'+str(section)+'s.xlsx')
    result = np.empty(4)
    result[0] = Accuracy
    result[1] = Recall
    result[2] = Precision
    result[3] = F_value
    np_excel(result, 'mel_filter/frog_data/predict/'+section_name+'_'+str(nfft)+'_'+str(hoplength)+'_'+para1+'_'+para2+'_'+way+'.xlsx')

if __name__ == '__main__':
    #path = 'mfcc/frog_data/predict.xlsx'
    path = 'mel_filter/frog_data/predict.xlsx'
    #path = 'haar_like/frog_data/predict.xlsx'
    frog_evaluation(path, 0.5)



