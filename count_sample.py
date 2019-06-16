import numpy as np
import pandas as pd

#change numpy array into excel
def np_excel(sheet, path):
    df = pd.DataFrame(sheet)
    df.to_excel(path)

def load_excel_np(path):
    df = pd.read_excel(path)
    df = df.values
    return df

def training_same_section(section):
    evaluation = load_excel_np('evaluation/frog_sheet_'+str(section)+'s.xlsx')
    count_sample = np.zeros((5, 2), int)
    count_sample1 = np.zeros((5, 2), int)
    for i in range(evaluation.shape[0]):
        for j in range(evaluation.shape[1]):
            if evaluation[i][j] == 1:
                count_sample1[i][0] += 1
            elif evaluation[i][j] == 0:
                count_sample1[i][1] += 1
    for i in range(evaluation.shape[0]):
        for j in range(evaluation.shape[0]):
            if i == j:
                continue
            else:
                count_sample[i][0] += count_sample1[j][0]
                count_sample[i][1] += count_sample1[j][1]
    np_excel(count_sample, 'evaluation/test_sample_'+str(section)+'s.xlsx')

def training_not_same_section():
    cross_validation = 5
    training_sample = np.empty((cross_validation, 2), dtype=int)
    training_sample[0][0] = 90 # fg in 1.wav
    training_sample[0][1] = 46 # no in 1.wav
    training_sample[1][0] = 81 # fg in 2.wav
    training_sample[1][1] = 51 # no in 2.wav
    training_sample[2][0] = 65 # fg in 3.wav
    training_sample[2][1] = 40 # no in 3.wav
    training_sample[3][0] = 82 # fg in 4.wav
    training_sample[3][1] = 48 # no in 4.wav
    training_sample[4][0] = 86 # fg in 5.wav
    training_sample[4][1] = 46 # no in 5.wav
    cross_vali_sample = np.empty((cross_validation, 2), dtype=int)
    cross_vali_sample[0][0] = 314 # fg in test1
    cross_vali_sample[0][1] = 185 # no in test1
    cross_vali_sample[1][0] = 323 # fg in test2
    cross_vali_sample[1][1] = 180 # no in test2
    cross_vali_sample[2][0] = 339 # fg in test3
    cross_vali_sample[2][1] = 191 # no in test3
    cross_vali_sample[3][0] = 322 # fg in test4
    cross_vali_sample[3][1] = 183 # no in test4
    cross_vali_sample[4][0] = 318 # fg in test5
    cross_vali_sample[4][1] = 185 # no in test5
    np_excel(training_sample, 'evaluation/training_sample.xlsx')
    np_excel(cross_vali_sample, 'evaluation/cross_validation_sample.xlsx')

if __name__ == '__main__':
    section = 1.0
    #training_same_section(section)
    training_same_section(1.0)
