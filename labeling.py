import numpy as np
import pandas as pd
import openpyxl

#get label_time_data.txt from audacity
def get_time(path):
    f = open(path, "r")
    label = f.readlines()
    time_sheet = np.empty((len(label),2))
    for i in range(len(label)):
        line = label[i].split()
        time_sheet[i][0] = float(line[0])
        time_sheet[i][1] = float(line[1])    
    if time_sheet[len(label)-1][1] > 120:
        time_sheet[len(label)-1][1] = 120
    return time_sheet

#make [0:1]sheet from label_time_data in numpy array
def sheet_np_array(time_sheet, wav_time, section):
    sheet = np.zeros(int(wav_time / section),dtype=int)
    for i in range(int(time_sheet.shape[0])):
        start = np.floor(time_sheet[i][0] / section) #ex 0.85→8.5→8
        stop = np.ceil(time_sheet[i][1] / section) #ex 0.85→8.5→9
        for j in range(int(start),int(stop)):
            sheet[j] = 1
    return sheet

#change numpy array into excel
def np_excel(sheet, path):
    df = pd.DataFrame(sheet)
    df.to_excel(path)

#frog excel sheet
def frog_labeling(section):
    wav_time = 120
    frog_sample = 5
    sheet = np.empty((frog_sample,int(wav_time / section)), dtype=int)
    for i in range(frog_sample):
        time_sheet = get_time('data_label/frog'+str(i+1)+'.txt')
        sheet[i] = sheet_np_array(time_sheet, wav_time, section)
    np_excel(sheet, 'evaluation/frog_sheet_'+str(section)+'s.xlsx')

frog_labeling(1.0)