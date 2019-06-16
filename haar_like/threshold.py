# in start[Hz] stop[Hz] features...
# out accuracy threshold start[Hz] stop[Hz] features...

import numpy as np
from excel import load_excel_np
import pickle

def set_threshold(training_one_mel, sample):
	filter_sample = training_one_mel.shape[0]
	# para_hz, fg, noに分割
	para_hz, one_mel_fg, one_mel_no = np.hsplit(training_one_mel, [2, 2 + sample[0]])
	threshold_data = np.empty(2) # save accuracy threshold
	# decide the threshold
	for i in range(filter_sample):
		training_max = max(training_one_mel[i][2:])
		training_min = min(training_one_mel[i][2:])
		df = (training_max - training_min) / 300.0
		accuracy_save = 0.0 # to ask for max accuracy
		for threshold in np.arange(training_min, training_max, df):
			# set no as left and fg as right 
			count = 0.0
			for feature in one_mel_fg[i]:
				if feature >= threshold:
					count+=1.0
			for feature in one_mel_no[i]:
				if feature < threshold:
					count+=1.0
			accuracy1 = count / (sample[0] + sample[1])
			# set fg as left and no as right 
			count = 0.0
			for feature in one_mel_fg[i]:
				if feature <= threshold:
					count+=1.0
			for feature in one_mel_no[i]:
				if feature > threshold:
					count+=1.0
			accuracy2 = count / (sample[0] + sample[1])
			if accuracy1 >= accuracy2:
				accuracy = accuracy1
			elif accuracy2 > accuracy1:
				accuracy = accuracy2
			if accuracy > accuracy_save:
				accuracy_save = accuracy
				threshold_save = threshold
		threshold_data1 = np.array([accuracy_save, threshold_save])
		threshold_data = np.vstack((threshold_data, threshold_data1))
	threshold_data = np.delete(threshold_data, obj=0, axis=0) # delite [0][]
	# save accuracy threshold start[Hz] stop[Hz] features...
	threshold_data = np.hstack((threshold_data, training_one_mel))
	return threshold_data

def experiment2_set_threshold(num):
	with open('haar_like/frog_data/experiment2/experiment2_train_x_'+str(num)+'.pkl', 'rb') as filters_pkl:
		filters_features = pickle.load(filters_pkl)
	sample2 = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	sample1 = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	sample = np.zeros(2, dtype=int)
	for i in range(5):
		if i == num:
			continue
		sample[0] += sample2[i][0]
		sample[1] = sample[1] + sample2[i][1] + sample1[i][0] + sample1[i][1]
	threshold_data = set_threshold(filters_features, sample)
	with open('haar_like/frog_data/experiment2/experiment2_train_threshold_'+str(num)+'.pkl', 'wb') as threshold_data_pkl:
		pickle.dump(threshold_data , threshold_data_pkl, protocol=2)
	del threshold_data
	print('finished:'+str(num))

def experiment1_set_threshold(num):
	with open('haar_like/frog_data/experiment1/experiment1_train_x_'+str(num)+'.pkl', 'rb') as filters_pkl:
		filters_features = pickle.load(filters_pkl)
	sample2 = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	sample1 = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	sample = np.zeros(2, dtype=int)
	for i in range(5):
		if i == num:
			continue
		sample[0] += sample1[i][0]
		sample[1] = sample[1] + sample1[i][1] + sample2[i][0] + sample2[i][1]
	threshold_data = set_threshold(filters_features, sample)
	with open('haar_like/frog_data/experiment1/experiment1_train_threshold_'+str(num)+'.pkl', 'wb') as threshold_data_pkl:
		pickle.dump(threshold_data , threshold_data_pkl, protocol=2)
	del threshold_data
	print('finished:'+str(num))
		
