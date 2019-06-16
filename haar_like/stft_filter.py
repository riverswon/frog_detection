#STFT→log10→mel-filter→mean
#in filters and training_data_-1_1
#out start[Hz] stop[Hz] features... [filter_sample][2+fg+no]

#アマミイシガエル
#fs=44100, nfft=2048, hoplength=512

import librosa
import numpy as np
import pickle
import scipy.signal
from excel import load_excel_np
from haar_like_filter import one_haarlike_filter

# features... [filter_sample]
def wav_stft_filter(x, nfft, filters):
	p = 0.97
	x = scipy.signal.lfilter([1.0, -p], 1, x)
	spec = np.abs(librosa.stft(x, n_fft=nfft, hop_length=int(nfft/2), window='hamming'))[:int(nfft/2)]
	filters_features = np.dot(filters, spec)
	filters_features = np.mean(filters_features, axis=1)
	return filters_features

def frog2_train_select(num):
	fs = 44100
	nfft = 2048
	with open('haar_like/frog_data/filters.pkl', 'rb') as filters_pkl:
		filters = pickle.load(filters_pkl)
	filter_sample = filters.shape[0]
	para_hz, filters = np.hsplit(filters, [2])
	sample = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	filter_feature = np.empty(filter_sample)
	if num != 0:
		for i in range(0, sample[0][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num != 1:
		for i in range(sample[0][0], sample[0][0]+sample[1][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num != 2:
		for i in range(sample[0][0]+sample[1][0], sample[0][0]+sample[1][0]+sample[2][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num != 3:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num != 4:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]+sample[4][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	filter_feature = np.delete(filter_feature, obj=0, axis=0)
	filter_feature = np.hstack((para_hz, filter_feature.T))
	with open('haar_like/frog_data/frog2/frog2_train_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(filter_feature, train_mfcc_pkl, protocol=2)
	del filter_feature
	print('finished:'+str(num))

def frog1_train_select(num):
	fs = 44100
	nfft = 2048
	with open('haar_like/frog_data/filters.pkl', 'rb') as filters_pkl:
		filters = pickle.load(filters_pkl)
	filter_sample = filters.shape[0]
	para_hz, filters = np.hsplit(filters, [2])
	sample = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	filter_feature = np.empty(filter_sample)
	if num != 0:
		for i in range(0, sample[0][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num != 1:
		for i in range(sample[0][0], sample[0][0]+sample[1][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num != 2:
		for i in range(sample[0][0]+sample[1][0], sample[0][0]+sample[1][0]+sample[2][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num != 3:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num != 4:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]+sample[4][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	filter_feature = np.delete(filter_feature, obj=0, axis=0)
	filter_feature = np.hstack((para_hz, filter_feature.T))
	with open('haar_like/frog_data/frog1/frog1_train_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(filter_feature, train_mfcc_pkl, protocol=2)
	del filter_feature
	print('finished:'+str(num))

def back2_train_select(num):
	fs = 44100
	nfft = 2048
	with open('haar_like/frog_data/filters.pkl', 'rb') as filters_pkl:
		filters = pickle.load(filters_pkl)
	filter_sample = filters.shape[0]
	para_hz, filters = np.hsplit(filters, [2])
	sample = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	filter_feature = np.empty(filter_sample)
	if num != 0:
		for i in range(0, sample[0][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num != 1:
		for i in range(sample[0][1], sample[0][1]+sample[1][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num != 2:
		for i in range(sample[0][1]+sample[1][1], sample[0][1]+sample[1][1]+sample[2][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num != 3:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num != 4:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]+sample[4][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	filter_feature = np.delete(filter_feature, obj=0, axis=0)
	filter_feature = np.hstack((para_hz, filter_feature.T))
	with open('haar_like/frog_data/back2/back2_train_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(filter_feature, train_mfcc_pkl, protocol=2)
	del filter_feature
	print('finished:'+str(num))

def back1_train_select(num):
	fs = 44100
	nfft = 2048
	with open('haar_like/frog_data/filters.pkl', 'rb') as filters_pkl:
		filters = pickle.load(filters_pkl)
	filter_sample = filters.shape[0]
	para_hz, filters = np.hsplit(filters, [2])
	sample = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	filter_feature = np.empty(filter_sample)
	if num != 0:
		for i in range(0, sample[0][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num != 1:
		for i in range(sample[0][1], sample[0][1]+sample[1][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num != 2:
		for i in range(sample[0][1]+sample[1][1], sample[0][1]+sample[1][1]+sample[2][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num != 3:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num != 4:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]+sample[4][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	filter_feature = np.delete(filter_feature, obj=0, axis=0)
	filter_feature = np.hstack((para_hz, filter_feature.T))
	with open('haar_like/frog_data/back1/back1_train_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(filter_feature, train_mfcc_pkl, protocol=2)
	del filter_feature
	print('finished:'+str(num))

def experint2_train_select(num):
	with open('haar_like/frog_data/frog2/frog2_train_'+str(num)+'.pkl', 'rb') as frog2_pkl:
		frog2 = pickle.load(frog2_pkl)
	with open('haar_like/frog_data/frog1/frog1_train_'+str(num)+'.pkl', 'rb') as frog1_pkl:
		frog1 = pickle.load(frog1_pkl)
	_, frog1 = np.hsplit(frog1, [2])
	frog2 = np.hstack((frog2, frog1))
	del frog1
	with open('haar_like/frog_data/back2/back2_train_'+str(num)+'.pkl', 'rb') as back2_pkl:
		back2 = pickle.load(back2_pkl)
	_, back2 = np.hsplit(back2, [2])
	frog2 = np.hstack((frog2, back2))
	del back2
	with open('haar_like/frog_data/back1/back1_train_'+str(num)+'.pkl', 'rb') as back1_pkl:
		back1 = pickle.load(back1_pkl)
	_, back1 = np.hsplit(back1, [2])
	frog2 = np.hstack((frog2, back1))
	with open('haar_like/frog_data/experiment2/experiment2_train_x_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(frog2, train_mfcc_pkl, protocol=2)
	del frog2
	print('finished:'+str(num))

def experint1_train_select(num):
	with open('haar_like/frog_data/frog1/frog1_train_'+str(num)+'.pkl', 'rb') as frog1_pkl:
		frog1 = pickle.load(frog1_pkl)
	with open('haar_like/frog_data/frog2/frog2_train_'+str(num)+'.pkl', 'rb') as frog2_pkl:
		frog2 = pickle.load(frog2_pkl)
	_, frog2 = np.hsplit(frog2, [2])
	frog1 = np.hstack((frog1, frog2))
	del frog2
	with open('haar_like/frog_data/back2/back2_train_'+str(num)+'.pkl', 'rb') as back2_pkl:
		back2 = pickle.load(back2_pkl)
	_, back2 = np.hsplit(back2, [2])
	frog1 = np.hstack((frog1, back2))
	del back2
	with open('haar_like/frog_data/back1/back1_train_'+str(num)+'.pkl', 'rb') as back1_pkl:
		back1 = pickle.load(back1_pkl)
	_, back1 = np.hsplit(back1, [2])
	frog1 = np.hstack((frog1, back1))
	with open('haar_like/frog_data/experiment1/experiment1_train_x_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(frog1, train_mfcc_pkl, protocol=2)
	del frog1
	print('finished:'+str(num))


def frog_training_stft_filter(fs, nfft, wav_time, section):
	cross_validation = 5
	with open('haar_like/frog_data/filters_'+str(nfft)+'.pkl', 'rb') as filters_pkl:
		filters_x = pickle.load(filters_pkl)
		filter_sample = filters_x.shape[0]
		para_hz, filters = np.hsplit(filters_x, [2])

		split_sample = int(wav_time / section)
		split_interval = int(fs * section)
		evaluation = load_excel_np('evaluation/frog_sheet_'+str(section)+'s.xlsx')

		filters_features = []
		for i in range(cross_validation):
			filters_features1 = np.empty(filter_sample)
			training_evaluation = np.empty(1, int)
			for j in range(cross_validation):
				if i == j:
					continue
				x, f = librosa.load('test_data/frog'+str(j+1)+'.wav', sr=fs)
				for k in range(split_sample):
					x_split = x[k*split_interval:(k+1)*split_interval]
					filters_features2 = wav_stft_filter(x_split, nfft, filters)
					filters_features1 = np.vstack((filters_features1, filters_features2)) # features[split_sample][filter_sample]
				training_evaluation = np.hstack((training_evaluation, evaluation[j]))
			filters_features1 = np.delete(filters_features1, 0, 0) # delite [0][]
			training_evaluation = training_evaluation[1:]
			training_evaluation = training_evaluation.reshape((split_sample*4, 1))
			filters_features1 = np.hstack((training_evaluation, filters_features1))
			filters_features1 = filters_features1[np.argsort(filters_features1[:, 0])[::-1]]
			filters_features1 = np.delete(filters_features1, obj=0, axis=1)
			filters_features1 = np.hstack((para_hz, filters_features1.T))
			filters_features.append(filters_features1)
	print('finished:frog_training_stft_filter')
	return filters_features

def frog_test_stft_filter(in_train_path, out_test_path, fs, nfft, wav_time, section):
	cross_validation = 5
	index_max = int(nfft / 2)
	split_sample = int(wav_time / section)
	split_interval = int(fs * section)
	with open(in_train_path, 'rb') as training_sort_pkl:
		training_sort = pickle.load(training_sort_pkl)

		filters_features = []
		for i in range(cross_validation):
			#make the filter in the upper 20th accuracy
			accy_thresh, para_hz, features = np.hsplit(training_sort[i], [2, 4])
			para_hz, no_use = np.vsplit(para_hz, [12])

			filterbanks = np.empty(index_max)
			for j in range(12):
				filterbank1 = one_haarlike_filter(fs, nfft, para_hz[j][0], para_hz[j][1])
				filterbanks = np.vstack((filterbanks, filterbank1))
			filterbanks = np.delete(filterbanks, 0, 0) # delite [0][]

			x, f = librosa.load('test_data/frog'+str(i+1)+'.wav', sr=fs)
			filters_features1 = np.empty(12) # use the training_data stack
			for j in range(split_sample):
				x_split = x[j*split_interval:(j+1)*split_interval]
				filters_features2 = wav_stft_filter(x_split, nfft, filterbanks)
				filters_features1 = np.vstack((filters_features1, filters_features2)) # features[split_sample][filter_sample]
			filters_features1 = np.delete(filters_features1, 0, 0) # delite [0][]
			filters_features.append(filters_features1)
	with open(out_test_path, 'wb') as test_sort_pkl:
		pickle.dump(filters_features , test_sort_pkl)
	print('finished:frog_test_stft_filter')
	return filters_features

#いらない
# start[Hz] stop[Hz] features... [filter_sample][2+fg+no]
def frog_training_stft_log_filter1(out_train_path, training_sample):
	cross_validation = 5
	with open('haar_like/frog_data/filters.pkl', 'rb') as filters_pkl:
		filters_x = pickle.load(filters_pkl)
		filter_sample = filters_x.shape[0]
		para_hz, filters = np.hsplit(filters_x, [2])

		filters_features = []
		for i in range(cross_validation):
			filters_features1 = np.empty(filter_sample)
			for j in range(cross_validation):
				if i == j:
					continue
				for k in range(training_sample[j][0]):
					x, f = librosa.load('training_data/'+str(j+1)+'fg'+str(k+1)+'.wav', sr=44100)
					filters_features2 = wav_stft_log_filter(x, filters)
					filters_features1 = np.vstack((filters_features1, filters_features2))
			for j in range(cross_validation):
				if i == j:
					continue
				for k in range(training_sample[j][1]):
					x, f = librosa.load('training_data/'+str(j+1)+'no'+str(k+1)+'.wav', sr=44100)
					filters_features2 = wav_stft_log_filter(x, filters)
					filters_features1 = np.vstack((filters_features1, filters_features2))
			filters_features1 = np.delete(filters_features1, obj=0, axis=0)
			filters_features1 = np.hstack((para_hz, filters_features1.T))
			filters_features.append(filters_features1)
		with open(out_train_path, 'wb') as filters_features_pkl:
			pickle.dump(filters_features , filters_features_pkl)
	print('finished:frog_training_stft_log_filter1')


