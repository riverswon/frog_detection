#STFT→log10→mel-filter→mean
#in filters and training_data_-1_1
#out start[Hz] stop[Hz] features... [filter_sample][2+fg+no]

import librosa
import numpy as np
import pickle
from excel import load_excel_np
from mel_filter import one_melFilterBank
import scipy.signal

# features... [filter_sample]
def wav_stft_filter(x, nfft, filters):
	p = 0.97
	y = scipy.signal.lfilter([1.0, -p], 1, x)
	spec = np.abs(librosa.stft(y, n_fft=nfft, hop_length=int(nfft/2), window='hamming'))[:int(nfft/2)]
	filters_features = np.dot(filters, spec)
	filters_features = np.log10(filters_features)
	filters_features = np.mean(filters_features, axis=1)
	return filters_features

def frog2_train_select(num):
	fs = 44100
	nfft = 2048
	with open('select_filters/frog_data/filters.pkl', 'rb') as filters_pkl:
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
	with open('select_filters/frog_data/frog2/frog2_train_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(filter_feature, train_mfcc_pkl, protocol=2)
	del filter_feature
	print('finished:'+str(num))

def frog1_train_select(num):
	fs = 44100
	nfft = 2048
	with open('select_filters/frog_data/filters.pkl', 'rb') as filters_pkl:
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
	with open('select_filters/frog_data/frog1/frog1_train_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(filter_feature, train_mfcc_pkl, protocol=2)
	del filter_feature
	print('finished:'+str(num))

def back2_train_select(num):
	fs = 44100
	nfft = 2048
	with open('select_filters/frog_data/filters.pkl', 'rb') as filters_pkl:
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
	with open('select_filters/frog_data/back2/back2_train_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(filter_feature, train_mfcc_pkl, protocol=2)
	del filter_feature
	print('finished:'+str(num))

def back1_train_select(num):
	fs = 44100
	nfft = 2048
	with open('select_filters/frog_data/filters.pkl', 'rb') as filters_pkl:
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
	with open('select_filters/frog_data/back1/back1_train_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(filter_feature, train_mfcc_pkl, protocol=2)
	del filter_feature
	print('finished:'+str(num))

def experiment2_train_select(num):
	with open('select_filters/frog_data/frog2/frog2_train_'+str(num)+'.pkl', 'rb') as frog2_pkl:
		frog2 = pickle.load(frog2_pkl)
	with open('select_filters/frog_data/frog1/frog1_train_'+str(num)+'.pkl', 'rb') as frog1_pkl:
		frog1 = pickle.load(frog1_pkl)
	_, frog1 = np.hsplit(frog1, [2])
	frog2 = np.hstack((frog2, frog1))
	del frog1
	with open('select_filters/frog_data/back2/back2_train_'+str(num)+'.pkl', 'rb') as back2_pkl:
		back2 = pickle.load(back2_pkl)
	_, back2 = np.hsplit(back2, [2])
	frog2 = np.hstack((frog2, back2))
	del back2
	with open('select_filters/frog_data/back1/back1_train_'+str(num)+'.pkl', 'rb') as back1_pkl:
		back1 = pickle.load(back1_pkl)
	_, back1 = np.hsplit(back1, [2])
	frog2 = np.hstack((frog2, back1))
	with open('select_filters/frog_data/experiment2/experiment2_train_x_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(frog2, train_mfcc_pkl, protocol=2)
	del frog2
	print('finished:'+str(num))

def experiment1_train_select(num):
	with open('select_filters/frog_data/frog1/frog1_train_'+str(num)+'.pkl', 'rb') as frog1_pkl:
		frog1 = pickle.load(frog1_pkl)
	with open('select_filters/frog_data/frog2/frog2_train_'+str(num)+'.pkl', 'rb') as frog2_pkl:
		frog2 = pickle.load(frog2_pkl)
	_, frog2 = np.hsplit(frog2, [2])
	frog1 = np.hstack((frog1, frog2))
	del frog2
	with open('select_filters/frog_data/back2/back2_train_'+str(num)+'.pkl', 'rb') as back2_pkl:
		back2 = pickle.load(back2_pkl)
	_, back2 = np.hsplit(back2, [2])
	frog1 = np.hstack((frog1, back2))
	del back2
	with open('select_filters/frog_data/back1/back1_train_'+str(num)+'.pkl', 'rb') as back1_pkl:
		back1 = pickle.load(back1_pkl)
	_, back1 = np.hsplit(back1, [2])
	frog1 = np.hstack((frog1, back1))
	with open('select_filters/frog_data/experiment1/experiment1_train_x_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(frog1, train_mfcc_pkl, protocol=2)
	del frog1
	print('finished:'+str(num))

#non frog1
def experiment4_train_select(num):
	with open('select_filters/frog_data/frog2/frog2_train_'+str(num)+'.pkl', 'rb') as frog2_pkl:
		frog2 = pickle.load(frog2_pkl)
	with open('select_filters/frog_data/back2/back2_train_'+str(num)+'.pkl', 'rb') as back2_pkl:
		back2 = pickle.load(back2_pkl)
	_, back2 = np.hsplit(back2, [2])
	frog2 = np.hstack((frog2, back2))
	del back2
	with open('select_filters/frog_data/back1/back1_train_'+str(num)+'.pkl', 'rb') as back1_pkl:
		back1 = pickle.load(back1_pkl)
	_, back1 = np.hsplit(back1, [2])
	frog2 = np.hstack((frog2, back1))
	with open('select_filters/frog_data/experiment4/experiment4_train_x_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(frog2, train_mfcc_pkl, protocol=2)
	del frog2
	print('finished:'+str(num))

#non frog2
def experiment3_train_select(num):
	with open('select_filters/frog_data/frog1/frog1_train_'+str(num)+'.pkl', 'rb') as frog1_pkl:
		frog1 = pickle.load(frog1_pkl)
	with open('select_filters/frog_data/back2/back2_train_'+str(num)+'.pkl', 'rb') as back2_pkl:
		back2 = pickle.load(back2_pkl)
	_, back2 = np.hsplit(back2, [2])
	frog1 = np.hstack((frog1, back2))
	del back2
	with open('select_filters/frog_data/back1/back1_train_'+str(num)+'.pkl', 'rb') as back1_pkl:
		back1 = pickle.load(back1_pkl)
	_, back1 = np.hsplit(back1, [2])
	frog1 = np.hstack((frog1, back1))
	with open('select_filters/frog_data/experiment3/experiment3_train_x_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(frog1, train_mfcc_pkl, protocol=2)
	del frog1
	print('finished:'+str(num))

def experiment2_test_select(num):
	fs = 44100
	nfft = 2048
	index_max = int(nfft / 2)
	with open('select_filters/frog_data/experiment2/experiment2_train_sort_'+str(num)+'.pkl', 'rb') as training_sort_pkl:
		training_sort = pickle.load(training_sort_pkl)
	_1, para_hz, _2 = np.hsplit(training_sort, [2, 4])
	del training_sort
	del _1
	del _2
	para_hz, _ = np.vsplit(para_hz, [20])
	filters = np.empty(index_max)
	for i in range(20):
		filterbank1 = one_melFilterBank(fs, nfft, para_hz[i][0], para_hz[i][1])
		filters = np.vstack((filters, filterbank1))
	filters = np.delete(filters, 0, 0) # delite [0][]
	sample = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	filter_feature = np.empty(20)
	#extraction frog2
	if num == 0:
		for i in range(0, sample[0][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 1:
		for i in range(sample[0][0], sample[0][0]+sample[1][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 2:
		for i in range(sample[0][0]+sample[1][0], sample[0][0]+sample[1][0]+sample[2][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 3:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 4:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]+sample[4][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	#extraction back2
	if num == 0:
		for i in range(0, sample[0][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 1:
		for i in range(sample[0][1], sample[0][1]+sample[1][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 2:
		for i in range(sample[0][1]+sample[1][1], sample[0][1]+sample[1][1]+sample[2][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 3:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 4:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]+sample[4][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	sample = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	#extraction frog1
	if num == 0:
		for i in range(0, sample[0][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 1:
		for i in range(sample[0][0], sample[0][0]+sample[1][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 2:
		for i in range(sample[0][0]+sample[1][0], sample[0][0]+sample[1][0]+sample[2][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 3:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 4:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]+sample[4][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	#extraction back1
	if num == 0:
		for i in range(0, sample[0][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 1:
		for i in range(sample[0][1], sample[0][1]+sample[1][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 2:
		for i in range(sample[0][1]+sample[1][1], sample[0][1]+sample[1][1]+sample[2][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 3:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 4:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]+sample[4][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))

	filter_feature = np.delete(filter_feature, obj=0, axis=0)
	sample = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	y_test1 = np.ones(sample[num][0], dtype=int)
	y_test2 = np.zeros(sample[num][1], dtype=int)
	y_test1 = np.hstack((y_test1, y_test2))
	sample = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	y_test2 = np.zeros(sample[num][0]+sample[num][1], dtype=int)
	y_test1 = np.hstack((y_test1, y_test2))
	y_test = np.reshape(y_test1, (y_test1.shape[0], 1))

	filter_feature = np.hstack((y_test, filter_feature))
	with open('select_filters/frog_data/experiment2/experiment2_test_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(filter_feature, train_mfcc_pkl, protocol=2)
	del filter_feature
	print('finished:'+str(num))

def experiment1_test_select(num):
	fs = 44100
	nfft = 2048
	index_max = int(nfft / 2)
	with open('select_filters/frog_data/experiment1/experiment1_train_sort_'+str(num)+'.pkl', 'rb') as training_sort_pkl:
		training_sort = pickle.load(training_sort_pkl)
	_1, para_hz, _2 = np.hsplit(training_sort, [2, 4])
	del training_sort
	del _1
	del _2
	para_hz, _ = np.vsplit(para_hz, [20])
	filters = np.empty(index_max)
	for i in range(20):
		filterbank1 = one_melFilterBank(fs, nfft, para_hz[i][0], para_hz[i][1])
		filters = np.vstack((filters, filterbank1))
	filters = np.delete(filters, 0, 0) # delite [0][]
	sample = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	filter_feature = np.empty(20)
	#extraction frog1
	if num == 0:
		for i in range(0, sample[0][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 1:
		for i in range(sample[0][0], sample[0][0]+sample[1][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 2:
		for i in range(sample[0][0]+sample[1][0], sample[0][0]+sample[1][0]+sample[2][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 3:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 4:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]+sample[4][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	#extraction back1
	if num == 0:
		for i in range(0, sample[0][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 1:
		for i in range(sample[0][1], sample[0][1]+sample[1][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 2:
		for i in range(sample[0][1]+sample[1][1], sample[0][1]+sample[1][1]+sample[2][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 3:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 4:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]+sample[4][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	sample = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	#extraction frog2
	if num == 0:
		for i in range(0, sample[0][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 1:
		for i in range(sample[0][0], sample[0][0]+sample[1][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 2:
		for i in range(sample[0][0]+sample[1][0], sample[0][0]+sample[1][0]+sample[2][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 3:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 4:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]+sample[4][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	#extraction back2
	if num == 0:
		for i in range(0, sample[0][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 1:
		for i in range(sample[0][1], sample[0][1]+sample[1][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 2:
		for i in range(sample[0][1]+sample[1][1], sample[0][1]+sample[1][1]+sample[2][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 3:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 4:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]+sample[4][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))

	filter_feature = np.delete(filter_feature, obj=0, axis=0)
	sample = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	y_test1 = np.ones(sample[num][0], dtype=int)
	y_test2 = np.zeros(sample[num][1], dtype=int)
	y_test1 = np.hstack((y_test1, y_test2))
	sample = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	y_test2 = np.zeros(sample[num][0]+sample[num][1], dtype=int)
	y_test1 = np.hstack((y_test1, y_test2))
	y_test = np.reshape(y_test1, (y_test1.shape[0], 1))

	filter_feature = np.hstack((y_test, filter_feature))
	with open('select_filters/frog_data/experiment1/experiment1_test_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(filter_feature, train_mfcc_pkl, protocol=2)
	del filter_feature
	print('finished:'+str(num))

def experiment4_test_select(num):
	fs = 44100
	nfft = 2048
	index_max = int(nfft / 2)
	with open('select_filters/frog_data/experiment4/experiment4_train_sort_'+str(num)+'.pkl', 'rb') as training_sort_pkl:
		training_sort = pickle.load(training_sort_pkl)
	_1, para_hz, _2 = np.hsplit(training_sort, [2, 4])
	del training_sort
	del _1
	del _2
	para_hz, _ = np.vsplit(para_hz, [20])
	filters = np.empty(index_max)
	for i in range(20):
		filterbank1 = one_melFilterBank(fs, nfft, para_hz[i][0], para_hz[i][1])
		filters = np.vstack((filters, filterbank1))
	filters = np.delete(filters, 0, 0) # delite [0][]
	sample = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	filter_feature = np.empty(20)
	#extraction frog2
	if num == 0:
		for i in range(0, sample[0][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 1:
		for i in range(sample[0][0], sample[0][0]+sample[1][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 2:
		for i in range(sample[0][0]+sample[1][0], sample[0][0]+sample[1][0]+sample[2][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 3:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 4:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]+sample[4][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	#extraction back2
	if num == 0:
		for i in range(0, sample[0][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 1:
		for i in range(sample[0][1], sample[0][1]+sample[1][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 2:
		for i in range(sample[0][1]+sample[1][1], sample[0][1]+sample[1][1]+sample[2][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 3:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 4:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]+sample[4][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	sample = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	#extraction frog1
	if num == 0:
		for i in range(0, sample[0][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 1:
		for i in range(sample[0][0], sample[0][0]+sample[1][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 2:
		for i in range(sample[0][0]+sample[1][0], sample[0][0]+sample[1][0]+sample[2][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 3:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 4:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]+sample[4][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	#extraction back1
	if num == 0:
		for i in range(0, sample[0][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 1:
		for i in range(sample[0][1], sample[0][1]+sample[1][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 2:
		for i in range(sample[0][1]+sample[1][1], sample[0][1]+sample[1][1]+sample[2][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 3:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 4:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]+sample[4][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))

	filter_feature = np.delete(filter_feature, obj=0, axis=0)
	sample = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	y_test1 = np.ones(sample[num][0], dtype=int)
	y_test2 = np.zeros(sample[num][1], dtype=int)
	y_test1 = np.hstack((y_test1, y_test2))
	sample = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	y_test2 = np.zeros(sample[num][0]+sample[num][1], dtype=int)
	y_test1 = np.hstack((y_test1, y_test2))
	y_test = np.reshape(y_test1, (y_test1.shape[0], 1))

	filter_feature = np.hstack((y_test, filter_feature))
	with open('select_filters/frog_data/experiment4/experiment4_test_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(filter_feature, train_mfcc_pkl, protocol=2)
	del filter_feature
	print('finished:'+str(num))

def experiment3_test_select(num):
	fs = 44100
	nfft = 2048
	index_max = int(nfft / 2)
	with open('select_filters/frog_data/experiment3/experiment3_train_sort_'+str(num)+'.pkl', 'rb') as training_sort_pkl:
		training_sort = pickle.load(training_sort_pkl)
	_1, para_hz, _2 = np.hsplit(training_sort, [2, 4])
	del training_sort
	del _1
	del _2
	para_hz, _ = np.vsplit(para_hz, [20])
	filters = np.empty(index_max)
	for i in range(20):
		filterbank1 = one_melFilterBank(fs, nfft, para_hz[i][0], para_hz[i][1])
		filters = np.vstack((filters, filterbank1))
	filters = np.delete(filters, 0, 0) # delite [0][]
	sample = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	filter_feature = np.empty(20)
	#extraction frog1
	if num == 0:
		for i in range(0, sample[0][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 1:
		for i in range(sample[0][0], sample[0][0]+sample[1][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 2:
		for i in range(sample[0][0]+sample[1][0], sample[0][0]+sample[1][0]+sample[2][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 3:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 4:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]+sample[4][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	#extraction back1
	if num == 0:
		for i in range(0, sample[0][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 1:
		for i in range(sample[0][1], sample[0][1]+sample[1][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 2:
		for i in range(sample[0][1]+sample[1][1], sample[0][1]+sample[1][1]+sample[2][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 3:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 4:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]+sample[4][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	sample = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	#extraction frog2
	if num == 0:
		for i in range(0, sample[0][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 1:
		for i in range(sample[0][0], sample[0][0]+sample[1][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 2:
		for i in range(sample[0][0]+sample[1][0], sample[0][0]+sample[1][0]+sample[2][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 3:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 4:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]+sample[4][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	#extraction back2
	if num == 0:
		for i in range(0, sample[0][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 1:
		for i in range(sample[0][1], sample[0][1]+sample[1][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 2:
		for i in range(sample[0][1]+sample[1][1], sample[0][1]+sample[1][1]+sample[2][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 3:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))
	if num == 4:
		for i in range(sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]+sample[4][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			filter_feature1 = wav_stft_filter(x, nfft, filters)
			filter_feature = np.vstack((filter_feature, filter_feature1))

	filter_feature = np.delete(filter_feature, obj=0, axis=0)
	sample = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	y_test1 = np.ones(sample[num][0], dtype=int)
	y_test2 = np.zeros(sample[num][1], dtype=int)
	y_test1 = np.hstack((y_test1, y_test2))
	sample = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	y_test2 = np.zeros(sample[num][0]+sample[num][1], dtype=int)
	y_test1 = np.hstack((y_test1, y_test2))
	y_test = np.reshape(y_test1, (y_test1.shape[0], 1))

	filter_feature = np.hstack((y_test, filter_feature))
	with open('select_filters/frog_data/experiment3/experiment3_test_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(filter_feature, train_mfcc_pkl, protocol=2)
	del filter_feature
	print('finished:'+str(num))





if __name__ == '__main__':
	for num in range(5):
		experiment3_test_select(num)
	for num in range(5):
		experiment4_test_select(num)