import librosa
import numpy as np
import scipy.signal
import pickle

#それぞれ特徴量は12次元で固定

#save in Excel
import pandas as pd
def np_excel(sheet, path):
	df = pd.DataFrame(sheet)
	df.to_excel(path)
def load_excel_np(path):
	df = pd.read_excel(path)
	df = df.values
	return df

#次元削減
from sklearn.decomposition import PCA
def dim_pca(dct_training, dct_test, p, n):
	y_train = np.zeros(p+n)
	#training_data:set 0 or 1
	y_train[:p] = 1
	#to 2 dimesion by PCA
	pca = PCA(n_components = 2)
	pca_training = pca.fit_transform(dct_training, y_train)
	pca_test = pca.transform(dct_test)
	return pca_training, pca_test
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
def dim_lda(dct_training, dct_test, p, n):
	y_train = np.zeros(p+n)
	#training_data:set 0 or 1
	y_train[:p] = 1
	lda = LDA(n_components = 1)
	lda_training = lda.fit_transform(dct_training, y_train)
	lda_test = lda.transform(dct_test)
	return lda_training, lda_test
def frog_dim_reduction(train_dct_x, test_dct_x, way, sample):
	cross_validation = 5
	train_dim_reduction = []
	test_dim_reduction = []
	print('finished:frog_dim_reduction_'+way)
	if way == 'pca':
		for i in range(cross_validation):
			train_pca1, test_pca1 = dim_pca(train_dct_x[i], test_dct_x[i], sample[i][0], sample[i][1])
			train_dim_reduction.append(train_pca1)
			test_dim_reduction.append(test_pca1)
		return train_dim_reduction, test_dim_reduction
	elif way == 'lda':
		for i in range(cross_validation):
			train_lda1, test_lda1 = dim_lda(train_dct_x[i], test_dct_x[i], sample[i][0], sample[i][1])
			train_dim_reduction.append(train_lda1)
			test_dim_reduction.append(test_lda1)
		return train_dim_reduction, test_dim_reduction

#学習データの特徴量をプロット
import matplotlib.pyplot as plt
def training_data_plot(feature_method, train_dim_data, sample, way, name):
	if way == 'pca':
		x_max = max(train_dim_data[:,0]) + (max(train_dim_data[:,0]) - min(train_dim_data[:,0])) / 10
		x_min = min(train_dim_data[:,0]) - (max(train_dim_data[:,0]) - min(train_dim_data[:,0])) / 10
		y_max = max(train_dim_data[:,1]) + (max(train_dim_data[:,1]) - min(train_dim_data[:,1])) / 10
		y_min = min(train_dim_data[:,1]) - (max(train_dim_data[:,1]) - min(train_dim_data[:,1])) / 10
		plt.figure()
		plt.title(name+'_fg')
		plt.xlim(x_min, x_max)
		plt.ylim(y_min, y_max)
		plt.scatter(x=train_dim_data[:sample[0],0], y=train_dim_data[:sample[0],1], color="blue", marker="o", label="fg")# fg...o,blue
		plt.legend() # 凡例を表示
		plt.savefig(feature_method+'/pictures/'+name+'_fg')
		plt.clf()
		plt.title(name+'_no')
		plt.xlim(x_min, x_max)
		plt.ylim(y_min, y_max)
		plt.scatter(x=train_dim_data[sample[0]:,0], y=train_dim_data[sample[0]:,1], color="red", marker="x", label="no") # no...x,red
		plt.legend() # 凡例を表示
		plt.savefig(feature_method+'/pictures/'+name+'_no')
		plt.clf()
		plt.title(name)
		plt.xlim(x_min, x_max)
		plt.ylim(y_min, y_max)
		plt.scatter(x=train_dim_data[:sample[0],0], y=train_dim_data[:sample[0],1], color="blue", marker="o", label="fg")# fg...o,blue
		plt.scatter(x=train_dim_data[sample[0]:,0], y=train_dim_data[sample[0]:,1], color="red", marker="x", label="no") # no...x,red
		plt.legend() # 凡例を表示
		plt.savefig(feature_method+'/pictures/'+name)
		plt.close()
	elif way == 'lda':
		x_max = max(train_dim_data) + (max(train_dim_data) - min(train_dim_data)) / 10
		x_min = min(train_dim_data) - (max(train_dim_data) - min(train_dim_data)) / 10
		plt.figure()
		plt.title(name+'_fg')
		plt.xlim(x_min, x_max)
		y_train = np.zeros(sample[0])
		plt.scatter(x=train_dim_data[:sample[0]], y=y_train, color="blue", marker="o", label="fg")# fg...o,blue
		plt.legend() # 凡例を表示
		plt.savefig(feature_method+'/pictures/'+name+'_fg')
		plt.clf()
		plt.title(name+'_no')
		plt.xlim(x_min, x_max)
		y_train = np.zeros(sample[1])
		plt.scatter(x=train_dim_data[sample[0]:], y=y_train, color="red", marker="x", label="no") # no...x,red
		plt.legend() # 凡例を表示
		plt.savefig(feature_method+'/pictures/'+name+'_no')
		plt.clf()
		plt.title(name)
		plt.xlim(x_min, x_max)
		y_train = np.zeros(sample[0])
		plt.scatter(x=train_dim_data[:sample[0]], y=y_train, color="blue", marker="o", label="fg")# fg...o,blue
		y_train = np.zeros(sample[1])
		plt.scatter(x=train_dim_data[sample[0]:], y=y_train, color="red", marker="x", label="no") # no...x,red
		plt.legend() # 凡例を表示
		plt.savefig(feature_method+'/pictures/'+name)
		plt.close()
def select_filter_data_plot(feature_method, training_sort, sample, name):
	plt.figure()
	for i in range(6):	
		plt.title(name+':'+str(training_sort[i][2])+"[Hz]-"+str(training_sort[i][3])+"[Hz]:"+str(round(training_sort[i][0],3)))
		y_train = np.zeros(sample[0])
		plt.scatter(x=training_sort[i,4:4+sample[0]], y=y_train, color="blue", marker="o", label="fg") # fg...o,blue,0
		y_train = np.ones(sample[1])
		plt.scatter(x=training_sort[i,4+sample[0]:], y=y_train, color="red", marker="x", label="no") # no...x,red,1
		plt.axvline(training_sort[i][1], color="green") # threshold...green
		plt.legend() # 凡例を表示	
		plt.savefig(feature_method+'/pictures/'+name+'_'+str(i+1))
		plt.clf()
	plt.close()

#分類器作成
from scipy.stats import norm
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
def linear_svm(train_data, test_data, sample):
	y_train = np.zeros(sample[0]+sample[1], int)
	y_train[:sample[0]] = 1
	#学習データから最適パラメータの導出
	para = {'C': np.logspace(-10, 9, 20, base=2.0), 'kernel': ['linear']}
	clf = GridSearchCV(SVC(), para, cv=5, scoring='f1')
	clf.fit(train_data, y_train)
	para_list = list(clf.best_params_.values())
	#評価データをフィッティング
	model = SVC(C=para_list[0], kernel=para_list[1])
	model.fit(train_data, y_train)
	predict = model.predict(test_data)
	return predict
def normal_distribution_predict(train_data, test_data, way, sample):
	if way == 'pca':
		pca_dim = 2 # the number of dimension by pca
		split_sample = test_data.shape[0] # the number of split

		mean = np.empty((2, pca_dim))
		sigma = np.empty(((2, pca_dim, pca_dim)))
		mean[0] = np.mean(train_data[:sample[0]], axis=0)
		mean[1] = np.mean(train_data[sample[0]:], axis=0)
		sigma[0] = np.cov(train_data[:sample[0]], rowvar=0, bias=1)
		sigma[1] = np.cov(train_data[sample[0]:], rowvar=0, bias=1)
		
		predict = np.zeros(split_sample, dtype=int)
		f_fg = lambda x, y: stats.multivariate_normal(mean[0], sigma[0]).pdf([x, y])
		f_no = lambda x, y: stats.multivariate_normal(mean[1], sigma[1]).pdf([x, y])
		for i in range(split_sample):
			fg_norm = np.vectorize(f_fg)(test_data[i][0], test_data[i][1])
			no_norm = np.vectorize(f_no)(test_data[i][0], test_data[i][1])
			judge = fg_norm - no_norm
			if judge >= 0:
				predict[i] = 1
		return predict
	elif way =='lda':
		split_sample = test_data.shape[0] # the number of split
		
		mean = np.empty(2)
		sigma = np.empty(2)
		mean[0] = np.mean(train_data[:sample[0]])
		mean[1] = np.mean(train_data[sample[0]:])
		sigma[0] = np.var(train_data[:sample[0]])
		sigma[1] = np.var(train_data[sample[0]:])
		
		predict = np.zeros(split_sample, dtype=int)
		for i in range(split_sample):
			fg_norm = norm.pdf(x=test_data[i], loc=mean[0], scale=sigma[0])
			no_norm = norm.pdf(x=test_data[i], loc=mean[1], scale=sigma[1])
			judge = fg_norm - no_norm
			if judge >= 0:
				predict[i] = 1
		return predict
def frog_classifier(train_data_x, test_data_x, out_predict_path, way, sample, wav_time, section):
	cross_validation = 5
	split_sample = int(wav_time / section)
	predict = np.empty(split_sample, dtype=int)
	if way == 'svm':
		for i in range(cross_validation):
			predict1 = linear_svm(train_data_x[i], test_data_x[i], sample[i])
			predict = np.vstack((predict, predict1))
		predict = np.delete(predict, obj=0, axis=0)
		np_excel(predict, out_predict_path)
		print('finished:predict_'+way)
	elif way == 'pca' or way =='lda':
		for i in range(cross_validation):
			predict1 = normal_distribution_predict(train_data_x[i], test_data_x[i], way, sample[i])
			predict = np.vstack((predict, predict1))
		predict = np.delete(predict, obj=0, axis=0)
		np_excel(predict, out_predict_path)
		print('finished:predict_'+way)

#評価を作成
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

##################################################################
#MFCCの特徴抽出
def wav_mfcc(x, fs, nfft):
	p = 0.97
	x = scipy.signal.lfilter([1.0, -p], 1, x)
	mfcc = librosa.feature.mfcc(x, sr=fs, n_mfcc=20, n_fft=nfft, hop_length=int(nfft/2))
	mfcc = np.mean(mfcc, axis = 1)
	mfcc = mfcc[:12]
	return mfcc
def frog_training_extract_mfcc(fs, nfft, wav_time, section):
	cross_vali_sample = 5
	split_sample = int(wav_time / section)
	split_interval = int(fs * section)
	evaluation = load_excel_np('evaluation/frog_sheet_'+str(section)+'s.xlsx')
	training_mfcc = []
	for i in range(cross_vali_sample):
		training_mfcc1 = np.empty(12)
		training_evaluation = np.empty(1, int)
		for j in range(cross_vali_sample):
			if i == j:
				continue
			x, f = librosa.load('test_data/frog'+str(j+1)+'.wav', sr=fs)
			training_mfcc2 = np.empty((split_sample, 12))
			for k in range(split_sample):
				x_split = x[k*split_interval:(k+1)*split_interval]
				training_mfcc2[k] = wav_mfcc(x_split, fs, nfft)
			training_mfcc1 = np.vstack((training_mfcc1, training_mfcc2))
			training_evaluation = np.hstack((training_evaluation,evaluation[j]))
		training_mfcc1 = np.delete(training_mfcc1, obj=0, axis=0)
		training_evaluation = training_evaluation[1:]
		training_evaluation = training_evaluation.reshape((split_sample*4, 1))
		training_mfcc1 = np.hstack((training_evaluation, training_mfcc1))
		training_mfcc1 = training_mfcc1[np.argsort(training_mfcc1[:, 0])[::-1]]
		training_mfcc1 = np.delete(training_mfcc1, obj=0, axis=1)
		training_mfcc.append(training_mfcc1)
	print('finished:train_extract_mfcc')
	return training_mfcc
def frog_test_extract_mfcc(fs, nfft, wav_time, section):
	cross_vali_sample = 5
	split_sample = int(wav_time / section)
	split_interval = int(fs * section)
	test_mfcc = []
	for i in range(cross_vali_sample):
		x, f = librosa.load('test_data/frog'+str(i+1)+'.wav', sr=fs)
		test_mfcc1 = np.empty((split_sample, 12))
		for j in range(split_sample):
			x_split = x[j*split_interval:(j+1)*split_interval]
			test_mfcc1[j] = wav_mfcc(x_split, fs, nfft)
		test_mfcc.append(test_mfcc1)
	print('finished:test_extract_mfcc')
	return test_mfcc

#MFCC特徴量を用いて音声区間検出
def VAD_MFCC():
	fs = 44100
	wav_time = 120
	nfft = 2048
	sections = [[0.1,0.5,1.0],['01','05','10']]

	for section, section_name in zip(sections[0], sections[1]):
		test_sample = load_excel_np('evaluation/test_sample_'+str(section)+'s.xlsx')
		#MFCCを抽出
		train_mfcc = frog_training_extract_mfcc(fs, nfft, wav_time, section)
		test_mfcc = frog_test_extract_mfcc(fs, nfft, wav_time, section)
		#図を作成
		train_pca, test_pca = frog_dim_reduction(train_mfcc, test_mfcc, 'pca', test_sample)
		name = section_name+'_mfcc'
		training_data_plot('mfcc', train_pca[0], test_sample[0], 'pca', name)
		#識別器で分類
		#SVM
		out_predict_path = 'mfcc/predict/predict_'+section_name+'_mfcc_svm.xlsx'
		frog_classifier(train_mfcc, test_mfcc, out_predict_path, 'svm', test_sample, wav_time, section)
		predict_path = 'mfcc/predict/predict_'+section_name+'_mfcc_svm.xlsx'
		answer_path = 'evaluation/frog_sheet_'+str(section)+'s.xlsx'
		result_path = 'mfcc/predict/result_'+section_name+'_mfcc_svm.xlsx'
		evaluate(predict_path, answer_path, result_path)
		#lda->normal distribution
		train_lda, test_lda = frog_dim_reduction(train_mfcc, test_mfcc, 'lda', test_sample)
		out_predict_path = 'mfcc/predict/predict_'+section_name+'_mfcc_lda.xlsx'
		frog_classifier(train_lda, test_lda, out_predict_path, 'lda', test_sample, wav_time, section)
		predict_path = 'mfcc/predict/predict_'+section_name+'_mfcc_lda.xlsx'
		answer_path = 'evaluation/frog_sheet_'+str(section)+'s.xlsx'
		result_path = 'mfcc/predict/result_'+section_name+'_mfcc_lda.xlsx'
		evaluate(predict_path, answer_path, result_path)
##################################################################

##################################################################
#select_filtersの特徴抽出
def hz_mel(f): # convert Hz into mel
	return 1127.01048 * np.log(f / 700.0 + 1.0)
def mel_hz(m): # convert mel into Hz
	return 700.0 * (np.exp(m / 1127.01048) - 1.0)
def one_melFilterBank(fs, nfft, f_start, f_stop): # make a mel-filter
	index_max = nfft / 2 # the number of index (remove mirror)
	df = fs / nfft # the interval Hz by frequency index
	# calculate the cernter frequency(mel)
	mel_start = hz_mel(f_start)
	mel_stop = hz_mel(f_stop)
	mel_center = (mel_start + mel_stop) / 2.0
	f_center = mel_hz(mel_center)
	index_center = np.round(f_center / df) # convert frequency into index in center
	index_start = np.round(f_start / df) # convert frequency into index in start
	index_stop = np.round(f_stop / df) # convert frequency into index in stop
	
	filterbank = np.zeros(int(index_max))
	# calculate the left of triangle filter
	increment = 1.0 / (index_center - index_start)
	for i in range(int(index_start), int(index_center)):
		filterbank[i] = (i - index_start) * increment
	# calculate the right of triangle filter
	decrement = 1.0 / (index_stop - index_center)
	for i in range(int(index_center), int(index_stop)):
		filterbank[i] = 1.0 - ((i - index_center) * decrement)
	
	return filterbank
def make_melfilters(fs, nfft, f_max):
	index_max = int(nfft / 2)

	# make mel-filters as width=200,300,...3000 num=0,100,..., 8000
	# start[Hz] stop[Hz] filter[0-fs/2[Hz]]
	filterbanks = np.empty(2 + index_max)
	for width in range(200, 3000, 100):
		for num in range(0, f_max - width, 100):
			filterbank1 = one_melFilterBank(fs, nfft, num, num + width)
			start_stop = np.array([num, num + width])
			filterbank1 = np.hstack((start_stop, filterbank1))
			filterbanks = np.vstack((filterbanks, filterbank1))
	filterbanks = np.delete(filterbanks, 0, 0) # delite [0][]
	#save filterbanks in pkl
	with open('select_filters/frog_data/filters_'+str(nfft)+'.pkl', 'wb') as filters_pkl:
		pickle.dump(filterbanks , filters_pkl)
	print('finished:make_melfilters')
def wav_stft_log_filter(x, nfft, filters):
	p = 0.97
	x = scipy.signal.lfilter([1.0, -p], 1, x)
	spec = np.abs(librosa.stft(x, n_fft=nfft, hop_length=int(nfft/2), window='hamming'))[:int(nfft/2)]
	filters_features = np.dot(filters, spec)
	filters_features = np.log10(filters_features)
	filters_features = np.mean(filters_features, axis=1)
	return filters_features
def frog_training_stft_log_filter(fs, nfft, wav_time, section):
	cross_validation = 5
	with open('select_filters/frog_data/filters_'+str(nfft)+'.pkl', 'rb') as filters_pkl:
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
					filters_features2 = wav_stft_log_filter(x_split, nfft, filters)
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
	print('finished:frog_training_stft_log_filter')
	return filters_features
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
def frog_set_threshold(filters_features, sample):
	cross_validation = 5    
	threshold_data = []
	for i in range(cross_validation):
		threshold_data1 = set_threshold(filters_features[i], sample[i])
		threshold_data.append(threshold_data1)
		print('finished:frog_set_threshold'+str(i+1))
	return threshold_data
def threshold_sort(training_threshold):
	training_sort = training_threshold[np.argsort(training_threshold[:, 0])[::-1]] # sort according to accuracy
	return training_sort
def frog_sort(training_threshold, out_train_path):
	cross_validation = 5
	training_sort = []
	for i in range(cross_validation):
		training_sort1 = threshold_sort(training_threshold[i])
		training_sort.append(training_sort1)
	with open(out_train_path, 'wb') as training_sort_pkl:
		pickle.dump(training_sort , training_sort_pkl)
	print('finished:frog_sort')
def frog_test_stft_log_filter(in_train_path, out_test_path, fs, nfft, wav_time, section):
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
			para_hz, no_use = np.vsplit(para_hz, [20])

			filterbanks = np.empty(index_max)
			for j in range(20):
				filterbank1 = one_melFilterBank(fs, nfft, para_hz[j][0], para_hz[j][1])
				filterbanks = np.vstack((filterbanks, filterbank1))
			filterbanks = np.delete(filterbanks, 0, 0) # delite [0][]

			x, f = librosa.load('test_data/frog'+str(i+1)+'.wav', sr=fs)
			filters_features1 = np.empty(20) # use the training_data stack
			for j in range(split_sample):
				x_split = x[j*split_interval:(j+1)*split_interval]
				filters_features2 = wav_stft_log_filter(x_split, nfft,filterbanks)
				filters_features1 = np.vstack((filters_features1, filters_features2)) # features[split_sample][filter_sample]
			filters_features1 = np.delete(filters_features1, 0, 0) # delite [0][]
			filters_features1 = filters_features1.T # features[filter_sample][split_sample]
			filters_features.append(filters_features1)
	with open(out_test_path, 'wb') as test_sort_pkl:
		pickle.dump(filters_features , test_sort_pkl)
	print('finished:frog_test_stft_log_filter')
import scipy.fftpack
def data_dct(filter_data):
	cross_vali_sample = filter_data.shape[0]
	filter_sample = filter_data.shape[1]
	dct_data = np.empty((cross_vali_sample, 12))
	for i in range(cross_vali_sample):
		dct_data1 = scipy.fftpack.realtransforms.dct(filter_data[i], type=2, norm="ortho", axis=-1)
		dct_data[i] = dct_data1[:12]
	return dct_data
def frog_dct(in_train_path, in_test_path):
	cross_validation = 5
	with open(in_train_path, 'rb') as training_sort_pkl:
		training_sort_x = pickle.load(training_sort_pkl)
		with open(in_test_path, 'rb') as test_sort_pkl:
			test_sort_x = pickle.load(test_sort_pkl)

			training_dct = []
			test_dct = []
			for i in range(cross_validation):
				# transform mel into dct in training_data
				training_sort, training_sort_below = np.vsplit(training_sort_x[i], [20]) # extract the 20 upper training_one_mel
				para, training_sort = np.hsplit(training_sort, [4])
				training_sort = training_sort.T # change [20][sample] for [sample][20]
				training_dct1 = data_dct(training_sort)
				# transform mel into dct in test_data
				test_sort = test_sort_x[i].T # change [20][sample] for [sample][20]
				test_dct1 = data_dct(test_sort)
				training_dct.append(training_dct1)
				test_dct.append(test_dct1)
	return training_dct, test_dct
	print('finished:frog_dct')

#select_filters特徴量を用いて音声区間検出
def VAD_SelectFilters():
	fs = 44100
	wav_time = 120
	nfft = 2048
	sections = [[0.1,0.5,1.0],['01','05','10']]
	f_max = 8000

	make_melfilters(fs, nfft, f_max)
	for section, section_name in zip(sections[0], sections[1]):
		test_sample = load_excel_np('evaluation/test_sample_'+str(section)+'s.xlsx')
		#select_filtersデータを抽出
		train_features = frog_training_stft_log_filter(fs, nfft, wav_time, section)
		train_threshold = frog_set_threshold(train_features, test_sample)
		out_train_path = 'select_filters/frog_data/'+section_name+'_train_sort.pkl'
		frog_sort(train_threshold, out_train_path)
		in_train_path = 'select_filters/frog_data/'+section_name+'_train_sort.pkl'
		out_test_path = 'select_filters/frog_data/'+section_name+'_test_sort.pkl'
		frog_test_stft_log_filter(in_train_path, out_test_path, fs, nfft, wav_time, section)
		#select_filtersデータdct
		in_train_path = 'select_filters/frog_data/'+section_name+'_train_sort.pkl'
		in_test_path = 'select_filters/frog_data/'+section_name+'_test_sort.pkl'
		train_dct, test_dct = frog_dct(in_train_path, in_test_path)
		#図を作成
		train_pca, test_pca = frog_dim_reduction(train_dct, test_dct, 'pca', test_sample)
		name = section_name+'_selectfilters_pca'
		training_data_plot('select_filters', train_pca[0], test_sample[0], 'pca', name)
		name = section_name+'_selectfilters'
		with open(in_train_path, 'rb') as training_sort_pkl:
			train_sort = pickle.load(training_sort_pkl)
			select_filter_data_plot('select_filters', train_sort[0], test_sample[0], name)
		#識別器で分類
		#SVM
		out_predict_path = 'select_filters/predict/predict_'+section_name+'_selectfilters_svm.xlsx'
		frog_classifier(train_dct, test_dct, out_predict_path, 'svm', test_sample, wav_time, section)
		predict_path = 'select_filters/predict/predict_'+section_name+'_selectfilters_svm.xlsx'
		answer_path = 'evaluation/frog_sheet_'+str(section)+'s.xlsx'
		result_path = 'select_filters/predict/result_'+section_name+'_selectfilters_svm.xlsx'
		evaluate(predict_path, answer_path, result_path)
		#lda->normal distribution
		train_lda, test_lda = frog_dim_reduction(train_dct, test_dct, 'lda', test_sample)
		out_predict_path = 'select_filters/predict/predict_'+section_name+'_selectfilters_lda.xlsx'
		frog_classifier(train_lda, test_lda, out_predict_path, 'lda', test_sample, wav_time, section)
		predict_path = 'select_filters/predict/predict_'+section_name+'_selectfilters_lda.xlsx'
		answer_path = 'evaluation/frog_sheet_'+str(section)+'s.xlsx'
		result_path = 'select_filters/predict/result_'+section_name+'_selectfilters_lda.xlsx'
		evaluate(predict_path, answer_path, result_path)
##################################################################

##################################################################
#Haar-like特徴量の抽出
def one_haarlike_filter(fs, nfft, f_start, f_stop): # make a haar_like_filter (minus plus minus)
	index_max = nfft / 2 # the number of index (remove mirror)
	df = fs / nfft # the interval Hz by frequency index
	index_start = np.round(f_start / df) # convert frequency into index in start
	index_stop = np.round(f_stop / df) # convert frequency into index in stop
	f_difference = (f_stop - f_start) / 3
	index_middle1 = np.round((f_start + f_difference) / df)
	index_middle2 = np.round((f_stop - f_difference) / df)

	#calculate white,black,white (white=minus,black=plus)
	filterbank = np.zeros(int(index_max))
	for i in range(int(index_start), int(index_middle1)):
		filterbank[i] = -1
	for i in range(int(index_middle1), int(index_middle2)):
		filterbank[i] = 1
	for i in range(int(index_middle2), int(index_stop)):
		filterbank[i] = -1
	
	return filterbank
def make_haarlike_filters(fs, nfft, f_max):
	index_max = int(nfft / 2)

	# make mel-filters as width=100,200,...3000 num=0,50,..., 8000
	# start[Hz] stop[Hz] filter[0-fs/2[Hz]]
	filterbanks = np.empty(2 + index_max)
	for width in range(100, 3000, 100):
		for num in range(0, f_max - width, 50):
			filterbank1 = one_haarlike_filter(fs, nfft, num, num + width)
			start_stop = np.array([num, num + width])
			filterbank1 = np.hstack((start_stop, filterbank1))
			filterbanks = np.vstack((filterbanks, filterbank1))
	filterbanks = np.delete(filterbanks, 0, 0) # delite [0][]
	#save filterbanks in pkl
	with open('haar_like/frog_data/filters_'+str(nfft)+'.pkl', 'wb') as filters_pkl:
		pickle.dump(filterbanks , filters_pkl)
	print('finished:make_haarlike_filters')
def wav_stft_filter(x, nfft, filters):
	p = 0.97
	x = scipy.signal.lfilter([1.0, -p], 1, x)
	spec = np.abs(librosa.stft(x, n_fft=nfft, hop_length=int(nfft/2), window='hamming'))[:int(nfft/2)]
	filters_features = np.dot(filters, spec)
	filters_features = np.mean(filters_features, axis=1)
	return filters_features
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
#set_threshold, frog_set_threshold, threshold_sort, frog_sortは同じ
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
def train_split(in_train_path):
	cross_validation = 5
	with open(in_train_path, 'rb') as training_sort_pkl:
		training_sort_x = pickle.load(training_sort_pkl)
		training_dct = []
		for i in range(cross_validation):
			# transform mel into dct in training_data
			training_sort, training_sort_below = np.vsplit(training_sort_x[i], [12]) # extract the 12 upper training_one_mel
			para, training_sort = np.hsplit(training_sort, [4])
			training_sort = training_sort.T # change [20][sample] for [sample][20]
			training_dct.append(training_sort)
	return training_dct
	print('finished:train_split')

#Haar-like特徴量を用いて音声区間検出
def VAD_HaarLike():
	fs = 44100
	wav_time = 120
	nfft = 2048
	sections = [[0.1,0.5,1.0],['01','05','10']]
	f_max = 8000

	make_haarlike_filters(fs, nfft, f_max)
	for section, section_name in zip(sections[0], sections[1]):
		test_sample = load_excel_np('evaluation/test_sample_'+str(section)+'s.xlsx')
		#haar_likeデータを抽出
		train_features = frog_training_stft_filter(fs, nfft, wav_time, section)
		train_threshold = frog_set_threshold(train_features, test_sample)
		out_train_path = 'haar_like/frog_data/'+section_name+'_train_sort.pkl'
		frog_sort(train_threshold, out_train_path)
		in_train_path = 'haar_like/frog_data/'+section_name+'_train_sort.pkl'
		out_test_path = 'haar_like/frog_data/'+section_name+'_test_sort.pkl'
		test_dct = frog_test_stft_filter(in_train_path, out_test_path, fs, nfft, wav_time, section)
		in_train_path = 'haar_like/frog_data/'+section_name+'_train_sort.pkl'
		train_dct = train_split(in_train_path)
		#図を作成
		train_pca, test_pca = frog_dim_reduction(train_dct, test_dct, 'pca', test_sample)
		name = section_name+'_haarlike_pca'
		training_data_plot('haar_like', train_pca[0], test_sample[0], 'pca', name)
		name = section_name+'_haarlike'
		with open(in_train_path, 'rb') as training_sort_pkl:
			train_sort = pickle.load(training_sort_pkl)
			select_filter_data_plot('haar_like', train_sort[0], test_sample[0], name)
		#識別器で分類
		#SVM
		out_predict_path = 'haar_like/predict/predict_'+section_name+'_haarlike_svm.xlsx'
		frog_classifier(train_dct, test_dct, out_predict_path, 'svm', test_sample, wav_time, section)
		predict_path = 'haar_like/predict/predict_'+section_name+'_haarlike_svm.xlsx'
		answer_path = 'evaluation/frog_sheet_'+str(section)+'s.xlsx'
		result_path = 'haar_like/predict/result_'+section_name+'_haarlike_svm.xlsx'
		evaluate(predict_path, answer_path, result_path)
		#lda->normal distribution
		train_lda, test_lda = frog_dim_reduction(train_dct, test_dct, 'lda', test_sample)
		out_predict_path = 'haar_like/predict/predict_'+section_name+'_haarlike_lda.xlsx'
		frog_classifier(train_lda, test_lda, out_predict_path, 'lda', test_sample, wav_time, section)
		predict_path = 'haar_like/predict/predict_'+section_name+'_haarlike_lda.xlsx'
		answer_path = 'evaluation/frog_sheet_'+str(section)+'s.xlsx'
		result_path = 'haar_like/predict/result_'+section_name+'_haarlike_lda.xlsx'
		evaluate(predict_path, answer_path, result_path)
##################################################################

if __name__ == "__main__":
	#VAD_MFCC()
	VAD_SelectFilters()
	#VAD_HaarLike()
