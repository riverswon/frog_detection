import numpy as np
import scipy.fftpack
import pickle
from excel import load_excel_np

def dct(filter_data):
	cross_vali_sample = filter_data.shape[0]
	filter_sample = filter_data.shape[1]
	dct_data = np.empty((cross_vali_sample, filter_sample))
	for i in range(cross_vali_sample):
		dct_data[i] = scipy.fftpack.realtransforms.dct(filter_data[i], type=2, norm="ortho", axis=-1)
	return dct_data

def dim_norm1(features):
	sample = features.shape[0]
	for i in range(sample):
		features_norm = np.linalg.norm(features[i])
		features[i] = features[i] / features_norm
	return features


def experiment1_dct(num):
	with open('select_filters/frog_data/experiment1/experiment1_train_sort_'+str(num)+'.pkl', 'rb') as train_pkl:
		train_data = pickle.load(train_pkl)
	with open('select_filters/frog_data/experiment1/experiment1_test_'+str(num)+'.pkl', 'rb') as test_pkl:
		test_data = pickle.load(test_pkl)
	_, train_data = np.hsplit(train_data, [4])
	train_data, _ = np.vsplit(train_data, [20])
	train_data = train_data.T
	y_test, test_data = np.hsplit(test_data, [1])

	train_data = dct(train_data)
	train_data, _ = np.hsplit(train_data, [13])
	sample = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	sample1 = 0
	for i in range(5):
		if i == num:
			continue
		sample1 += sample[i][0]
	y_train1 = np.ones(sample1, dtype=int)
	sample1 = 0
	for i in range(5):
		if i == num:
			continue
		sample1 += sample[i][1]
	y_train2 = np.zeros(sample1, dtype=int)
	y_train1 = np.hstack((y_train1, y_train2))
	sample = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	sample1 = 0
	for i in range(5):
		if i == num:
			continue
		sample1 = sample1 + sample[i][0] + sample[i][1]
	y_train2 = np.zeros(sample1, dtype=int)
	y_train1 = np.hstack((y_train1, y_train2))
	y_train = np.reshape(y_train1, (y_train1.shape[0], 1))
	train_data = np.hstack((y_train, train_data))
	
	with open('select_filters/frog_data/experiment1/experiment1_train_dct_'+str(num)+'.pkl', 'wb') as training_dct_pkl:
		pickle.dump(train_data , training_dct_pkl)
	del train_data

	test_data = dct(test_data)
	test_data, _ = np.hsplit(test_data, [13])
	test_data = np.hstack((y_test, test_data))

	with open('select_filters/frog_data/experiment1/experiment1_test_dct_'+str(num)+'.pkl', 'wb') as test_dct_pkl:
		pickle.dump(test_data , test_dct_pkl)

def experiment2_dct(num):
	with open('select_filters/frog_data/experiment2/experiment2_train_sort_'+str(num)+'.pkl', 'rb') as train_pkl:
		train_data = pickle.load(train_pkl)
	with open('select_filters/frog_data/experiment2/experiment2_test_'+str(num)+'.pkl', 'rb') as test_pkl:
		test_data = pickle.load(test_pkl)
	_, train_data = np.hsplit(train_data, [4])
	train_data, _ = np.vsplit(train_data, [20])
	train_data = train_data.T
	y_test, test_data = np.hsplit(test_data, [1])

	train_data = dct(train_data)
	train_data, _ = np.hsplit(train_data, [13])
	sample = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	sample1 = 0
	for i in range(5):
		if i == num:
			continue
		sample1 += sample[i][0]
	y_train1 = np.ones(sample1, dtype=int)
	sample1 = 0
	for i in range(5):
		if i == num:
			continue
		sample1 += sample[i][1]
	y_train2 = np.zeros(sample1, dtype=int)
	y_train1 = np.hstack((y_train1, y_train2))
	sample = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	sample1 = 0
	for i in range(5):
		if i == num:
			continue
		sample1 = sample1 + sample[i][0] + sample[i][1]
	y_train2 = np.zeros(sample1, dtype=int)
	y_train1 = np.hstack((y_train1, y_train2))
	y_train = np.reshape(y_train1, (y_train1.shape[0], 1))
	train_data = np.hstack((y_train, train_data))
	with open('select_filters/frog_data/experiment2/experiment2_train_dct_'+str(num)+'.pkl', 'wb') as training_dct_pkl:
		pickle.dump(train_data , training_dct_pkl)
	del train_data
	
	test_data = dct(test_data)
	test_data, _ = np.hsplit(test_data, [13])
	test_data = np.hstack((y_test, test_data))
	with open('select_filters/frog_data/experiment2/experiment2_test_dct_'+str(num)+'.pkl', 'wb') as test_dct_pkl:
		pickle.dump(test_data , test_dct_pkl)

def experiment3_dct(num):
	with open('select_filters/frog_data/experiment3/experiment3_train_sort_'+str(num)+'.pkl', 'rb') as train_pkl:
		train_data = pickle.load(train_pkl)
	with open('select_filters/frog_data/experiment3/experiment3_test_'+str(num)+'.pkl', 'rb') as test_pkl:
		test_data = pickle.load(test_pkl)
	_, train_data = np.hsplit(train_data, [4])
	train_data, _ = np.vsplit(train_data, [20])
	train_data = train_data.T
	y_test, test_data = np.hsplit(test_data, [1])

	train_data = dct(train_data)
	train_data, _ = np.hsplit(train_data, [13])
	sample = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	sample1 = 0
	for i in range(5):
		if i == num:
			continue
		sample1 += sample[i][0]
	y_train1 = np.ones(sample1, dtype=int)
	sample1 = 0
	for i in range(5):
		if i == num:
			continue
		sample1 += sample[i][1]
	y_train2 = np.zeros(sample1, dtype=int)
	y_train1 = np.hstack((y_train1, y_train2))
	sample = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	sample1 = 0
	for i in range(5):
		if i == num:
			continue
		sample1 = sample1 + sample[i][1]
	y_train2 = np.zeros(sample1, dtype=int)
	y_train1 = np.hstack((y_train1, y_train2))
	y_train = np.reshape(y_train1, (y_train1.shape[0], 1))
	train_data = np.hstack((y_train, train_data))
	
	with open('select_filters/frog_data/experiment3/experiment3_train_dct_'+str(num)+'.pkl', 'wb') as training_dct_pkl:
		pickle.dump(train_data , training_dct_pkl)
	del train_data

	test_data = dct(test_data)
	test_data, _ = np.hsplit(test_data, [13])
	test_data = np.hstack((y_test, test_data))

	with open('select_filters/frog_data/experiment3/experiment3_test_dct_'+str(num)+'.pkl', 'wb') as test_dct_pkl:
		pickle.dump(test_data , test_dct_pkl)

def experiment4_dct(num):
	with open('select_filters/frog_data/experiment4/experiment4_train_sort_'+str(num)+'.pkl', 'rb') as train_pkl:
		train_data = pickle.load(train_pkl)
	with open('select_filters/frog_data/experiment4/experiment4_test_'+str(num)+'.pkl', 'rb') as test_pkl:
		test_data = pickle.load(test_pkl)
	_, train_data = np.hsplit(train_data, [4])
	train_data, _ = np.vsplit(train_data, [20])
	train_data = train_data.T
	y_test, test_data = np.hsplit(test_data, [1])

	train_data = dct(train_data)
	train_data, _ = np.hsplit(train_data, [13])
	sample = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	sample1 = 0
	for i in range(5):
		if i == num:
			continue
		sample1 += sample[i][0]
	y_train1 = np.ones(sample1, dtype=int)
	sample1 = 0
	for i in range(5):
		if i == num:
			continue
		sample1 += sample[i][1]
	y_train2 = np.zeros(sample1, dtype=int)
	y_train1 = np.hstack((y_train1, y_train2))
	sample = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	sample1 = 0
	for i in range(5):
		if i == num:
			continue
		sample1 = sample1 + sample[i][1]
	y_train2 = np.zeros(sample1, dtype=int)
	y_train1 = np.hstack((y_train1, y_train2))
	y_train = np.reshape(y_train1, (y_train1.shape[0], 1))
	train_data = np.hstack((y_train, train_data))
	with open('select_filters/frog_data/experiment4/experiment4_train_dct_'+str(num)+'.pkl', 'wb') as training_dct_pkl:
		pickle.dump(train_data , training_dct_pkl)
	del train_data
	
	test_data = dct(test_data)
	test_data, _ = np.hsplit(test_data, [13])
	test_data = np.hstack((y_test, test_data))
	with open('select_filters/frog_data/experiment4/experiment4_test_dct_'+str(num)+'.pkl', 'wb') as test_dct_pkl:
		pickle.dump(test_data , test_dct_pkl)






def frog_dct(in_train_path, in_test_path, out_train_path, out_test_path, filter_sample, preprocessing2):
	cross_validation = 5
	with open(in_train_path, 'rb') as training_sort_pkl:
		training_sort_x = pickle.load(training_sort_pkl)
		with open(in_test_path, 'rb') as test_sort_pkl:
			test_sort_x = pickle.load(test_sort_pkl)

			training_dct = []
			test_dct = []
			for i in range(cross_validation):
				# transform mel into dct in training_data
				training_sort, training_sort_below = np.vsplit(training_sort_x[i], [filter_sample]) # extract the 20 upper training_one_mel
				para, training_sort = np.hsplit(training_sort, [4])
				training_sort = training_sort.T # change [20][sample] for [sample][20]
				training_dct1 = dct(training_sort)
				if preprocessing2 == 'norm1':
					training_dct1 = dim_norm1(training_dct1)
				# transform mel into dct in test_data
				test_sort = test_sort_x[i].T # change [20][sample] for [sample][20]
				test_dct1 = dct(test_sort)
				if preprocessing2 == 'norm1':
					test_dct1 = dim_norm1(test_dct1)
				training_dct.append(training_dct1)
				test_dct.append(test_dct1)
			
			with open(out_train_path, 'wb') as training_dct_pkl:
				pickle.dump(training_dct , training_dct_pkl)
			with open(out_test_path, 'wb') as test_dct_pkl:
				pickle.dump(test_dct , test_dct_pkl)
	print('finished:frog_dct')

if __name__ == '__main__':
	for i in range(5):
		experiment3_dct(i)
	for i in range(5):
		experiment4_dct(i)