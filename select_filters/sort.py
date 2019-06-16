import numpy as np
import pickle

def threshold_sort(training_threshold):
	training_sort = training_threshold[np.argsort(training_threshold[:, 0])[::-1]] # sort according to accuracy
	return training_sort

def experiment2_sort(num):
	with open('select_filters/frog_data/experiment2/experiment2_train_threshold_'+str(num)+'.pkl', 'rb') as filters_pkl:
		train_threshold = pickle.load(filters_pkl)
	train_threshold = threshold_sort(train_threshold)
	with open('select_filters/frog_data/experiment2/experiment2_train_sort_'+str(num)+'.pkl', 'wb') as training_sort_pkl:
		pickle.dump(train_threshold , training_sort_pkl)
	print('finished:'+str(num))

def experiment1_sort(num):
	with open('select_filters/frog_data/experiment1/experiment1_train_threshold_'+str(num)+'.pkl', 'rb') as filters_pkl:
		train_threshold = pickle.load(filters_pkl)
	train_threshold = threshold_sort(train_threshold)
	with open('select_filters/frog_data/experiment1/experiment1_train_sort_'+str(num)+'.pkl', 'wb') as training_sort_pkl:
		pickle.dump(train_threshold , training_sort_pkl)
	print('finished:'+str(num))

def experiment4_sort(num):
	with open('select_filters/frog_data/experiment4/experiment4_train_threshold_'+str(num)+'.pkl', 'rb') as filters_pkl:
		train_threshold = pickle.load(filters_pkl)
	train_threshold = threshold_sort(train_threshold)
	with open('select_filters/frog_data/experiment4/experiment4_train_sort_'+str(num)+'.pkl', 'wb') as training_sort_pkl:
		pickle.dump(train_threshold , training_sort_pkl)
	print('finished:'+str(num))

def experiment3_sort(num):
	with open('select_filters/frog_data/experiment3/experiment3_train_threshold_'+str(num)+'.pkl', 'rb') as filters_pkl:
		train_threshold = pickle.load(filters_pkl)
	train_threshold = threshold_sort(train_threshold)
	with open('select_filters/frog_data/experiment3/experiment3_train_sort_'+str(num)+'.pkl', 'wb') as training_sort_pkl:
		pickle.dump(train_threshold , training_sort_pkl)
	print('finished:'+str(num))




if __name__ == '__main__':
	for i in range(5):
		experiment4_sort(i)
	for i in range(5):
		experiment3_sort(i)
