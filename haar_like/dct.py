import numpy as np
import pickle

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

if __name__ == '__main__':
	in_train_path = 'select_filters/frog_data/training_sort2.pkl'
	in_test_path = 'select_filters/frog_data/test_sort2.pkl'
	out_train_path = 'select_filters/frog_data/training_dct2.pkl'
	out_test_path = 'select_filters/frog_data/test_dct2.pkl'
	filter_sample = 20
	frog_dct(in_train_path, in_test_path, out_train_path, out_test_path, filter_sample)