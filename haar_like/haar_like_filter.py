import numpy as np
import pickle

def one_haarlike_filter(fs, nfft, f_start, f_stop): # make a haar_like_filter (minus plus minus)
	index_max = nfft / 2 # the number of index (remove mirror)
	df = fs / nfft # the interval Hz by frequency index
	index_start = np.round(f_start / df) # convert frequency into index in start
	index_stop = np.round(f_stop / df) # convert frequency into index in stop
	f_difference = (f_stop - f_start) / 3
	index_middle1 = np.round((f_start + f_difference) / df)
	index_middle2 = np.round((f_stop - f_difference) / df)

	#calculate white,black,white (white=minus,black=plus)
	filterbank = np.zeros(int(index_max), dtype=int)
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
	filterbanks = np.empty(2 + index_max, dtype=int)
	for width in range(100, 6000, 200):
		for num in range(0, f_max - width, 50):
			filterbank1 = one_haarlike_filter(fs, nfft, num, num + width)
			start_stop = np.array([num, num + width])
			filterbank1 = np.hstack((start_stop, filterbank1))
			filterbanks = np.vstack((filterbanks, filterbank1))
		print('finished:width=', width)
	filterbanks = np.delete(filterbanks, 0, 0) # delite [0][]
	#save filterbanks in pkl
	with open('haar_like/frog_data/filters.pkl', 'wb') as filters_pkl:
		pickle.dump(filterbanks , filters_pkl)
	print('finished:make_haarlike_filters')





