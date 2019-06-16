import numpy as np
import pickle
import matplotlib.pyplot as plt

def hz_mel(f): # convert Hz into mel
	return 1000 / np.log10(2) * np.log(f / 1000.0 + 1.0)

def mel_hz(m): # convert mel into Hz
	return 1000.0 * (np.exp(m * np.log10(2) / 1000) - 1.0)

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
	top = 100 / (f_stop - f_start)
	increment = top / (index_center - index_start)
	for i in range(int(index_start), int(index_center)):
		filterbank[i] = (i - index_start) * increment
	# calculate the right of triangle filter
	decrement = top / (index_stop - index_center)
	for i in range(int(index_center), int(index_stop)):
		filterbank[i] = top - ((i - index_center) * decrement)
	
	return filterbank

def filterplot():
	with open('select_filters/frog_data/experiment3/experiment3_train_sort_0.pkl', 'rb') as train_pkl:
		train_data = pickle.load(train_pkl)
	
	fs = 44100
	nfft = 2048
	numChannels = 20 # メルフィルタバンクのチャネル数
	df = fs / nfft   # 周波数解像度（周波数インデックス1あたりのHz幅）
	filterbanks = []
	for i in range(numChannels):
		filterbank1 = one_melFilterBank(fs, nfft, train_data[i][2], train_data[i][3])
		filterbanks.append(filterbank1)
	# メルフィルタバンクのプロット
	for c in np.arange(20):
		plt.xlabel('frequency[Hz]')
		plt.plot(np.arange(0, nfft / 2) * df, filterbanks[c])
		print(train_data[i][0])
	#plt.savefig("filterbank1_20.png")
	plt.show()

if __name__ == "__main__":
	filterplot()