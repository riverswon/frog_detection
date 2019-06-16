import numpy as np
import pickle

def hz_mel(f): # convert Hz into mel
	return 1127.01048 * np.log(f / 700.0 + 1.0)

def mel_hz(m): # convert mel into Hz
	return 700.0 * (np.exp(m / 1127.01048) - 1.0)

def melFilterBank(fs, nfft, numChannels): # make mel-filter-bank
	f_max = fs / 2 # Nyquist frequency(Hz)
	mel_max = hz_mel(f_max) # Nyquist frequency(mel)
	index_max = nfft / 2 # the number of index (remove mirror)
	df = fs / nfft # the interval Hz by frequency index
	# calculate the cernter frequency(mel) in each filter
	dmel = mel_max / (numChannels + 1)
	mel_centers = np.arange(1, numChannels + 1) * dmel
	f_centers = mel_hz(mel_centers) # convert mel into Hz in each the cernter frequency
	index_center = np.round(f_centers / df) # convert frequency into index in each the cernter frequency
	index_start = np.hstack(([0], index_center[0:numChannels - 1])) # the start of index in each filter
	index_stop = np.hstack((index_center[1:numChannels], [index_max])) #  the end of index in each filter

	f_start = np.hstack(([0], f_centers[:numChannels-1]))
	f_stop = np.hstack((f_centers[1:], [fs]))
	
	filterbank = np.zeros((numChannels, int(index_max)))
	for c in range(numChannels):
		# calculate the left of triangle filter
		#top = 50 / (f_stop[c] - f_start[c])
		top = 1
		increment = top / (index_center[c] - index_start[c]) # 1.0->top
		for i in range(int(index_start[c]), int(index_center[c])):
			filterbank[c][i] = (i - index_start[c]) * increment
		# calculate the right of triangle filter
		decrement = top / (index_stop[c] - index_center[c])
		for i in range(int(index_center[c]), int(index_stop[c])):
			filterbank[c][i] = top - ((i - index_center[c]) * decrement)

	return filterbank, f_centers
	


if __name__ == '__main__':
	
	import matplotlib.pyplot as plt
	
	"""
	filterbank, fcenters = melFilterBank(fs, nfft, numChannels)
	with open('mfcc/frog_data/experiment1/melfilterbank2.pkl', 'wb') as filterbank_pkl:
		pickle.dump(filterbank , filterbank_pkl, protocol=2)
	"""

	fs = 44100
	nfft = 2048
	numChannels = 20  # メルフィルタバンクのチャネル数
	df = fs / nfft   # 周波数解像度（周波数インデックス1あたりのHz幅）
	filterbank, fcenters = melFilterBank(fs, nfft, numChannels)
	# メルフィルタバンクのプロット
	for c in np.arange(0, numChannels):
		plt.xlabel('frequency[Hz]')
		plt.plot(np.arange(0, nfft / 2) * df, filterbank[c])
	plt.savefig("melfilterbank.png")
	print(fcenters)
	plt.show()
	
	fs = 44100
	nfft = 2048
	numChannels = 20  # メルフィルタバンクのチャネル数
	filterbank, fcenters = melFilterBank(fs, nfft, numChannels)
	print(fcenters)
	