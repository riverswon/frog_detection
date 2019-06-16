import librosa
import numpy as np
from excel import load_excel_np
import scipy.signal
import scipy.fftpack
import pickle

# ner way for mfcc
def wav_mfcc(x, nfft):
	p = 0.97
	y = scipy.signal.lfilter([1.0, -p], 1, x)
	y = librosa.feature.mfcc(y, sr=44100, n_mfcc=20)
	y = np.mean(y, axis = 1)
	y = y[:13]
	return y

def experiment1_train_mfcc(num):
	fs = 44100
	nfft = 2048
	sample1 = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	sample2 = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	mfcc = np.empty(13)
	if num != 0:
		for i in range(0, sample1[0][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(0, sample1[0][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(0, sample2[0][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(0, sample2[0][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num != 1:
		for i in range(sample1[0][0], sample1[0][0]+sample1[1][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1], sample1[0][1]+sample1[1][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][0], sample2[0][0]+sample2[1][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1], sample2[0][1]+sample2[1][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num != 2:
		for i in range(sample1[0][0]+sample1[1][0], sample1[0][0]+sample1[1][0]+sample1[2][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1], sample1[0][1]+sample1[1][1]+sample1[2][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][0]+sample2[1][0], sample2[0][0]+sample2[1][0]+sample2[2][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1], sample2[0][1]+sample2[1][1]+sample2[2][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num != 3:
		for i in range(sample1[0][0]+sample1[1][0]+sample1[2][0], sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1]+sample1[2][1], sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][0]+sample2[1][0]+sample2[2][0], sample2[0][0]+sample2[1][0]+sample2[2][0]+sample2[3][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1]+sample2[2][1], sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num != 4:
		for i in range(sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0], sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0]+sample1[4][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1], sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1]+sample1[4][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][0]+sample2[1][0]+sample2[2][0]+sample2[3][0], sample2[0][0]+sample2[1][0]+sample2[2][0]+sample2[3][0]+sample2[4][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1], sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1]+sample2[4][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))

	mfcc = np.delete(mfcc, obj=0, axis=0)
	y_test1 = np.empty(1, dtype=int)
	for i in range(5):
		if i == num:
			continue
		y_test2 = np.ones(sample1[i][0], dtype=int)
		y_test1 = np.hstack((y_test1, y_test2))
		y_test2 = np.zeros(sample1[i][1]+sample2[i][0]+sample2[i][1], dtype=int)
		y_test1 = np.hstack((y_test1, y_test2))
	y_test1 = y_test1[1:]
	y_test = np.reshape(y_test1, (y_test1.shape[0], 1))
		
	mfcc = np.hstack((y_test, mfcc))
	with open('mfcc/frog_data/experiment1/experiment1_train_mfcc_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(mfcc, train_mfcc_pkl, protocol=2)
	del mfcc
	del train_mfcc_pkl
	print('finished:1_train_mfcc'+str(num))

def experiment2_train_mfcc(num):
	fs = 44100
	nfft = 2048
	sample1 = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	sample2 = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	mfcc = np.empty(13)
	if num != 0:
		for i in range(0, sample1[0][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(0, sample1[0][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(0, sample2[0][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(0, sample2[0][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num != 1:
		for i in range(sample1[0][0], sample1[0][0]+sample1[1][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1], sample1[0][1]+sample1[1][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][0], sample2[0][0]+sample2[1][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1], sample2[0][1]+sample2[1][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num != 2:
		for i in range(sample1[0][0]+sample1[1][0], sample1[0][0]+sample1[1][0]+sample1[2][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1], sample1[0][1]+sample1[1][1]+sample1[2][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][0]+sample2[1][0], sample2[0][0]+sample2[1][0]+sample2[2][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1], sample2[0][1]+sample2[1][1]+sample2[2][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num != 3:
		for i in range(sample1[0][0]+sample1[1][0]+sample1[2][0], sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1]+sample1[2][1], sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][0]+sample2[1][0]+sample2[2][0], sample2[0][0]+sample2[1][0]+sample2[2][0]+sample2[3][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1]+sample2[2][1], sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num != 4:
		for i in range(sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0], sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0]+sample1[4][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1], sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1]+sample1[4][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][0]+sample2[1][0]+sample2[2][0]+sample2[3][0], sample2[0][0]+sample2[1][0]+sample2[2][0]+sample2[3][0]+sample2[4][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1], sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1]+sample2[4][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))

	mfcc = np.delete(mfcc, obj=0, axis=0)
	y_test1 = np.empty(1, dtype=int)
	for i in range(5):
		if i == num:
			continue
		y_test2 = np.ones(sample1[i][0], dtype=int)
		y_test1 = np.hstack((y_test1, y_test2))
		y_test2 = np.zeros(sample1[i][1]+sample2[i][0]+sample2[i][1], dtype=int)
		y_test1 = np.hstack((y_test1, y_test2))
	y_test1 = y_test1[1:]
	y_test = np.reshape(y_test1, (y_test1.shape[0], 1))
		
	mfcc = np.hstack((y_test, mfcc))
	with open('mfcc/frog_data/experiment2/experiment2_train_mfcc_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(mfcc, train_mfcc_pkl, protocol=2)
	del mfcc
	del train_mfcc_pkl
	print('finished:2_train_mfcc'+str(num))

def experiment1_test_mfcc(num):
	fs = 44100
	nfft = 2048
	sample1 = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	sample2 = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	mfcc = np.empty(13)
	if num == 0:
		for i in range(0, sample1[0][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(0, sample1[0][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(0, sample2[0][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(0, sample2[0][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num == 1:
		for i in range(sample1[0][0], sample1[0][0]+sample1[1][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1], sample1[0][1]+sample1[1][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][0], sample2[0][0]+sample2[1][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1], sample2[0][1]+sample2[1][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num == 2:
		for i in range(sample1[0][0]+sample1[1][0], sample1[0][0]+sample1[1][0]+sample1[2][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1], sample1[0][1]+sample1[1][1]+sample1[2][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][0]+sample2[1][0], sample2[0][0]+sample2[1][0]+sample2[2][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1], sample2[0][1]+sample2[1][1]+sample2[2][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num == 3:
		for i in range(sample1[0][0]+sample1[1][0]+sample1[2][0], sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1]+sample1[2][1], sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][0]+sample2[1][0]+sample2[2][0], sample2[0][0]+sample2[1][0]+sample2[2][0]+sample2[3][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1]+sample2[2][1], sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num == 4:
		for i in range(sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0], sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0]+sample1[4][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1], sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1]+sample1[4][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][0]+sample2[1][0]+sample2[2][0]+sample2[3][0], sample2[0][0]+sample2[1][0]+sample2[2][0]+sample2[3][0]+sample2[4][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1], sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1]+sample2[4][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))

	mfcc = np.delete(mfcc, obj=0, axis=0)
	y_test1 = np.ones(sample1[num][0], dtype=int)
	y_test2 = np.zeros(sample1[num][1]+sample2[num][0]+sample2[num][1])
	y_test1 = np.hstack((y_test1, y_test2))
	y_test = np.reshape(y_test1, (y_test1.shape[0], 1))
		
	mfcc = np.hstack((y_test, mfcc))
	with open('mfcc/frog_data/experiment1/experiment1_test_mfcc_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(mfcc, train_mfcc_pkl, protocol=2)
	del mfcc
	del train_mfcc_pkl
	print('finished:1_test_mfcc'+str(num))

def experiment2_test_mfcc(num):
	fs = 44100
	nfft = 2048
	sample1 = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	sample2 = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	mfcc = np.empty(13)
	if num == 0:
		for i in range(0, sample1[0][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(0, sample1[0][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(0, sample2[0][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(0, sample2[0][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num == 1:
		for i in range(sample1[0][0], sample1[0][0]+sample1[1][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1], sample1[0][1]+sample1[1][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][0], sample2[0][0]+sample2[1][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1], sample2[0][1]+sample2[1][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num == 2:
		for i in range(sample1[0][0]+sample1[1][0], sample1[0][0]+sample1[1][0]+sample1[2][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1], sample1[0][1]+sample1[1][1]+sample1[2][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][0]+sample2[1][0], sample2[0][0]+sample2[1][0]+sample2[2][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1], sample2[0][1]+sample2[1][1]+sample2[2][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num == 3:
		for i in range(sample1[0][0]+sample1[1][0]+sample1[2][0], sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1]+sample1[2][1], sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][0]+sample2[1][0]+sample2[2][0], sample2[0][0]+sample2[1][0]+sample2[2][0]+sample2[3][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1]+sample2[2][1], sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num == 4:
		for i in range(sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0], sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0]+sample1[4][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1], sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1]+sample1[4][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][0]+sample2[1][0]+sample2[2][0]+sample2[3][0], sample2[0][0]+sample2[1][0]+sample2[2][0]+sample2[3][0]+sample2[4][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1], sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1]+sample2[4][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))

	mfcc = np.delete(mfcc, obj=0, axis=0)
	y_test1 = np.ones(sample1[num][0], dtype=int)
	y_test2 = np.zeros(sample1[num][1]+sample2[num][0]+sample2[num][1])
	y_test1 = np.hstack((y_test1, y_test2))
	y_test = np.reshape(y_test1, (y_test1.shape[0], 1))
		
	mfcc = np.hstack((y_test, mfcc))
	with open('mfcc/frog_data/experiment2/experiment2_test_mfcc_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(mfcc, train_mfcc_pkl, protocol=2)
	del mfcc
	del train_mfcc_pkl
	print('finished:2_test_mfcc'+str(num))

#non frog2
def experiment3_train_mfcc(num):
	fs = 44100
	nfft = 2048
	sample1 = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	sample2 = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	mfcc = np.empty(13)
	if num != 0:
		for i in range(0, sample1[0][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(0, sample1[0][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(0, sample2[0][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num != 1:
		for i in range(sample1[0][0], sample1[0][0]+sample1[1][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1], sample1[0][1]+sample1[1][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1], sample2[0][1]+sample2[1][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num != 2:
		for i in range(sample1[0][0]+sample1[1][0], sample1[0][0]+sample1[1][0]+sample1[2][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1], sample1[0][1]+sample1[1][1]+sample1[2][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1], sample2[0][1]+sample2[1][1]+sample2[2][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num != 3:
		for i in range(sample1[0][0]+sample1[1][0]+sample1[2][0], sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1]+sample1[2][1], sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1]+sample2[2][1], sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num != 4:
		for i in range(sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0], sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0]+sample1[4][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1], sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1]+sample1[4][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1], sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1]+sample2[4][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))

	mfcc = np.delete(mfcc, obj=0, axis=0)
	y_test1 = np.empty(1, dtype=int)
	for i in range(5):
		if i == num:
			continue
		y_test2 = np.ones(sample1[i][0], dtype=int)
		y_test1 = np.hstack((y_test1, y_test2))
		y_test2 = np.zeros(sample1[i][1]+sample2[i][1], dtype=int)
		y_test1 = np.hstack((y_test1, y_test2))
	y_test1 = y_test1[1:]
	y_test = np.reshape(y_test1, (y_test1.shape[0], 1))
		
	mfcc = np.hstack((y_test, mfcc))
	with open('mfcc/frog_data/experiment3/experiment3_train_mfcc_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(mfcc, train_mfcc_pkl, protocol=2)
	del mfcc
	del train_mfcc_pkl
	print('finished:3_train_mfcc'+str(num))

#non frog1
def experiment4_train_mfcc(num):
	fs = 44100
	nfft = 2048
	sample1 = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	sample2 = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	mfcc = np.empty(13)
	if num != 0:
		for i in range(0, sample1[0][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(0, sample1[0][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(0, sample2[0][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num != 1:
		for i in range(sample1[0][0], sample1[0][0]+sample1[1][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1], sample1[0][1]+sample1[1][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1], sample2[0][1]+sample2[1][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num != 2:
		for i in range(sample1[0][0]+sample1[1][0], sample1[0][0]+sample1[1][0]+sample1[2][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1], sample1[0][1]+sample1[1][1]+sample1[2][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1], sample2[0][1]+sample2[1][1]+sample2[2][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num != 3:
		for i in range(sample1[0][0]+sample1[1][0]+sample1[2][0], sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1]+sample1[2][1], sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1]+sample2[2][1], sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
	if num != 4:
		for i in range(sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0], sample1[0][0]+sample1[1][0]+sample1[2][0]+sample1[3][0]+sample1[4][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1], sample1[0][1]+sample1[1][1]+sample1[2][1]+sample1[3][1]+sample1[4][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))
		for i in range(sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1], sample2[0][1]+sample2[1][1]+sample2[2][1]+sample2[3][1]+sample2[4][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc1 = wav_mfcc(x, nfft)
			mfcc = np.vstack((mfcc, mfcc1))

	mfcc = np.delete(mfcc, obj=0, axis=0)
	y_test1 = np.empty(1, dtype=int)
	for i in range(5):
		if i == num:
			continue
		y_test2 = np.ones(sample1[i][0], dtype=int)
		y_test1 = np.hstack((y_test1, y_test2))
		y_test2 = np.zeros(sample1[i][1]+sample2[i][1], dtype=int)
		y_test1 = np.hstack((y_test1, y_test2))
	y_test1 = y_test1[1:]
	y_test = np.reshape(y_test1, (y_test1.shape[0], 1))
		
	mfcc = np.hstack((y_test, mfcc))
	with open('mfcc/frog_data/experiment4/experiment4_train_mfcc_'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(mfcc, train_mfcc_pkl, protocol=2)
	del mfcc
	del train_mfcc_pkl
	print('finished:4_train_mfcc'+str(num))


def frog1_test_mfcc(num):
	fs = 44100
	nfft = 2048
	with open('mfcc/frog_data/experiment1/melfilterbank1.pkl', 'rb') as filterbank_pkl:
		melfilterbank = pickle.load(filterbank_pkl)
	del filterbank_pkl
	sample = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	mfcc = np.empty((np.sum(sample[num]), 13))
	mfcc_count = 0
	if num == 0:
		for i in range(0, sample[0][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 1
			mfcc_count += 1
		for i in range(0, sample[0][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
	if num == 1:
		for i in range(sample[0][0], sample[0][0]+sample[1][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 1
			mfcc_count += 1
		for i in range(sample[0][1], sample[0][1]+sample[1][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
	if num == 2:
		for i in range(sample[0][0]+sample[1][0], sample[0][0]+sample[1][0]+sample[2][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 1
			mfcc_count += 1
		for i in range(sample[0][1]+sample[1][1], sample[0][1]+sample[1][1]+sample[2][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
	if num == 3:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 1
			mfcc_count += 1
		for i in range(sample[0][1]+sample[1][1]+sample[2][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
	if num == 4:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]+sample[4][0]):
			x, _ = librosa.load('audio_data/frog1/frog'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 1
			mfcc_count += 1
		for i in range(sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]+sample[4][1]):
			x, _ = librosa.load('audio_data/frog1/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1

	with open('mfcc/frog_data/experiment1/test_mfcc'+str(num)+'.pkl', 'wb') as test_mfcc_pkl:
		pickle.dump(mfcc, test_mfcc_pkl, protocol=2)
	del mfcc
	del test_mfcc_pkl
	print('finished:test_mfcc'+str(num))

def frog2_train_mfcc(num):
	fs = 192000
	nfft = 8192
	with open('mfcc/frog_data/experiment2/melfilterbank2.pkl', 'rb') as filterbank_pkl:
		melfilterbank = pickle.load(filterbank_pkl)
	del filterbank_pkl
	sample = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	sample_sum = np.sum(sample) - np.sum(sample[num])
	mfcc = np.empty((sample_sum, 13))
	mfcc_count = 0
	if num != 0:
		for i in range(0, sample[0][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 1
			mfcc_count += 1
		for i in range(0, sample[0][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
		for i in range(0, sample[0][2]):
			x, _ = librosa.load('audio_data/frog3/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
	if num != 1:
		for i in range(sample[0][0], sample[0][0]+sample[1][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 1
			mfcc_count += 1
		for i in range(sample[0][1], sample[0][1]+sample[1][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
		for i in range(sample[0][2], sample[0][2]+sample[1][2]):
			x, _ = librosa.load('audio_data/frog3/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
	if num != 2:
		for i in range(sample[0][0]+sample[1][0], sample[0][0]+sample[1][0]+sample[2][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 1
			mfcc_count += 1
		for i in range(sample[0][1]+sample[1][1], sample[0][1]+sample[1][1]+sample[2][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
		for i in range(sample[0][2]+sample[1][2], sample[0][2]+sample[1][2]+sample[2][2]):
			x, _ = librosa.load('audio_data/frog3/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
	if num != 3:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 1
			mfcc_count += 1
		for i in range(sample[0][1]+sample[1][1]+sample[2][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
		for i in range(sample[0][2]+sample[1][2]+sample[2][2], sample[0][2]+sample[1][2]+sample[2][2]+sample[3][2]):
			x, _ = librosa.load('audio_data/frog3/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
	if num != 4:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]+sample[4][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 1
			mfcc_count += 1
		for i in range(sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]+sample[4][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
		for i in range(sample[0][2]+sample[1][2]+sample[2][2]+sample[3][2], sample[0][2]+sample[1][2]+sample[2][2]+sample[3][2]+sample[4][2]):
			x, _ = librosa.load('audio_data/frog3/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1

	with open('mfcc/frog_data/experiment2/train_mfcc'+str(num)+'.pkl', 'wb') as train_mfcc_pkl:
		pickle.dump(mfcc, train_mfcc_pkl, protocol=2)
	del mfcc
	del train_mfcc_pkl
	print('finished:train_mfcc'+str(num))

def frog2_test_mfcc(num):
	fs = 192000
	nfft = 8192
	with open('mfcc/frog_data/experiment2/melfilterbank2.pkl', 'rb') as filterbank_pkl:
		melfilterbank = pickle.load(filterbank_pkl)
	del filterbank_pkl
	sample = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	mfcc = np.empty((np.sum(sample[num]), 13))
	mfcc_count = 0
	if num == 0:
		for i in range(0, sample[0][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 1
			mfcc_count += 1
		for i in range(0, sample[0][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
		for i in range(0, sample[0][2]):
			x, _ = librosa.load('audio_data/frog3/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
	if num == 1:
		for i in range(sample[0][0], sample[0][0]+sample[1][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 1
			mfcc_count += 1
		for i in range(sample[0][1], sample[0][1]+sample[1][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
		for i in range(sample[0][2], sample[0][2]+sample[1][2]):
			x, _ = librosa.load('audio_data/frog3/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
	if num == 2:
		for i in range(sample[0][0]+sample[1][0], sample[0][0]+sample[1][0]+sample[2][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 1
			mfcc_count += 1
		for i in range(sample[0][1]+sample[1][1], sample[0][1]+sample[1][1]+sample[2][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
		for i in range(sample[0][2]+sample[1][2], sample[0][2]+sample[1][2]+sample[2][2]):
			x, _ = librosa.load('audio_data/frog3/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
	if num == 3:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 1
			mfcc_count += 1
		for i in range(sample[0][1]+sample[1][1]+sample[2][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
		for i in range(sample[0][2]+sample[1][2]+sample[2][2], sample[0][2]+sample[1][2]+sample[2][2]+sample[3][2]):
			x, _ = librosa.load('audio_data/frog3/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
	if num == 4:
		for i in range(sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0], sample[0][0]+sample[1][0]+sample[2][0]+sample[3][0]+sample[4][0]):
			x, _ = librosa.load('audio_data/frog2/frog'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 1
			mfcc_count += 1
		for i in range(sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1], sample[0][1]+sample[1][1]+sample[2][1]+sample[3][1]+sample[4][1]):
			x, _ = librosa.load('audio_data/frog2/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1
		for i in range(sample[0][2]+sample[1][2]+sample[2][2]+sample[3][2], sample[0][2]+sample[1][2]+sample[2][2]+sample[3][2]+sample[4][2]):
			x, _ = librosa.load('audio_data/frog3/back'+str(i)+'.wav', sr=fs)
			mfcc[mfcc_count][1:13] = wav_mfcc(x, nfft, melfilterbank)
			mfcc[mfcc_count][0] = 0
			mfcc_count += 1

	with open('mfcc/frog_data/experiment2/test_mfcc'+str(num)+'.pkl', 'wb') as test_mfcc_pkl:
		pickle.dump(mfcc, test_mfcc_pkl, protocol=2)
	del mfcc
	del test_mfcc_pkl
	print('finished:test_mfcc'+str(num))




if __name__ == "__main__":
	with open('mfcc/frog_data/experiment1/train_pca1.pkl', 'rb') as train_mfcc_pkl:
		x = pickle.load(train_mfcc_pkl)
		print(x.shape)