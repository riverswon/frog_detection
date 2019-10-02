# normalization [-1, 1]

import librosa
import numpy as np

def norm(x):
	x_max = max(x)
	x_min = min(x)
	x[i] = (x[i] - x_min) / (x_max - x_min) * 2.0 - 1.0
	return x

if __name__ == "__main__":
	inpath = 'data/...'
	outpath = 'data/...'
	fs = 44100
	x, f = librosa.load(in_path, sr=fs)
	x = norm(x)
	librosa.output.write_wav(out_path, x, fs)