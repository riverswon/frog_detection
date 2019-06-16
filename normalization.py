#normalization of 1,...,5.wav by [-1:1]
#in  1,...,5.wav
#out norm1,...,5.wav

import librosa
import numpy as np

#normalization of 1,...5.wav by [-1:1]
def norm1(in_path, out_path, fs):
    x, f = librosa.load(in_path, sr=fs)
    x_max = max(x)
    x_min = min(x)
    for i in range(len(x)):
        x[i] = (x[i] - x_min) / (x_max - x_min) * 2.0 - 1.0
    librosa.output.write_wav(out_path, x, fs)

def frog_normalization():
    fs = 44100
    frog_sample = 5
    for i in range(frog_sample):
        norm1('original_data/frog'+str(i+1)+'.wav', 'test_data/frog'+str(i+1)+'.wav', fs)

def bat_normalization():
    fs = 44100
    norm1('original_data/bat.wav', 'test_data/bat.wav', fs)

#frog_normalization()
#bat_normalization()