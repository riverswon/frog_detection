import numpy as np
import pickle

def hz_mel(f): # convert Hz into mel
    return 1000 / np.log10(2) * np.log(f / 1000.0 + 1.0)

def mel_hz(m): # convert mel into Hz
    return 1000.0 * (np.exp(m * np.log10(2) / 1000) - 1.0)

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
    
    filterbank = np.zeros((numChannels, int(index_max)))
    for c in range(numChannels):
        # calculate the left of triangle filter
        increment = 1.0 / (index_center[c] - index_start[c])
        for i in range(int(index_start[c]), int(index_center[c])):
            filterbank[c][i] = (i - index_start[c]) * increment
        # calculate the right of triangle filter
        decrement = 1.0 / (index_stop[c] - index_center[c])
        for i in range(int(index_center[c]), int(index_stop[c])):
            filterbank[c][i] = 1.0 - ((i - index_center[c]) * decrement)

    return filterbank, f_centers
    
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
    top = 50 / (f_stop - f_start)
    increment = top / (index_center - index_start)
    for i in range(int(index_start), int(index_center)):
        filterbank[i] = (i - index_start) * increment
    # calculate the right of triangle filter
    decrement = top / (index_stop - index_center)
    for i in range(int(index_center), int(index_stop)):
        filterbank[i] = top - ((i - index_center) * decrement)
    
    return filterbank

def make_filters(fs, nfft):
    #f_max = int(fs / 2)
    f_max = 8000
    index_max = int(nfft / 2)

    # make mel-filters as width=300,500,...3000 num=0,100,...
    # start[Hz] stop[Hz] filter[0-fs/2[Hz]]
    filterbanks = np.empty(2 + index_max)
    for width in range(100, 3000, 100):
        for num in range(0, f_max - width, 50):
            filterbank1 = one_melFilterBank(fs, nfft, num, num + width)
            start_stop = np.array([num, num + width])
            filterbank1 = np.hstack((start_stop, filterbank1))
            filterbanks = np.vstack((filterbanks, filterbank1))
        print('finished:width=', width)
    filterbanks = np.delete(filterbanks, 0, 0) # delite [0][]
    #save filterbanks in pkl
    with open('select_filters/frog_data/filters.pkl', 'wb') as filters_pkl:
        pickle.dump(filterbanks , filters_pkl)

if __name__ == '__main__':
    fs = 44100
    nfft = 2048
    make_filters(fs, nfft)



