import numpy as np
from excel import load_excel_np
from mel_filter import make_filters
from stft_log_filter import frog_training_stft_log_filter1, frog_training_stft_log_filter2, frog_test_stft_log_filter
from threshold import frog_set_threshold
from sort import frog_sort
from dct import frog_dct
from dim_reduction import frog_dim_reduction
from plot import *

#学習データを適当にとりモデル作成
def frog1():
	fs = 44100
	wav_time = 120
	sections = [[0.1,0.5,1.0],['01','05','10']]
	preprocessings = [['non','zscore'],['non','norm1']]
	nfft = 2048
	hoplength = 512
	filter_sample = 20
	way = 'pca'

	train_sample = load_excel_np('evaluation/training_sample.xlsx')
	cross_vali_sample = load_excel_np('evaluation/cross_validation_sample.xlsx')

	for section, section_name in zip(sections[0],sections[1]):
		test_sample = load_excel_np('evaluation/test_sample_'+str(section)+'s.xlsx')
		for preprocessing1 in preprocessings[0]:
			for preprocessing2 in preprocessings[1]:
				out_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_filters_features2.pkl'
				frog_training_stft_log_filter2(out_train_path, fs, wav_time, section, nfft, hoplength, preprocessing1)
				




#学習データと評価データを同じものにする
def frog2():
	fs = 44100
	wav_time = 120
	sections = [[0.1,0.5,1.0],['01','05','10']]
	preprocessings = [['non','zscore'],['non','norm1']]
	nfft = 2048
	hoplength = 512
	filter_sample = 20
	way = 'pca'

	make_filters(fs, nfft)
	for section, section_name in zip(sections[0],sections[1]):
		test_sample = load_excel_np('evaluation/test_sample_'+str(section)+'s.xlsx')
		for preprocessing1 in preprocessings[0]:
			for preprocessing2 in preprocessings[1]:
				out_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_filters_features2.pkl'
				frog_training_stft_log_filter2(out_train_path, fs, wav_time, section, nfft, hoplength, preprocessing1)
				
				in_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_filters_features2.pkl'
				out_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_threshold2.pkl'
				frog_set_threshold(in_train_path, out_train_path, test_sample)

				in_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_threshold2.pkl'
				out_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_sort2.pkl'
				frog_sort(in_train_path, out_train_path)

				in_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_sort2.pkl'
				out_test_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_test_sort2.pkl'
				frog_test_stft_log_filter(in_train_path, out_test_path, fs, wav_time, section, nfft, hoplength, filter_sample, preprocessing1)

				in_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_sort2.pkl'
				in_test_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_test_sort2.pkl'
				out_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_dct2.pkl'
				out_test_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_test_dct2.pkl'
				frog_dct(in_train_path, in_test_path, out_train_path, out_test_path, filter_sample, preprocessing2)
				
				#ここから画像の可視化
				in_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_dct2.pkl'
				in_test_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_test_dct2.pkl'
				out_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_'+way+'2.pkl'
				out_test_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_test_'+way+'2.pkl'
				frog_dim_reduction(in_train_path, in_test_path, out_train_path, out_test_path, way, test_sample)

				in_train_path1 = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_'+way+'2.pkl'
				in_train_path2 = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_sort2.pkl'
				name1 = section_name+'_'+preprocessing1+'_'+preprocessing2+'_select_filters_'+way
				name2 = section_name+'_'+preprocessing1+'_'+preprocessing2+'_select_filter'
				training_data_plot(in_train_path1, test_sample[0], way, name1)
				select_filter_data_plot(in_train_path2, test_sample[0], name2)

if __name__ == '__main__':
	#frog1()
	frog2()
