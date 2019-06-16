import numpy as np
from excel import load_excel_np
from haar_like_filter import make_haarlike_filters
from stft_filter import frog2_train_select, frog1_train_select, back2_train_select, back1_train_select, experint2_train_select, experint1_train_select
from threshold import experiment2_set_threshold, experiment1_set_threshold
from sort import experiment2_sort, experiment1_sort

"""
#いらない
#学習データを適当にとりモデル作成
def frog1():
	fs = 44100
	wav_time = 120
	sections = [[0.1,0.5,1.0],['01','05','10']]
	preprocessings = [['non','zscore'],['non','norm1']]
	nfft = 2048
	hoplength = 512
	filter_sample = 20
	way1 = 'pca'
	way2 = 'svm'

	train_sample = load_excel_np('evaluation/training_sample.xlsx')
	cross_vali_sample = load_excel_np('evaluation/cross_validation_sample.xlsx')

	for section, section_name in zip(sections[0],sections[1]):
		for preprocessing1 in preprocessings[0]:
			for preprocessing2 in preprocessings[1]:
				out_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_filters_features1.pkl'
				frog_training_stft_log_filter1(out_train_path, fs, nfft, hoplength, train_sample, preprocessing1)
				
				in_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_filters_features1.pkl'
				out_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_threshold1.pkl'
				frog_set_threshold(in_train_path, out_train_path, cross_vali_sample)

				in_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_threshold1.pkl'
				out_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_sort1.pkl'
				frog_sort(in_train_path, out_train_path)

				in_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_sort1.pkl'
				out_test_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_test_sort1.pkl'
				frog_test_stft_log_filter(in_train_path, out_test_path, fs, wav_time, section, nfft, hoplength, filter_sample, preprocessing1)

				in_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_sort1.pkl'
				in_test_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_test_sort1.pkl'
				out_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_dct1.pkl'
				out_test_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_test_dct1.pkl'
				frog_dct(in_train_path, in_test_path, out_train_path, out_test_path, filter_sample, preprocessing2)
				
				#ここから画像の可視化
				in_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_dct1.pkl'
				in_test_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_test_dct1.pkl'
				out_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_'+way1+'1.pkl'
				out_test_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_test_'+way1+'1.pkl'
				frog_dim_reduction(in_train_path, in_test_path, out_train_path, out_test_path, way1, cross_vali_sample)

				in_train_path1 = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_'+way1+'1.pkl'
				in_train_path2 = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_sort1.pkl'
				name1 = section_name+'_'+preprocessing1+'_'+preprocessing2+'_select_filters_'+way1
				name2 = section_name+'_'+preprocessing1+'_'+preprocessing2+'_select_filter'
				training_data_plot(in_train_path1, cross_vali_sample[0], way1, name1)
				select_filter_data_plot(in_train_path2, cross_vali_sample[0], name2)

				in_train_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_train_dct1.pkl'
				in_test_path = 'select_filters/frog_data/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_test_dct1.pkl'
				out_predict_path = 'select_filters/frog_data/predict/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_select_filters_predict1.xlsx'
				frog_classifier(in_train_path, in_test_path, out_predict_path, way2, cross_vali_sample, wav_time, section)

				predict_path = 'select_filters/frog_data/predict/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_select_filters_predict1.xlsx'
				answer_path = 'evaluation/frog_sheet_'+str(section)+'s.xlsx'
				result_path = 'select_filters/frog_data/predict/'+section_name+'_'+preprocessing1+'_'+preprocessing2+'_select_filters_result1.xlsx'
				evaluate(predict_path, answer_path, result_path)

#学習データと評価データを同じものにする
def VAD_HaarLike():
	fs = 44100
	wav_time = 120
	nfft = 2048
	sections = [[0.1,0.5,1.0],['01','05','10']]
	f_max = 8000

	make_haarlike_filters(fs, nfft, f_max)
	for section, section_name in zip(sections[0], sections[1]):
		test_sample = load_excel_np('evaluation/test_sample_'+str(section)+'s.xlsx')
		#haar_likeデータを抽出
		train_features = frog_training_stft_filter(fs, nfft, wav_time, section)
		train_threshold = frog_set_threshold(train_features, test_sample)
		out_train_path = 'haar_like/frog_data/'+section_name+'_train_sort.pkl'
		frog_sort(train_threshold, out_train_path)
		in_train_path = 'haar_like/frog_data/'+section_name+'_train_sort.pkl'
		out_test_path = 'haar_like/frog_data/'+section_name+'_test_sort.pkl'
		test_dct = frog_test_stft_filter(in_train_path, out_test_path, fs, nfft, wav_time, section)
		in_train_path = 'haar_like/frog_data/'+section_name+'_train_sort.pkl'
		train_dct = train_split(in_train_path)
		#図を作成
		train_pca, test_pca = frog_dim_reduction(train_dct, test_dct, 'pca', test_sample)
		name = section_name+'_haarlike_pca'
		training_data_plot('haar_like', train_pca[0], test_sample[0], 'pca', name)
		name = section_name+'_haarlike'
		with open(in_train_path, 'rb') as training_sort_pkl:
			train_sort = pickle.load(training_sort_pkl)
			select_filter_data_plot('haar_like', train_sort[0], test_sample[0], name)
		#識別器で分類
		#SVM
		out_predict_path = 'haar_like/predict/predict_'+section_name+'_haarlike_svm.xlsx'
		frog_classifier(train_dct, test_dct, out_predict_path, 'svm', test_sample, wav_time, section)
		predict_path = 'haar_like/predict/predict_'+section_name+'_haarlike_svm.xlsx'
		answer_path = 'evaluation/frog_sheet_'+str(section)+'s.xlsx'
		result_path = 'haar_like/predict/result_'+section_name+'_haarlike_svm.xlsx'
		evaluate(predict_path, answer_path, result_path)
		#lda->normal distribution
		train_lda, test_lda = frog_dim_reduction(train_dct, test_dct, 'lda', test_sample)
		out_predict_path = 'haar_like/predict/predict_'+section_name+'_haarlike_lda.xlsx'
		frog_classifier(train_lda, test_lda, out_predict_path, 'lda', test_sample, wav_time, section)
		predict_path = 'haar_like/predict/predict_'+section_name+'_haarlike_lda.xlsx'
		answer_path = 'evaluation/frog_sheet_'+str(section)+'s.xlsx'
		result_path = 'haar_like/predict/result_'+section_name+'_haarlike_lda.xlsx'
		evaluate(predict_path, answer_path, result_path)
"""

if __name__ == '__main__':
	fs = 44100
	nfft = 2048
	f_max = 12000
	make_haarlike_filters(fs, nfft, f_max)
	for i in range(5):
		frog2_train_select(i)
	for i in range(5):
		frog1_train_select(i)
	for i in range(5):
		back2_train_select(i)
	for i in range(5):
		back1_train_select(i)
	for i in range(5):
		experint1_train_select(i)
	for i in range(5):
		experint2_train_select(i)
	for i in range(5):
		experiment2_set_threshold(i)
	for i in range(5):
		experiment1_set_threshold(i)
	for i in range(5):
		experiment2_sort(i)
	for i in range(5):
		experiment1_sort(i)

