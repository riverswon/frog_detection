import numpy as np
from excel import load_excel_np
from extract_mfcc import frog_training_extract_mfcc, frog_test_extract_mfcc
from dim_reduction import frog_dim_reduction
from plot import training_data_plot
from classifier import frog_classifier
from evaluation import evaluate

#学習データを適当にとりモデル作成
def frog1():
	fs = 44100
	wav_time = 120
	sections = [0.1, '01']
	preprocessings = ['non','norm1']
	mfcc_dim = 12
	way1 = 'pca'
	way2 = 'svm'

	train_sample = load_excel_np('evaluation/training_sample.xlsx')
	cross_vali_sample = load_excel_np('evaluation/cross_validation_sample.xlsx')
	
	for section, section_name in zip(sections[0],sections[1]):
		for preprocessing in preprocessings:
				out_train_path = 'mfcc/frog_data/'+section_name+'_'+preprocessing+'_train_mfcc1.pkl'
				frog_training_extract_mfcc1(out_train_path, fs, mfcc_dim, train_sample, preprocessing)

				#out_test_path = 'mfcc/frog_data/'+section_name+'_'+preprocessing+'_test_mfcc.pkl'
				#frog_test_extract_mfcc(out_test_path, fs, wav_time, section, mfcc_dim, preprocessing)

				#ここから画像の可視化
				in_train_path = 'mfcc/frog_data/'+section_name+'_'+preprocessing+'_train_mfcc1.pkl'
				#in_test_path = 'mfcc/frog_data/'+section_name+'_'+preprocessing+'_test_mfcc.pkl'
				out_train_path = 'mfcc/frog_data/'+section_name+'_'+preprocessing+'_train_'+way1+'1.pkl'
				#out_test_path = 'mfcc/frog_data/'+section_name+'_'+preprocessing+'_test_'+way1+'.pkl'
				frog_dim_reduction(in_train_path, in_test_path, out_train_path, out_test_path, way1, test_sample)

				in_train_path = 'mfcc/frog_data/'+section_name+'_'+preprocessing+'_train_'+way1+'1.pkl'
				name = section_name+'_'+preprocessing+'_mfcc_'+way1
				training_data_plot(in_train_path, test_sample[0], way1, name)

				in_train_path = 'mfcc/frog_data/'+section_name+'_'+preprocessing+'_train_mfcc2.pkl'
				in_test_path = 'mfcc/frog_data/'+section_name+'_'+preprocessing+'_test_mfcc.pkl'
				out_predict_path = 'mfcc/frog_data/predict/'+section_name+'_'+preprocessing+'_mfcc_predict1.xlsx'
				frog_classifier(in_train_path, in_test_path, out_predict_path, way2, test_sample, wav_time, section)

				predict_path = 'mfcc/frog_data/predict/'+section_name+'_'+preprocessing+'_mfcc_predict1.xlsx'
				answer_path = 'evaluation/frog_sheet_'+str(section)+'s.xlsx'
				result_path = 'mfcc/frog_data/predict/'+section_name+'_'+preprocessing+'_mfcc_result1.xlsx'
				evaluate(predict_path, answer_path, result_path)

#学習データと評価データを同じものにする
def VAD_MFCC():
	fs = 44100
	wav_time = 120
	nfft = 2048
	sections = [[0.1,0.5,1.0],['01','05','10']]

	for section, section_name in zip(sections[0], sections[1]):
		test_sample = load_excel_np('evaluation/test_sample_'+str(section)+'s.xlsx')
		#MFCCを抽出
		train_mfcc = frog_training_extract_mfcc(fs, nfft, wav_time, section)
		test_mfcc = frog_test_extract_mfcc(fs, nfft, wav_time, section)
		#図を作成
		train_pca, test_pca = frog_dim_reduction(train_mfcc, test_mfcc, 'pca', test_sample)
		name = section_name+'_mfcc'
		training_data_plot('mfcc', train_pca[0], test_sample[0], 'pca', name)
		#識別器で分類
		#SVM
		out_predict_path = 'mfcc/predict/predict_'+section_name+'_mfcc_svm.xlsx'
		frog_classifier(train_mfcc, test_mfcc, out_predict_path, 'svm', test_sample, wav_time, section)
		predict_path = 'mfcc/predict/predict_'+section_name+'_mfcc_svm.xlsx'
		answer_path = 'evaluation/frog_sheet_'+str(section)+'s.xlsx'
		result_path = 'mfcc/predict/result_'+section_name+'_mfcc_svm.xlsx'
		evaluate(predict_path, answer_path, result_path)
		#lda->normal distribution
		train_lda, test_lda = frog_dim_reduction(train_mfcc, test_mfcc, 'lda', test_sample)
		out_predict_path = 'mfcc/predict/predict_'+section_name+'_mfcc_lda.xlsx'
		frog_classifier(train_lda, test_lda, out_predict_path, 'lda', test_sample, wav_time, section)
		predict_path = 'mfcc/predict/predict_'+section_name+'_mfcc_lda.xlsx'
		answer_path = 'evaluation/frog_sheet_'+str(section)+'s.xlsx'
		result_path = 'mfcc/predict/result_'+section_name+'_mfcc_lda.xlsx'
		evaluate(predict_path, answer_path, result_path)

if __name__ == '__main__':
	VAD_MFCC()