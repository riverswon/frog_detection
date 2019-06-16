import numpy as np
from excel import load_excel_np
from scipy.stats import norm
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def linear_svm(train_data, test_data, sample):
	y_train = np.zeros(sample[0]+sample[1], int)
	y_train[:sample[0]] = 1
	#学習データから最適パラメータの導出
	para = {'C': np.logspace(-10, 9, 20, base=2.0), 'kernel': ['linear']}
	clf = GridSearchCV(SVC(), para, cv=5, scoring='f1')
	clf.fit(train_data, y_train)
	para_list = list(clf.best_params_.values())
	#評価データをフィッティング
	model = SVC(C=para_list[0], kernel=para_list[1])
	model.fit(train_data, y_train)
	predict = model.predict(test_data)
	return predict

def normal_distribution_predict(train_data, test_data, way, sample):
	if way == 'pca':
		pca_dim = 2 # the number of dimension by pca
		split_sample = test_data.shape[0] # the number of split

		mean = np.empty((2, pca_dim))
		sigma = np.empty(((2, pca_dim, pca_dim)))
		mean[0] = np.mean(train_data[:sample[0]], axis=0)
		mean[1] = np.mean(train_data[sample[0]:], axis=0)
		sigma[0] = np.cov(train_data[:sample[0]], rowvar=0, bias=1)
		sigma[1] = np.cov(train_data[sample[0]:], rowvar=0, bias=1)
		
		predict = np.zeros(split_sample, dtype=int)
		f_fg = lambda x, y: stats.multivariate_normal(mean[0], sigma[0]).pdf([x, y])
		f_no = lambda x, y: stats.multivariate_normal(mean[1], sigma[1]).pdf([x, y])
		for i in range(split_sample):
			fg_norm = np.vectorize(f_fg)(test_data[i][0], test_data[i][1])
			no_norm = np.vectorize(f_no)(test_data[i][0], test_data[i][1])
			judge = fg_norm - no_norm
			if judge >= 0:
				predict[i] = 1
		return predict
	elif way =='lda':
		split_sample = test_data.shape[0] # the number of split
		
		mean = np.empty(2)
		sigma = np.empty(2)
		mean[0] = np.mean(train_data[:sample[0]])
		mean[1] = np.mean(train_data[sample[0]:])
		sigma[0] = np.var(train_data[:sample[0]])
		sigma[1] = np.var(train_data[sample[0]:])
		
		predict = np.zeros(split_sample, dtype=int)
		for i in range(split_sample):
			fg_norm = norm.pdf(x=test_data[i], loc=mean[0], scale=sigma[0])
			no_norm = norm.pdf(x=test_data[i], loc=mean[1], scale=sigma[1])
			judge = fg_norm - no_norm
			if judge >= 0:
				predict[i] = 1
		return predict

def frog_classifier(train_data_x, test_data_x, out_predict_path, way, sample, wav_time, section):
	cross_validation = 5
	split_sample = int(wav_time / section)
	predict = np.empty(split_sample, dtype=int)
	if way == 'svm':
		for i in range(cross_validation):
			predict1 = linear_svm(train_data_x[i], test_data_x[i], sample[i])
			predict = np.vstack((predict, predict1))
		predict = np.delete(predict, obj=0, axis=0)
		np_excel(predict, out_predict_path)
		print('finished:predict_'+way)
	elif way == 'pca' or way =='lda':
		for i in range(cross_validation):
			predict1 = normal_distribution_predict(train_data_x[i], test_data_x[i], way, sample[i])
			predict = np.vstack((predict, predict1))
		predict = np.delete(predict, obj=0, axis=0)
		np_excel(predict, out_predict_path)
		print('finished:predict_'+way)


if __name__ == '__main__':
    in_train_path = 'mfcc/frog_data/05_non_non_train_mfcc2.pkl'
    in_test_path = 'mfcc/frog_data/05_non_non_test_mfcc.pkl'
    out_predict_path = 'mfcc/predict.xlsx'
    way = 'svm'
    test_sample = load_excel_np('evaluation/test_sample_0.5s.xlsx')
    wav_time = 120
    section = 0.5
    frog_classifier(in_train_path, in_test_path, out_predict_path, way, test_sample, wav_time, section)