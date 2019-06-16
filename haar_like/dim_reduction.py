import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#次元削減
def dim_pca(dct_training, dct_test, p, n):
	y_train = np.zeros(p+n)
	#training_data:set 0 or 1
	y_train[:p] = 1
	#to 2 dimesion by PCA
	pca = PCA(n_components = 2)
	pca_training = pca.fit_transform(dct_training, y_train)
	pca_test = pca.transform(dct_test)
	return pca_training, pca_test

def dim_lda(dct_training, dct_test, p, n):
	y_train = np.zeros(p+n)
	#training_data:set 0 or 1
	y_train[:p] = 1
	lda = LDA(n_components = 1)
	lda_training = lda.fit_transform(dct_training, y_train)
	lda_test = lda.transform(dct_test)
	return lda_training, lda_test

def frog_dim_reduction(train_dct_x, test_dct_x, way, sample):
	cross_validation = 5
	train_dim_reduction = []
	test_dim_reduction = []
	print('finished:frog_dim_reduction_'+way)
	if way == 'pca':
		for i in range(cross_validation):
			train_pca1, test_pca1 = dim_pca(train_dct_x[i], test_dct_x[i], sample[i][0], sample[i][1])
			train_dim_reduction.append(train_pca1)
			test_dim_reduction.append(test_pca1)
		return train_dim_reduction, test_dim_reduction
	elif way == 'lda':
		for i in range(cross_validation):
			train_lda1, test_lda1 = dim_lda(train_dct_x[i], test_dct_x[i], sample[i][0], sample[i][1])
			train_dim_reduction.append(train_lda1)
			test_dim_reduction.append(test_lda1)
		return train_dim_reduction, test_dim_reduction


if __name__ == '__main__':
    section = 0.5
    cross_vali_sample = load_excel_np('evaluation/cross_validation_sample.xlsx')
    test_sample = load_excel_np('evaluation/test_sample_'+str(section)+'s.xlsx')
    way = 'pca'
    in_train_path = 'mfcc/frog_data/training_mfcc2.pkl'
    in_test_path = 'mfcc/frog_data/test_mfcc2.pkl'
    out_train_path = 'mfcc/frog_data/training_pca2.pkl'
    out_test_path = 'mfcc/frog_data/test_pca2.pkl'
    frog_dim_reduction(in_train_path, in_test_path, out_train_path, out_test_path, way, test_sample)