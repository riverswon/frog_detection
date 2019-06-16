import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from excel import load_excel_np

def dim_pca(train_data):
	y_train, x_train = np.hsplit(train_data, [1])
	y_train1 = np.ravel(y_train)
	pca = PCA(n_components = 2)
	pca_train = pca.fit_transform(x_train, y_train1)
	pca_train = np.hstack((y_train, pca_train))
	return pca_train

def experiment1_dim_reduction(num):
	with open('select_filters/frog_data/experiment1/experiment1_train_dct_'+str(num)+'.pkl', 'rb') as mfcc_pkl:
		train_mfcc = pickle.load(mfcc_pkl)
	train_pca = dim_pca(train_mfcc)
	with open('select_filters/frog_data/experiment1/experiment1_train_pca'+str(num)+'.pkl', 'wb') as pca_pkl:
		pickle.dump(train_pca, pca_pkl, protocol=2)
	with open('select_filters/frog_data/experiment1/experiment1_test_dct_'+str(num)+'.pkl', 'rb') as mfcc_pkl:
		train_mfcc = pickle.load(mfcc_pkl)
	train_pca = dim_pca(train_mfcc)
	with open('select_filters/frog_data/experiment1/experiment1_test_pca'+str(num)+'.pkl', 'wb') as pca_pkl:
		pickle.dump(train_pca, pca_pkl, protocol=2)
	print('finished:pca'+str(num))

def experiment2_dim_reduction(num):
	with open('select_filters/frog_data/experiment2/experiment2_train_dct_'+str(num)+'.pkl', 'rb') as mfcc_pkl:
		train_mfcc = pickle.load(mfcc_pkl)
	train_pca = dim_pca(train_mfcc)
	with open('select_filters/frog_data/experiment2/experiment2_train_pca'+str(num)+'.pkl', 'wb') as pca_pkl:
		pickle.dump(train_pca, pca_pkl, protocol=2)
	with open('select_filters/frog_data/experiment2/experiment2_test_dct_'+str(num)+'.pkl', 'rb') as mfcc_pkl:
		train_mfcc = pickle.load(mfcc_pkl)
	train_pca = dim_pca(train_mfcc)
	with open('select_filters/frog_data/experiment2/experiment2_test_pca'+str(num)+'.pkl', 'wb') as pca_pkl:
		pickle.dump(train_pca, pca_pkl, protocol=2)
	print('finished:pca'+str(num))

def experiment3_dim_reduction(num):
	with open('select_filters/frog_data/experiment3/experiment3_train_dct_'+str(num)+'.pkl', 'rb') as mfcc_pkl:
		train_mfcc = pickle.load(mfcc_pkl)
	train_pca = dim_pca(train_mfcc)
	with open('select_filters/frog_data/experiment3/experiment3_train_pca'+str(num)+'.pkl', 'wb') as pca_pkl:
		pickle.dump(train_pca, pca_pkl, protocol=2)
	with open('select_filters/frog_data/experiment3/experiment3_test_dct_'+str(num)+'.pkl', 'rb') as mfcc_pkl:
		train_mfcc = pickle.load(mfcc_pkl)
	train_pca = dim_pca(train_mfcc)
	with open('select_filters/frog_data/experiment3/experiment3_test_pca'+str(num)+'.pkl', 'wb') as pca_pkl:
		pickle.dump(train_pca, pca_pkl, protocol=2)
	print('finished:pca'+str(num))

def experiment4_dim_reduction(num):
	with open('select_filters/frog_data/experiment4/experiment4_train_dct_'+str(num)+'.pkl', 'rb') as mfcc_pkl:
		train_mfcc = pickle.load(mfcc_pkl)
	train_pca = dim_pca(train_mfcc)
	with open('select_filters/frog_data/experiment4/experiment4_train_pca'+str(num)+'.pkl', 'wb') as pca_pkl:
		pickle.dump(train_pca, pca_pkl, protocol=2)
	with open('select_filters/frog_data/experiment4/experiment4_test_dct_'+str(num)+'.pkl', 'rb') as mfcc_pkl:
		train_mfcc = pickle.load(mfcc_pkl)
	train_pca = dim_pca(train_mfcc)
	with open('select_filters/frog_data/experiment4/experiment4_test_pca'+str(num)+'.pkl', 'wb') as pca_pkl:
		pickle.dump(train_pca, pca_pkl, protocol=2)
	print('finished:pca'+str(num))



if __name__ == '__main__':
	for i in range(5):
		experiment1_dim_reduction(i)
		experiment2_dim_reduction(i)
	