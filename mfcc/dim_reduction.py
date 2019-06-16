import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#次元削減
def dim_pca(train_data):
	y_train, x_train = np.hsplit(train_data, [1])
	y_train1 = np.ravel(y_train)
	pca = PCA(n_components = 2)
	pca_train = pca.fit_transform(x_train)
	print(pca_train.shape)
	#pca_train = np.hstack((y_train, pca_train))
	return pca_train

def frog1_dim_reduction(num):
	with open('mfcc/frog_data/experiment1/experiment1_train_mfcc_'+str(num)+'.pkl', 'rb') as mfcc_pkl:
		train_mfcc = pickle.load(mfcc_pkl)
	del mfcc_pkl
	train_pca = dim_pca(train_mfcc)
	with open('mfcc/frog_data/experiment1/experiment1_train_pca'+str(num)+'.pkl', 'wb') as pca_pkl:
		pickle.dump(train_pca, pca_pkl, protocol=2)
	print('finished:pca'+str(num))

def frog2_dim_reduction(num):
	with open('mfcc/frog_data/experiment2/train_mfcc'+str(num)+'.pkl', 'rb') as mfcc_pkl:
		train_mfcc = pickle.load(mfcc_pkl)
	del mfcc_pkl
	train_pca = dim_pca(train_mfcc)
	with open('mfcc/frog_data/experiment2/train_pca'+str(num)+'.pkl', 'wb') as pca_pkl:
		pickle.dump(train_pca, pca_pkl, protocol=2)
	print('finished:pca'+str(num))

if __name__ == '__main__':
	for i in range(1):
		frog1_dim_reduction(i+1)
		#frog2_dim_reduction(i)