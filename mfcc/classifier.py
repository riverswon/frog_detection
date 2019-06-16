import numpy as np
import pickle
from excel import np_excel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

def linear_svm(train_data, test_data):
	y_train, x_train = np.hsplit(train_data, [1])
	y_test, x_test = np.hsplit(test_data, [1])
	y_train = np.ravel(y_train)
	y_test = np.ravel(y_test)
	#学習データから最適パラメータの導出
	para = {'C': np.logspace(-10, 9, 20, base=2.0), 'kernel': ['linear']}
	kfold = KFold(n_splits=5)
	clf = GridSearchCV(SVC(), para, cv=kfold, scoring='f1')
	clf.fit(x_train, y_train)
	para_list = list(clf.best_params_.values())
	#評価データをフィッティング
	model = SVC(C=para_list[0], kernel=para_list[1])
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	
	result = np.empty(8)
	result[3], result[2], result[1], result[0] = confusion_matrix(y_test, y_pred).ravel()
	result[4] = accuracy_score(y_test,y_pred)
	result[5] = recall_score(y_test, y_pred)
	result[6] = precision_score(y_test, y_pred)
	result[7] = f1_score(y_test, y_pred)
	return result
	
def experiment2_classifier():
	result = np.empty((5,8))
	for num in range(5):
		with open('mfcc/frog_data/experiment2/experiment2_train_mfcc_'+str(num)+'.pkl', 'rb') as train_pkl:
			train_data = pickle.load(train_pkl)
		with open('mfcc/frog_data/experiment2/experiment2_test_mfcc_'+str(num)+'.pkl', 'rb') as test_pkl:
			test_data = pickle.load(test_pkl)
		
		result[num] = linear_svm(train_data, test_data)
		print('finished:'+str(num))
	
	all_score = np.sum(result, axis=0)
	for i in range(4, 8):
		all_score[i] = all_score[i] / 5
	result = np.vstack((result, all_score))
	np_excel(result, 'mfcc/predict/result_experiment2.xlsx')

def experiment1_classifier():
	result = np.empty((5,8))
	for num in range(5):
		with open('mfcc/frog_data/experiment1/experiment1_train_mfcc_'+str(num)+'.pkl', 'rb') as train_pkl:
			train_data = pickle.load(train_pkl)
		with open('mfcc/frog_data/experiment1/experiment1_test_mfcc_'+str(num)+'.pkl', 'rb') as test_pkl:
			test_data = pickle.load(test_pkl)
		
		result[num] = linear_svm(train_data, test_data)
		print('finished:'+str(num))
	
	all_score = np.sum(result, axis=0)
	for i in range(4, 8):
		all_score[i] = all_score[i] / 5
	result = np.vstack((result, all_score))
	np_excel(result, 'mfcc/predict/result_experiment1.xlsx')

def experiment3_classifier():
	result = np.empty((5,8))
	for num in range(5):
		with open('mfcc/frog_data/experiment3/experiment3_train_mfcc_'+str(num)+'.pkl', 'rb') as train_pkl:
			train_data = pickle.load(train_pkl)
		with open('mfcc/frog_data/experiment1/experiment1_test_mfcc_'+str(num)+'.pkl', 'rb') as test_pkl:
			test_data = pickle.load(test_pkl)
		
		result[num] = linear_svm(train_data, test_data)
		print('finished:'+str(num))
	
	all_score = np.sum(result, axis=0)
	for i in range(4, 8):
		all_score[i] = all_score[i] / 5
	result = np.vstack((result, all_score))
	np_excel(result, 'mfcc/predict/result_experiment3.xlsx')

def experiment4_classifier():
	result = np.empty((5,8))
	for num in range(5):
		with open('mfcc/frog_data/experiment4/experiment4_train_mfcc_'+str(num)+'.pkl', 'rb') as train_pkl:
			train_data = pickle.load(train_pkl)
		with open('mfcc/frog_data/experiment2/experiment2_test_mfcc_'+str(num)+'.pkl', 'rb') as test_pkl:
			test_data = pickle.load(test_pkl)
		
		result[num] = linear_svm(train_data, test_data)
		print('finished:'+str(num))
	
	all_score = np.sum(result, axis=0)
	for i in range(4, 8):
		all_score[i] = all_score[i] / 5
	result = np.vstack((result, all_score))
	np_excel(result, 'mfcc/predict/result_experiment4.xlsx')



def frog1_classifier():
	result = np.empty((5,8))
	for i in range(5):
		with open('mfcc/frog_data/experiment1/train_mfcc'+str(i)+'.pkl', 'rb') as train_pkl:
			train_data = pickle.load(train_pkl)
		with open('mfcc/frog_data/experiment1/test_mfcc'+str(i)+'.pkl', 'rb') as test_pkl:
			test_data = pickle.load(test_pkl)
		result[i] = linear_svm(train_data, test_data)
	all_score = np.sum(result, axis=0)
	for i in range(4, 8):
		all_score[i] = all_score[i] / 5
	result = np.vstack((result, all_score))
	np_excel(result, 'mfcc/predict/result_experiment1.xlsx')
	

def frog2_classifier():
	result = np.empty((5,8))
	for i in range(5):
		with open('mfcc/frog_data/experiment2/train_mfcc'+str(i)+'.pkl', 'rb') as train_pkl:
			train_data = pickle.load(train_pkl)
		with open('mfcc/frog_data/experiment2/test_mfcc'+str(i)+'.pkl', 'rb') as test_pkl:
			test_data = pickle.load(test_pkl)
		result[i] = linear_svm(train_data, test_data)
	all_score = np.sum(result, axis=0)
	for i in range(4, 8):
		all_score[i] = all_score[i] / 5
	result = np.vstack((result, all_score))
	np_excel(result, 'mfcc/predict/result_experiment2.xlsx')

if __name__ == '__main__':
	#experiment3_classifier()
	experiment4_classifier()