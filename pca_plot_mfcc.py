import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import pandas as pd

#change numpy array into excel
def np_excel(sheet, path):
    df = pd.DataFrame(sheet)
    df.to_excel(path)

def load_excel_np(path):
    df = pd.read_excel(path)
    df = df.values
    return df

def frog1_data_plot():
	with open('mfcc/frog_data/experiment1/experiment1_train_pca1.pkl', 'rb') as train_pkl:
		train_dim_data = pickle.load(train_pkl)
	sample1 = load_excel_np('audio_data/cross_validation_sample1.xlsx')
	sample2 = load_excel_np('audio_data/cross_validation_sample2.xlsx')
	sample1, x = np.vsplit(sample1, [1])
	_, x = np.vsplit(x, [1])
	sample1 = np.vstack((sample1, x))
	sample2, x = np.vsplit(sample2, [1])
	_, x = np.vsplit(x, [1])
	sample2 = np.vstack((sample2, x))
	frog1 = np.empty(2)
	frog2 = np.empty(2)
	back1 = np.empty(2)
	back2 = np.empty(2)
	for i in range(4):
		x, train_dim_data = np.vsplit(train_dim_data, [sample1[i][0]])
		frog1 = np.vstack((frog1, x))
		x, train_dim_data = np.vsplit(train_dim_data, [sample1[i][1]])
		back1 = np.vstack((back1, x))
		x, train_dim_data = np.vsplit(train_dim_data, [sample2[i][0]])
		frog2 = np.vstack((frog2, x))
		x, train_dim_data = np.vsplit(train_dim_data, [sample2[i][1]])
		back2 = np.vstack((back2, x))
	frog1 = np.delete(frog1, 0, 0)
	frog2 = np.delete(frog2, 0, 0)
	back1 = np.delete(back1, 0, 0)
	back2 = np.delete(back2, 0, 0)
	plt.figure()
	plt.errorbar(x=frog1[:,0], y=frog1[:,1], fmt='o', fillstyle='none', color='red', label='label:0')
	plt.errorbar(x=frog2[:,0], y=frog2[:,1], fmt='o', fillstyle='none', color='blue', label='label:1')
	plt.errorbar(x=back1[:,0], y=back1[:,1], fmt='o', fillstyle='none', color='green', label='label:2')
	plt.errorbar(x=back2[:,0], y=back2[:,1], fmt='o', fillstyle='none', color='yellow', label='label:3')
	#plt.legend()
	plt.savefig('mfcc/experiment1_pca')
	plt.show()
	sys.exit()



	sys.exit()
	sample_fg1 = [141, 140, 140, 140]
	sample_fg2 = [236, 235, 235, 235]
	sample_bk1 = [84, 84, 84, 83]
	sample_bk2 = [24, 23, 23, 24]
	frog1, train_dim_data = np.vsplit(train_dim_data, [141])
	back1, train_dim_data = np.vsplit(train_dim_data, [84])
	frog2, train_dim_data = np.vsplit(train_dim_data, [236])
	back2, train_dim_data = np.vsplit(train_dim_data, [24])
	back1 = np.vstack((back1, back2))
	x, train_dim_data = np.vsplit(train_dim_data, [140])
	frog1 = np.vstack((frog1, x))
	x, train_dim_data = np.vsplit(train_dim_data, [84])
	back1 = np.vstack((back1, x))
	_, train_dim_data = np.vsplit(train_dim_data, [235])
	x, train_dim_data = np.vsplit(train_dim_data, [23])
	back1 = np.vstack((back1, x))
	
	#frog1, x = np.vsplit(train_dim_data, [sample1])
	#frog2, back = np.vsplit(x, [sample2])
	"""
	x_max = max(train_dim_data[:,1]) + (max(train_dim_data[:,1]) - min(train_dim_data[:,1])) / 10
	x_min = min(train_dim_data[:,1]) - (max(train_dim_data[:,1]) - min(train_dim_data[:,1])) / 10
	y_max = max(train_dim_data[:,2]) + (max(train_dim_data[:,2]) - min(train_dim_data[:,2])) / 10
	y_min = min(train_dim_data[:,2]) - (max(train_dim_data[:,2]) - min(train_dim_data[:,2])) / 10
	plt.figure()
	plt.title('experiment1_mfcc_fg'+str(num))
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	for i in range(train_dim_data.shape[0]):
		if train_dim_data[i][0] == 0:
			plt.scatter(x=train_dim_data[i][1], y=train_dim_data[i][2], color="blue", marker="o", label="fg")
	# fg...o,blue
	#plt.legend() # 凡例を表示
	plt.savefig('mfcc/pictures/experiment1/mfcc_fg'+str(num))
	plt.clf()
	plt.title('experiment1_mfcc_no'+str(num))
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	for i in range(train_dim_data.shape[0]):
		if train_dim_data[i][0] == 1:
			plt.scatter(x=train_dim_data[i][1], y=train_dim_data[i][2], color="red", marker="x", label="no")
	# no...x,red
	#plt.legend() # 凡例を表示
	plt.savefig('mfcc/pictures/experiment1/mfcc_no'+str(num))
	plt.clf()
	"""
	plt.figure()
	#plt.title('experiment1_mfcc'+str(num))
	#plt.xlim(x_min, x_max)
	#plt.ylim(y_min, y_max)
	plt.errorbar(x=back1[:200,0], y=back1[:200,1], fmt='o', fillstyle='none', color='green')
	plt.errorbar(x=frog2[:200,0], y=frog2[:200,1], fmt='o', fillstyle='none', color='blue')
	plt.errorbar(x=frog1[:200,0], y=frog1[:200,1], fmt='o', fillstyle='none', color='red')
	plt.show()
	plt.savefig('experiment2_mfcc')



if __name__ == "__main__":
	frog1_data_plot()