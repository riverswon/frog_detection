import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

def frog1_data_plot():
	with open('select_filters/frog_data/experiment1/experiment1_train_pca1.pkl', 'rb') as train_pkl:
		train_dim_data = pickle.load(train_pkl)
		
	sample = np.array([561, 941, 94, 335])
	_, train_dim_data = np.hsplit(train_dim_data, [1])
	frog1, train_dim_data = np.vsplit(train_dim_data, [sample[0]])
	frog2, train_dim_data = np.vsplit(train_dim_data, [sample[1]])
	back2, back1 = np.vsplit(train_dim_data, [sample[2]])

	plt.figure()
	
	plt.errorbar(x=frog2[:470,0], y=frog2[:470,1], fmt='o', fillstyle='none', color='blue', label='label:1')
	
	plt.errorbar(x=back2[:47,0], y=back2[:47,1], fmt='o', fillstyle='none', color='yellow', label='label:3')
	plt.errorbar(x=back1[:167,0], y=back1[:167,1], fmt='o', fillstyle='none', color='green', label='label:2')
	plt.errorbar(x=frog1[:280,0], y=frog1[:280,1], fmt='o', fillstyle='none', color='red', label='label:0')
	#plt.legend()
	plt.savefig('select_filters/experiment2_pca')
	plt.show()
	sys.exit()
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
	plt.savefig('experiment2_mfcc')
	plt.show()



if __name__ == "__main__":
	frog1_data_plot()