# plot fg_sample as blue and no_sample as red

import numpy as np
import matplotlib.pyplot as plt
import pickle

def frog1_data_plot(num):
	with open('mfcc/frog_data/experiment1/train_pca'+str(num)+'.pkl', 'rb') as train_pkl:
		train_dim_data = pickle.load(train_pkl)
	del train_pkl
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
	plt.title('experiment1_mfcc'+str(num))
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	for i in range(train_dim_data.shape[0]):
		if train_dim_data[i][0] == 1:
			plt.scatter(x=train_dim_data[i][1], y=train_dim_data[i][2], color="red", marker="x", label="no")
	for i in range(train_dim_data.shape[0]):
		if train_dim_data[i][0] == 0:
			plt.scatter(x=train_dim_data[i][1], y=train_dim_data[i][2], color="blue", marker="o", label="fg")
	#plt.legend() # 凡例を表示
	plt.savefig('mfcc/pictures/experiment1/mfcc'+str(num))
	plt.close()

def frog2_data_plot(num):
	with open('mfcc/frog_data/experiment2/train_pca'+str(num)+'.pkl', 'rb') as train_pkl:
		train_dim_data = pickle.load(train_pkl)
	del train_pkl
	x_max = max(train_dim_data[:,1]) + (max(train_dim_data[:,1]) - min(train_dim_data[:,1])) / 10
	x_min = min(train_dim_data[:,1]) - (max(train_dim_data[:,1]) - min(train_dim_data[:,1])) / 10
	y_max = max(train_dim_data[:,2]) + (max(train_dim_data[:,2]) - min(train_dim_data[:,2])) / 10
	y_min = min(train_dim_data[:,2]) - (max(train_dim_data[:,2]) - min(train_dim_data[:,2])) / 10
	plt.figure()
	plt.title('experiment2_mfcc_fg'+str(num))
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	for i in range(train_dim_data.shape[0]):
		if train_dim_data[i][0] == 0:
			plt.scatter(x=train_dim_data[i][1], y=train_dim_data[i][2], color="blue", marker="o", label="fg")
	# fg...o,blue
	#plt.legend() # 凡例を表示
	plt.savefig('mfcc/pictures/experiment2/mfcc_fg'+str(num))
	plt.clf()
	plt.title('experiment2_mfcc_no'+str(num))
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	for i in range(train_dim_data.shape[0]):
		if train_dim_data[i][0] == 1:
			plt.scatter(x=train_dim_data[i][1], y=train_dim_data[i][2], color="red", marker="x", label="no")
	# no...x,red
	#plt.legend() # 凡例を表示
	plt.savefig('mfcc/pictures/experiment2/mfcc_no'+str(num))
	plt.clf()
	plt.title('experiment2_mfcc'+str(num))
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	for i in range(train_dim_data.shape[0]):
		if train_dim_data[i][0] == 1:
			plt.scatter(x=train_dim_data[i][1], y=train_dim_data[i][2], color="red", marker="x", label="no")
	for i in range(train_dim_data.shape[0]):
		if train_dim_data[i][0] == 0:
			plt.scatter(x=train_dim_data[i][1], y=train_dim_data[i][2], color="blue", marker="o", label="fg")
	#plt.legend() # 凡例を表示
	plt.savefig('mfcc/pictures/experiment2/mfcc'+str(num))
	plt.close()

if __name__ == '__main__':
	for i in range(5):
		frog1_data_plot(i)
		frog2_data_plot(i)