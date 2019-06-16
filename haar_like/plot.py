# plot fg_sample as blue and no_sample as red

import numpy as np
import matplotlib.pyplot as plt

def training_data_plot(feature_method, train_dim_data, sample, way, name):
	if way == 'pca':
		x_max = max(train_dim_data[:,0]) + (max(train_dim_data[:,0]) - min(train_dim_data[:,0])) / 10
		x_min = min(train_dim_data[:,0]) - (max(train_dim_data[:,0]) - min(train_dim_data[:,0])) / 10
		y_max = max(train_dim_data[:,1]) + (max(train_dim_data[:,1]) - min(train_dim_data[:,1])) / 10
		y_min = min(train_dim_data[:,1]) - (max(train_dim_data[:,1]) - min(train_dim_data[:,1])) / 10
		plt.figure()
		plt.title(name+'_fg')
		plt.xlim(x_min, x_max)
		plt.ylim(y_min, y_max)
		plt.scatter(x=train_dim_data[:sample[0],0], y=train_dim_data[:sample[0],1], color="blue", marker="o", label="fg")# fg...o,blue
		plt.legend() # 凡例を表示
		plt.savefig(feature_method+'/pictures/'+name+'_fg')
		plt.clf()
		plt.title(name+'_no')
		plt.xlim(x_min, x_max)
		plt.ylim(y_min, y_max)
		plt.scatter(x=train_dim_data[sample[0]:,0], y=train_dim_data[sample[0]:,1], color="red", marker="x", label="no") # no...x,red
		plt.legend() # 凡例を表示
		plt.savefig(feature_method+'/pictures/'+name+'_no')
		plt.clf()
		plt.title(name)
		plt.xlim(x_min, x_max)
		plt.ylim(y_min, y_max)
		plt.scatter(x=train_dim_data[:sample[0],0], y=train_dim_data[:sample[0],1], color="blue", marker="o", label="fg")# fg...o,blue
		plt.scatter(x=train_dim_data[sample[0]:,0], y=train_dim_data[sample[0]:,1], color="red", marker="x", label="no") # no...x,red
		plt.legend() # 凡例を表示
		plt.savefig(feature_method+'/pictures/'+name)
		plt.close()
	elif way == 'lda':
		x_max = max(train_dim_data) + (max(train_dim_data) - min(train_dim_data)) / 10
		x_min = min(train_dim_data) - (max(train_dim_data) - min(train_dim_data)) / 10
		plt.figure()
		plt.title(name+'_fg')
		plt.xlim(x_min, x_max)
		y_train = np.zeros(sample[0])
		plt.scatter(x=train_dim_data[:sample[0]], y=y_train, color="blue", marker="o", label="fg")# fg...o,blue
		plt.legend() # 凡例を表示
		plt.savefig(feature_method+'/pictures/'+name+'_fg')
		plt.clf()
		plt.title(name+'_no')
		plt.xlim(x_min, x_max)
		y_train = np.zeros(sample[1])
		plt.scatter(x=train_dim_data[sample[0]:], y=y_train, color="red", marker="x", label="no") # no...x,red
		plt.legend() # 凡例を表示
		plt.savefig(feature_method+'/pictures/'+name+'_no')
		plt.clf()
		plt.title(name)
		plt.xlim(x_min, x_max)
		y_train = np.zeros(sample[0])
		plt.scatter(x=train_dim_data[:sample[0]], y=y_train, color="blue", marker="o", label="fg")# fg...o,blue
		y_train = np.zeros(sample[1])
		plt.scatter(x=train_dim_data[sample[0]:], y=y_train, color="red", marker="x", label="no") # no...x,red
		plt.legend() # 凡例を表示
		plt.savefig(feature_method+'/pictures/'+name)
		plt.close()

def select_filter_data_plot(feature_method, training_sort, sample, name):
	plt.figure()
	for i in range(6):	
		plt.title(name+':'+str(training_sort[i][2])+"[Hz]-"+str(training_sort[i][3])+"[Hz]:"+str(round(training_sort[i][0],3)))
		y_train = np.zeros(sample[0])
		plt.scatter(x=training_sort[i,4:4+sample[0]], y=y_train, color="blue", marker="o", label="fg") # fg...o,blue,0
		y_train = np.ones(sample[1])
		plt.scatter(x=training_sort[i,4+sample[0]:], y=y_train, color="red", marker="x", label="no") # no...x,red,1
		plt.axvline(training_sort[i][1], color="green") # threshold...green
		plt.legend() # 凡例を表示	
		plt.savefig(feature_method+'/pictures/'+name+'_'+str(i+1))
		plt.clf()
	plt.close()




if __name__ == '__main__':
	in_train_path1 = 'select_filters/frog_data/training_pca2.pkl'
	in_train_path2 = 'select_filters/frog_data/training_sort2.pkl'
	section = 0.5
	test_sample = load_excel_np('evaluation/test_sample_'+str(section)+'s.xlsx')
	way = 'pca'
	name1 = 'select_filters_05_pca'
	name2 = 'select_filter_05'
	filter_sample = 6
	training_data_plot(in_train_path1, test_sample[0], way, name1)
	select_filter_data_plot(in_train_path2, test_sample[0], name2)