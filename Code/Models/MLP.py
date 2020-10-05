import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from numpy import newaxis
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE  # , KMeansSMOTE

from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import LSTM, Dense, Dropout, CuDNNLSTM, Flatten, Concatenate, Input, merge
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, accuracy_score
from keras.utils import to_categorical

from keras.utils import np_utils
import keras.backend as K
from itertools import product
from functools import partial
# =====================================================================================================================
# Custom loss function with costs

def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    print ('final_mask:',final_mask.shape)
    y_pred_max = K.max(y_pred, axis=1)
    print('y_pred_max:', y_pred_max.shape)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    print('y_pred_max with expand dims:', y_pred_max.shape)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    print('y_pred_max_mat:', y_pred_max_mat.shape)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))

    return K.categorical_crossentropy(y_pred, y_true) * final_mask

w_array = np.ones((3,3))
w_array[0, 1] = 1.5   #penalty for classifying a dry day as normal
w_array[0, 2] = 1.2   #penalty for classifying a dry day as wet
w_array[2, 1] = 1.5   #penalty for classifying a wet day as normal
w_array[2, 0] = 1.2   #penalty for classifying a wet day as dry

defined_loss_func = partial(w_categorical_crossentropy, weights=w_array)

#====================================================
# returns the month index and days in that month
def months(index):

	#Indexes that correspond to June, July, Aug, Sep
	if 0 <= index < 30:
		return 0, 30
	if 30 <= index < 61:
		return 1, 31
	if 61 <= index < 92:
		return 2, 31
	if 92 <= index < 122:
		return 3, 30

def month_class(index):
	day = index % 122
	if day == 0 & index != 0:
		return 3, 30
	else:
		return months(day)

def Grid2TimeArr(Grid):

	'''
	Takes in a 4d grid in form (Levels, Time Steps, Heighth, Width)
	and returns an array in form (Levels, Time Steps, 1)
	where each step is the mean from the HxW grid
	'''

	t_steps = Grid.shape[1]
	t_array = np.zeros((Grid.shape[0], t_steps))

	for j in range(0,Grid.shape[0]):
		for i in range(0,t_steps):
			m_val = Grid[j,i].mean()
			t_array[j,i] = m_val

	return t_array

# #================================================
#Creating rainfall features
'''
Note, if you are doing this (history) for Enso/Nino, 
you must take history before upscaling to 8174 format!
'''
def CreateHist(Dataset, hist):

	climate_data_new = np.zeros((Dataset.shape[1], 1))
	Dataset = Dataset[:, np.newaxis]

	for j in range(0,Dataset.shape[0]):

		climate_data_new = Dataset[j,:].T

		for i in range(1, hist):

				climate_data_hist = np.roll(Dataset[j,:].T, i, axis = 0)
				climate_data_new = np.concatenate(
					(climate_data_new, climate_data_hist), axis = 1)

		if j == 0:
			climate_data_final = climate_data_new
		else:
			climate_data_final = np.concatenate((climate_data_final, climate_data_new), axis = 1)

	x = pd.DataFrame(climate_data_final)

	return x

# ==========================================================
# This function extracts the Nino dataset for our summer months and 
# upscales them to 8174
def CreateNinoData(Dataset, hist):

	# Extracting Nino Dataset 
	Nino = np.asarray(Dataset)
	month_nino = []
	i = 0
	j = 0

	# Month indexes (i.e. Jan = 0, Feb = 1, ...) to choose from
	mon_1 = 5
	mon_2 = 9
	mons = abs(mon_1 - mon_2)

	while i < Nino.shape[0]:
		month_nino[j:(j + mons)] = Nino[(mon_1 + i):(mon_2 + i)]
		i += 12
		j += mons
	month_nino = np.asarray(month_nino)

	#Creating Nino Dataset in 8174x1 form
	day_nino = np.zeros((8174,1))
	i = 0
	j = 0
	dataset = day_nino
	for k in range (1,hist):
		month_nino = np.roll(month_nino, k, axis = 0)
		while j < month_nino.shape[0]:
			month, days = month_class(i)
			day_nino[i : (i + days)] = month_nino[j]
			i += days
			j += 1
		dataset = np.concatenate((dataset, day_nino), axis = 1)

	Nino = pd.DataFrame(dataset)
	return Nino

#==============================================================
# This function fixes the overlap when rolling lables:
# It removes the last X (Lead) from each year
def FixRollingOverlap(Dataset, Labels, Lead):

	Dataset = np.asarray(Dataset)
	Labels = np.asarray(Labels)

	Labels = np.roll(Labels, - Lead)
	Labels[:] = Labels[:] - 1

	indices = []
	i = 0
	while i < Dataset.shape[0]:
		temp_ind = np.arange((i + 122),(i + 122 + Lead))
		indices = np.concatenate((indices, temp_ind), axis = None)
		i += 122

	x = np.delete(Dataset, indices, axis = 0)
	y = np.delete(Labels, indices, axis = 0)

	x = pd.DataFrame(x)
	y = pd.DataFrame(y)
	print('Fixed X:', x.shape, 'Fixed Y:', y.shape)

	return [x,y]

#================================================================
# Creating Enso Categorical Dataset
def CreateEnsoData(Data):

	Data.Type = Data.Type.fillna('N')
	Data.Type[1] = 'ML'

	yr_enso = np.asarray(Data.Type)
	day_enso = pd.DataFrame(np.zeros((8174, 1)).astype(str))
	i = 0
	j = 0
	while j < 67:
		day_enso[i:(i+122)] = yr_enso[j]
		i += 122
		j += 1
	x = pd.get_dummies(day_enso)

	return x
#==================================================
# Standarizes the dataset with mean = 0 and std = 1
def StandarizeData(Dataset):

	# Dataset = pd.DataFrame(Dataset)

	#Standarizing
	for i in range(0,Dataset.shape[1]):
		if Dataset.all() == 0 or Dataset.all() == 1:
			continue
		Dataset[i] = (Dataset[i] - Dataset[i].mean()) / Dataset[i].std()
	return Dataset
#=====================================================
# Determines the Month for each Index and turns it into a categorical pandas dataframe
def MonthClassDataset(length):

	Month_Class_data = np.zeros((length,1))
	for i in range (0,length):
		month, days = month_class(i)
		Month_Class_data[i] = month
	
	Month_Class_data = pd.DataFrame(Month_Class_data)
	X = pd.get_dummies(Month_Class_data.astype(str))

	return X

# function: divide the set into training, validation, and test set
#============================================================
def divideIntoSets(x, y, test_ratio):

	x_train1, x_test, y_train1, y_test = train_test_split(
		x, y, test_size=test_ratio, shuffle=True)
	x_train, x_valid, y_train, y_valid = train_test_split(
		x_train1, y_train1, test_size=test_ratio, shuffle=True)

	return [x_train, x_valid, x_test, y_train, y_valid, y_test]

#====================================================================================================================
# function: count the instances of each class of the classification problem
def countClasses(classes):

	c0 = 0
	c1 = 0
	c2 = 0
	for elt in classes:
		if elt == 0:
			c0 = c0 + 1
		elif elt == 1:
			c1 = c1 + 1
		elif elt == 2:
			c2 = c2 + 1

	return [c0,c1,c2]

#================================================================
def FNN_model(X_train, y_train, x_valid, y_valid, bs, lr, ep):

	# Constants
	# Have batch size and learning rate in an array to test??

	#=====================================
	# Model Structure
	# model = Sequential()
	inputs = Input(shape=(X_train.shape[1],1))
	flat_lay = Flatten()(inputs)
	# model.add(Dropout(0.4))

	lay_1 = Dense(64, activation="tanh")(flat_lay)  # kernel_regularizer = l2(0.01)
	drop_1 = Dropout(0.4)(lay_1)
	lay_2 = Dense(64, activation="tanh")(drop_1)
	drop_2 = Dropout(0.4)(lay_2)

	testing = Concatenate()([drop_2, lay_2])
	
	# y = model.layers[-1].output
	# print(x.shape)
	# model.add(Dropout(0.4))
	# z = model.layers[-1].output
	# print(y.shape)
	# model.pop()
	# model.add(testing)
	predictions = Dense(3, activation="softmax")(testing)
	model = Model(inputs=inputs, outputs=predictions)
	#=====================================

	# Optimizing and fitting
	opt = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
	# opt = Adam(lr=lr)
	model.compile(loss=defined_loss_func,
			optimizer=opt, metrics=['accuracy'])

	model.summary()

	print("\n********Testing with a batch size of " + str(bs)
			+ " and a learning rate of " + str(lr) + "********\n")

	result = model.fit(X_train, y_train, epochs=ep, batch_size=bs, validation_data=(
		x_valid, y_valid), verbose=2, shuffle=True)

	return [model,result]
#===============================================================
# Plots the model
def PlotModelStats(history, path, filename):

	plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Var')
	plt.xlabel('Epoch')
	plt.legend(['train_loss', 'validation_loss'], loc='upper right')

	plt.savefig(path + filename + '.png')
	plt.close()
	
	# plt.show()

# function: providing different classification performance measures
#===================================================================================================================
def modelPerformanceClassificationMetrics(y_test_nor, y_pred_nor, path, filename):

	file = open(path + filename + ".txt", 'w')
	file.write("\nClassification report:\n")
	file.write(classification_report(y_test_nor, y_pred_nor))
	file.write("\n\nConfusion matrix:\n")
	file.write(np.array2string(confusion_matrix(
		y_test_nor, y_pred_nor), separator=', '))
	file.write("\n\nBalanced accuracy score:")
	file.write(str(balanced_accuracy_score(y_test_nor, y_pred_nor)))
	file.write("\n\nAccuracy:")
	file.write(str(accuracy_score(y_test_nor, y_pred_nor)))

#=====================================================================================================================
# function: convert the softmax probability of class to class label


def predictToNumeric(pred):

	pred_numeric = []
	for i in range(pred.shape[0]):
		if pred[i, 0] >= pred[i, 1] and pred[i, 0] >= pred[i, 2]:
			pred_numeric.append(0)
		elif pred[i, 1] >= pred[i, 0] and pred[i, 1] >= pred[i, 2]:
			pred_numeric.append(1)
		elif pred[i, 2] >= pred[i, 0] and pred[i, 2] >= pred[i, 1]:
			pred_numeric.append(2)
		else:
			continue

	pred_numeric = np.asarray(pred_numeric)
	print(pred.shape, pred_numeric.shape)

	return pred_numeric

def SelectVariable(num, rain_data_hist, Enso, Nino_hist, Month_classes, uwnd, vwnd):
	if num == 0:
		return rain_data_hist
	elif num == 1:
		return Enso
	elif num == 2:
		return Nino_hist
	elif num == 3:
		return Month_classes
	elif num == 4:
		return uwnd
	elif num == 5:
		return vwnd
	else:
		return -1

def ExtrCentReg(Data,levels):

	long_min = 4
	long_max = 10
	lat_north = 5
	lat_south = 7

	# Fixing Time
	Data_new = np.empty((levels + 1, 8174, Data.shape[2], Data.shape[3]))
	for i in range(0,levels+1):

		j = 0
		while j < 8174:
			Data_new[i,j:(122 + j),:,:] = Data[i,(16 + j):(16 + 122 + j),:,:]
			j += 122
	
	#Extracting Lat/Long
	x = Data_new[:, :, lat_north:lat_south, long_min:long_max]

	return x

def ExtractPredictedSamples(y_pred, y_test):

	# This array contains the index of the maximum value (per row) and a boolean for if it was correct or not
	index_flag = np.zeros((y_pred.shape[0], y_pred.shape[1]))
	for i in range(0, y_pred.shape[0]):
		max_val = 0
		max_ind = -1
		for j in range(0, y_pred.shape[1]):
			if y_pred[i][j] > max_val:
				max_val = y_pred[i][j]
				max_ind = j
		index_flag[i][max_ind] = 1
	
	# Comparing predicted to test to extract index for incorrect samples
	same_ind = []
	diff_ind = []
	for i in range(0, y_test.shape[0]):
		if (y_test[i] == index_flag[i]).all():
			same_ind.append(i)
		else:
			diff_ind.append(i)
	# print('PRED:\n',y_pred,'\nINDEX:\n',index_flag,'\nTEST:\n', y_test)
	# print('INDEX DIFF\n',diff_ind)
	# print('\nINDEX SAME\n',same_ind)

	# =================================================================
	# Creating an array of the differences between the true column and the predicted column
	true_colns = []
	pred_colns = []
	for i in range(0, y_test.shape[0]):
		for j in range(0, y_test.shape[1]):
			if y_test[i][j] == 1:
				true_colns.append(j)
			if index_flag[i][j] == 1:
				pred_colns.append(j)

	true_colns = np.array(true_colns).reshape((len(true_colns), 1))
	pred_colns = np.array(pred_colns).reshape((len(pred_colns), 1))

	temp_colns = np.concatenate((pred_colns, true_colns), axis = 1)
	vals_and_colns = np.concatenate((y_pred, temp_colns), axis = 1)
	
	# same_arr = np.delete(y_pred, diff_ind, axis = 0)
	diff_arr = np.asarray(np.delete(vals_and_colns, same_ind, axis = 0))

	return diff_arr

def GraphingDifference(arr):

	diff_arr = []
	for i in range(0, arr.shape[0]):
		# Obtains the predicted pred/true values
		true_ind = int(arr[i][4])
		pred_ind = int(arr[i][3])

		pred_true = arr[i][true_ind]
		pred_pred = arr[i][pred_ind]
		difference = abs(pred_pred - pred_true)
		diff_arr.append(difference)
	diff_arr = np.asarray(diff_arr)

	# Graphing now
	x = np.arange((diff_arr.shape[0]))
	y = diff_arr

	plt.scatter(x, y)
	plt.show()

#================================================================
# Main Function
def main():

	print('Starting Main...\n')

	#Loading in Raw Data
	rain_class_og = np.loadtxt(
		"C:/Users/user/Documents/Personal/Research/MachineLearningClimate19/DataSets/dataset_rainfall/Rainfall_Class_SC.csv", delimiter=',')
	# 	"C:/Users/user/Desktop/original_datasets/test_enso.csv", sep=',')
	# Nino = pd.read_csv(
	# 	"C:/Users/user/Desktop/original_datasets/nino_3.4_monthly.csv")
	# uwnd = np.load("C:/Users/user/Desktop/original_datasets/uwnd_cent_single_norm.npy")
	# vwnd = np.load("C:/Users/user/Desktop/original_datasets/vwnd_cent_single_norm.npy")
	# Loading in Vwnd, Uwnd, and Hgt
	uwnd_25 = np.load("C:/Users/user/Desktop/original_datasets/PressureData/uwnd_mult_norm.npy")
	vwnd_25 = np.load("C:/Users/user/Desktop/original_datasets/PressureData/vwnd_mult_norm.npy")
	hgt_25 = np.load("C:/Users/user/Desktop/original_datasets/PressureData/hgt_mult_norm.npy")

	# A n_level of 0 corresponds to surface level
	n_levels = [0,2,2]
	history = [0,2,2]

	for i in range(2,len(history)):

		rain_data = np.loadtxt("C:/Users/user/Documents/Personal/Research/MachineLearningClimate19/DataSets/dataset_rainfall/rainfall_data_SC.csv", delimiter=',')
		rain_data = rain_data[:, np.newaxis]
		rain_data = np.swapaxes(rain_data, 0, 1)

		uwnd = ExtrCentReg(uwnd_25,n_levels[i])
		vwnd = ExtrCentReg(vwnd_25,n_levels[i])
		hgt = ExtrCentReg(hgt_25,n_levels[i])

		uwnd = Grid2TimeArr(uwnd)
		vwnd = Grid2TimeArr(vwnd)
		hgt = Grid2TimeArr(hgt)

		# Putting it in row x coln form
		if history == 0:
			uwnd = np.swapaxes(uwnd, 0, 1)
			vwnd = np.swapaxes(vwnd, 0, 1)
			hgt = np.swapaxes(hgt, 0, 1)
			rain_data = np.swapaxes(rain_data, 0, 1)
		else:
			# Adding History
			uwnd = CreateHist(uwnd, history[i])
			vwnd = CreateHist(vwnd, history[i])
			hgt = CreateHist(hgt, history[i])
			rain_data = CreateHist(rain_data, history[i])

		if i == 2:
			X = pd.DataFrame(np.concatenate((uwnd, vwnd, hgt, rain_data), axis = 1))
		else:
			X = pd.DataFrame(np.concatenate((uwnd, vwnd, hgt), axis = 1))
		#Fixing shape to be like the others (8174x1 vs. 1x8174)
		# rain_class_og = rain_class_og[:, np.newaxis]

		#Creating Datasets from functions with history - Note Enso/months are constant so it is outside the loop
		# Enso = CreateEnsoData(Enso)
		# Month_classes = MonthClassDataset(8174)

		# history = [3, 4, 5]
		# batch_size = [15, 25, 100, 150]
		bs = 75
		# learning_rate = [0.00001,0.000001]
		lr = 0.00001
		epochs = 30
		# var_names = ['rain_data_hist', 'Enso', 'Nino_hist', 'Month_classes', 'uwnd', 'vwnd']
		# for i in range(0,len(history)):

		# 	print('Starting history of ' + str(history[i]) + ' days...')
			# Nino_hist = CreateNinoData(Nino, history[i])
			# rain_data_hist = RainFallHistData(rain_data, history[i])
			# print('\nEnso:', Enso.shape, 'Nino:', Nino_hist.shape, 'Rain:', rain_data_hist.shape, 'Months:', Month_classes.shape)
			
			# print(Nino_hist.shape)
			# exit()
			# X = [rain_data_hist, Enso, Nino_hist, Month_classes, uwnd, vwnd]
			# exit()
			
			
			# X = pd.DataFrame(np.concatenate((rain_data_hist, Enso, Nino_hist, Month_classes, uwnd, vwnd), axis=1))
			# print('X data:', X.shape)

			#Fixing the rollover from one end of year predicting the begining of the next (NOT consistent)
			#Note, I was not sure how to fix the monthly Nino rollover without reducing the sample size dramatically!
		lead = 3
		X, y = FixRollingOverlap(X, rain_class_og, lead)

			# for lr in learning_rate:
			# for bs in batch_size:

			# for j in range(0,6):
			# var_num = 0
			# print('\nStarting variable ' + var_names[var_num] + '...')
			# var = SelectVariable(var_num, rain_data_hist, Enso, Nino_hist, Month_classes, uwnd, vwnd)
			# var, y = FixRollingOverlap(var, rain_class_og, lead)
			
			# filename = var_names[var_num] + '_' + str(history[i]) + "_hist_" + str(lr) + "_" + str(bs)
		test_ratio = 0.15
		X = np.asarray(X)
		y = np.asarray(y)

		#Loading and fixing data to work with the model
		[X_train, x_valid, x_test, y_train, y_valid,
			y_test] = divideIntoSets(X, y, test_ratio)
		
		# c1,c2,c3 = countClasses(y_train)
		# print('classes:',c1,c2,c3)

		#Balancing Data
		X_train, y_train = SMOTE().fit_resample(X_train, y_train)
		x_valid, y_valid = SMOTE().fit_resample(x_valid, y_valid)
		x_test_over, y_test_over = SMOTE().fit_resample(x_test, y_test)

		#Fixing Axes to work with FNN
		X_train = X_train[:, :, np.newaxis]
		x_valid = x_valid[:, :, np.newaxis]
		x_test = x_test[:, :, np.newaxis]
		x_test_over = x_test_over[:, :, np.newaxis]
		
		#Standarize
		# X_train, y_train = StandarizeData(X_train), y_train

		# one hot encode
		y_train = to_categorical(y_train)
		y_test = to_categorical(y_test)
		y_test_over = to_categorical(y_test_over)
		y_valid = to_categorical(y_valid)

		filename = "reg_loss_baseline_" + str(i+1) + "_" + str(history[i]) + "_hist_" + str(n_levels[i]) + "levels_" + str(lr) + "_" + str(bs)

		path = "C:/Users/user/Desktop/original_datasets/FNN_results/"

		model, result = FNN_model(X_train, y_train, x_valid, y_valid, bs, lr, epochs)
		# PlotModelStats(result, path, filename)
		# model = load_model('C:/Users/user/Desktop/model_FNN_TEST.h5')
		model.save('C:/Users/user/Desktop/model_FNN_TEST.h5')
		exit()
		y_pred = model.predict(x_test)

		y_pred_over = model.predict(x_test_over)
		# print('y_pred shape:', y_pred_over.shape)
		# print('y_pred values',y_pred_over, y_test_over)
		diff = ExtractPredictedSamples(y_pred_over, y_test_over)
		# print(diff)
		GraphingDifference(diff)
		# print('DIFF:', diff)
		exit()
		# y_pred_nor = predictToNumeric(y_pred)
		# y_test_nor = predictToNumeric(y_test)
		y_pred_nor_over = predictToNumeric(y_pred_over)
		y_test_nor_over = predictToNumeric(y_test_over)

		modelPerformanceClassificationMetrics(y_test_nor, y_pred_nor, path, filename)
		modelPerformanceClassificationMetrics(y_test_nor_over, y_pred_nor_over, path, 'oversample_' + filename)

if __name__ == "__main__":
	main()



