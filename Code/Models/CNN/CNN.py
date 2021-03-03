# ======================================================================================================================
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, accuracy_score, mean_squared_error
import random
import tensorflow as tf
import time
import skimage.transform
import scipy.ndimage
from scipy.stats import pearsonr

# ===================================================================================================================
# trying to predict with "present lead" number of days ahead
present_lead = [5, 10, 15, 20]
addingrainfallflag = 0
s_flag = 2 # 1 = spell class, #2 = average rainfall spell, #3 = classification of rain days
case_num = 3
# test and validation ratio
test_ratio = 0.2131 # default is 0.15 (we use 21.31% here to match 1951-1999 (train) and 2000-2011 (test))
test_ratio1 = 0.15 # for validation

# history of days taken
hist = 3

#history of past rainfall
rainfall_hist = 3

# pressure labels of data
pres_levels = 3

epochs = 10

# path in computer and clusters
path_comp_local_moumita = "/media/moumita/Research/Files/University_Colorado/Work/work4/Spells_data_results/"
path_comp_brandon = "C:/Users/user/Documents/Personal/Research/MachineLearningClimate19/"
path_cluster = "/projects/mosa2108/Spells_data_results/"
results_folder = 'CNN_Avg_Rainfall/test'

path = path_comp_brandon


# =====================================================================================================================
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

	return pred_numeric


# ===================================================================================================================
# function: creating dataset at lead as features
def createDataSetWithLead(Dataset, Labels, lead):
	# Creating Lead Dataset, the rainfall label are rolled up by lead
	Labels = np.asarray(Labels)
	Labels = np.roll(Labels, - lead)

	# Concatenate Labels and Features
	# finding the indices corresponding to false rainfall rows(9999)
	indices = []
	for i in range(0, len(Labels)):
		if Labels[i] != 9999:
			Labels[i] = Labels[i] - 1  # labeling the class level from 0 to 2 instead of 1 to 3
		if Labels[i] == 9999:
			indices.append(i)
	indices = np.asarray(indices)

	# Delete Images samples and rainfall labels corresponding to false row (9999)
	x = np.delete(Dataset, indices, axis=0)
	y = np.delete(Labels, indices, axis=0)

	return [x, y]


# ===================================================================================================================
# function: creating dataset with history of days at lead as features
def createDataSetWithLeadandHistory(Dataset, Labels, lead, hist):
	# adding the history.......
	climate_data_new = Dataset
	for i in range(1, hist):
		climate_data_hist = np.roll(Dataset, i, axis=0)
		climate_data_new = np.concatenate((climate_data_new, climate_data_hist), axis=1)

	# Creating Lead Dataset, the rainfall label are rolled up by lead
	Labels = np.asarray(Labels)
	Labels = np.roll(Labels, - lead)

	# Concatenate Labels and Features
	# finding the indices corresponding to false rainfall rows(9999)
	indices = []
	for i in range(0, len(Labels)):
		if Labels[i] != 9999:
			Labels[i] = Labels[i] - 1  # labeling the class level from 0 to 2 instead of 1 to 3
	#     if Labels[i] == 9999:
	#         indices.append(i)
	# indices = np.asarray(indices)

	# # Delete Images samples and rainfall labels corresponding to false row (9999)
	# x = np.delete(climate_data_new, indices, axis=0)
	# y = np.delete(Labels, indices, axis=0)

	print(climate_data_new.shape, Labels.shape)

	return [climate_data_new, Labels]


# ===================================================================================================================
# function: dividing the dataset into train, validation, and test sets
def divideIntoSets(x, y, test_ratio):
	x_train1, x_test, y_train1, y_test = train_test_split(x, y, test_size=test_ratio, shuffle=False)
	x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_train1, test_size=test_ratio1, shuffle=False)

	return [x_train, x_valid, x_test, y_train, y_valid, y_test]


# ====================================================================================================================
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

	return [c0, c1, c2]


# ====================================================================================================================
# function: find the indices of the samples belonging to individual class
def findClassIndices(classes):
	c0 = []
	c1 = []
	c2 = []
	for i in range(classes.shape[0]):
		if classes[i] == 0:
			c0.append(i)
		elif classes[i] == 1:
			c1.append(i)
		elif classes[i] == 2:
			c2.append(i)

	return [c0, c1, c2]


# ====================================================================================================================
# function: randomly select "num_samples" number of samples from the set
def randomlySelect(x, y, ind, num_samples):
	random.seed(10)
	randIndex = random.sample(range(len(ind)), num_samples)
	randIndex.sort()
	selectedindex = [ind[i] for i in randIndex]

	new_x = [x[i] for i in selectedindex]
	new_y = [y[i] for i in selectedindex]

	return [new_x, new_y]


# ====================================================================================================================
# function: separating the samples classwise and randomly selecting equal number of samples from each class
# and finally shuffling all the selected samples together
def selectSamples(x, y, i0, i1, i2, num_samples):
	# for class 0
	[new_x0, new_y0] = randomlySelect(x, y, i0, num_samples)
	# for class 1
	[new_x1, new_y1] = randomlySelect(x, y, i1, num_samples)
	# for class 2
	[new_x2, new_y2] = randomlySelect(x, y, i2, num_samples)

	# concatenating all three classes
	new_x = np.concatenate((new_x0, new_x1, new_x2), axis=0)
	new_y = np.concatenate((new_y0, new_y1, new_y2), axis=0)

	# shuffling the set
	comb = list(zip(new_x, new_y))
	random.shuffle(comb)
	new_x, new_y = zip(*comb)

	new_x = np.asarray(new_x)
	new_y = np.asarray(new_y)

	return [new_x, new_y]


# ====================================================================================================================
# function: balancing all the classes by undersampling: selecting number of samples from each class as minimum number
# of samples present for any class in the original set
def balanceClassesByUndersampling(x_train1, x_valid1, x_test1, y_train1, y_valid1, y_test1):
	# count the instances of each classes
	[c0_tr, c1_tr, c2_tr] = countClasses(y_train1)
	train_sample = min(c0_tr, c1_tr, c2_tr)
	[c0_vl, c1_vl, c2_vl] = countClasses(y_valid1)
	valid_sample = min(c0_vl, c1_vl, c2_vl)
	[c0_ts, c1_ts, c2_ts] = countClasses(y_test1)
	test_sample = min(c0_ts, c1_ts, c2_ts)

	# find the indices of the corresponding classes
	[i0_tr, i1_tr, i2_tr] = findClassIndices(y_train1)
	[i0_vl, i1_vl, i2_vl] = findClassIndices(y_valid1)
	[i0_ts, i1_ts, i2_ts] = findClassIndices(y_test1)

	# select the samples from each class equal to the number of minimum number of samples in any class
	[x_train, y_train] = selectSamples(x_train1, y_train1, i0_tr, i1_tr, i2_tr, train_sample)
	[x_valid, y_valid] = selectSamples(x_valid1, y_valid1, i0_vl, i1_vl, i2_vl, valid_sample)
	[x_test, y_test] = selectSamples(x_test1, y_test1, i0_ts, i1_ts, i2_ts, test_sample)

	return [x_train, x_valid, x_test, y_train, y_valid, y_test]


# =========================================================================================================================
#### Some data augmentation techniques:  Can be used for increasing the samples (for oversampling)...........................

# ====================================================================================================================
# technique 1: salt and pepper noise: this is like binary 0 or 1 so not that useful in all cases
def addSaltPepperNoise(X_imgs):
	# Need to produce a copy as to not modify the original image
	X_imgs_copy = X_imgs.copy()
	row, col, channels = X_imgs_copy[0].shape
	salt_vs_pepper = 0.2
	amount = 0.004
	num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
	num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))

	for X_img in X_imgs_copy:
		# Add Salt noise
		coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
		X_img[coords[0], coords[1], :] = 1
		# Add Pepper noise
		coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
		X_img[coords[0], coords[1], :] = 0

	return X_imgs_copy


# ====================================================================================================================
# technique 2: adding gaussian noise
def addGaussianNoise(X_imgs, sample):
	gaussian_noise_imgs = []
	row, col, ch = X_imgs[0].shape
	mean = 0.0
	var = 0.1
	sigma = var ** 0.5

	count = 0
	while count <= sample:
		for X_img in X_imgs:
			X_img = np.array(X_img).astype(np.float32)
			gauss = np.random.normal(mean, sigma, (row, col, ch))
			gauss = gauss.reshape(row, col, ch)
			output = X_img + gauss
			count = count + 1
			if count <= sample:
				gaussian_noise_imgs.append(output)
			else:
				break

	gaussian_noise_imgs = np.asarray(gaussian_noise_imgs)

	return gaussian_noise_imgs

# ====================================================================================================================
# technique 3: flipping the image up-down or left-right
def addFlipImage(X_imgs, sample):
	flip_imgs = []
	row, col, channels = X_imgs[0].shape

	count = 0
	while count <= sample:
		for X_img in X_imgs:
			random.seed(time.time())
			sel = random.random()
			if sel <= 0.5:
				flip_img = np.flipud(X_img)
			else:
				flip_img = np.fliplr(X_img)

			count = count + 1
			if count <= sample:
				flip_imgs.append(flip_img)
			else:
				break
	flip_imgs = np.asarray(flip_imgs)

	return flip_imgs

# ====================================================================================================================
# technique 4: rotating the image with angles
def addRotateImage(X_imgs, sample):
	rotate_imgs = []
	row, col, channels = X_imgs[0].shape

	count = 0
	while count <= sample:
		for X_img in X_imgs:
			random.seed(time.time())
			randrot = random.sample(range(5, 355), 1)
			rotate_img = skimage.transform.rotate(X_img, randrot, resize=False, center=None,
												  order=1, mode='constant', cval=0, clip=True, preserve_range=False)
			count = count + 1
			if count <= sample:
				rotate_imgs.append(rotate_img)
			else:
				break

	rotate_imgs = np.asarray(rotate_imgs)

	return rotate_imgs

# technique 5: scaling the image with angles
def addScaleImage(X_imgs, sample):
	scale_imgs = []
	row, col, channels = X_imgs[0].shape
	original_size = [row, col, channels]

	count = 0
	while count <= sample:
		for X_img in X_imgs:
			random.seed(time.time())
			scalefactor = random.uniform(1.1, 2.5)
			scale_img = skimage.transform.rescale(X_img, scale=scalefactor, mode='constant')
			output = skimage.transform.resize(scale_img, original_size)  # resizing to the original shape

			count = count + 1
			if count <= sample:
				scale_imgs.append(output)
			else:
				break

	scale_imgs = np.asarray(scale_imgs)

	return scale_imgs


# ====================================================================================================================
# technique 5: Translating the image with proper padding
def addTranslateImage(X_imgs, sample):
	sess = tf.InteractiveSession()
	trans_imgs = []
	row, col, channels = X_imgs[0].shape
	original_size = [row, col, channels]
	shift = 2

	count = 0
	while count <= sample:
		for X_img in X_imgs:
			X_img = np.array(X_img).astype(np.float32)
			output = scipy.ndimage.shift(X_img, shift, output=None, order=3, mode='reflect', cval=0.0, prefilter=True)
			count = count + 1
			if count <= sample:
				trans_imgs.append(output)
			else:
				break

	trans_imgs = np.asarray(trans_imgs)

	return trans_imgs


# ====================================================================================================================
# function: separate the samples according to their classes
def separateSamplesClasswise(x, i0, i1, i2):
	x0 = np.asarray([x[i] for i in i0])
	x1 = np.asarray([x[i] for i in i1])
	x2 = np.asarray([x[i] for i in i2])

	return [x0, x1, x2]


# ====================================================================================================================
# function: selecting any of the oversampling technique for oversampling the data to make balanced classes,
# number of samples added for each class is the difference between the number of samples of specific class
# and maximum samples of any class
def selectOversamplingMethod(x0, x1, x2, sample, select, c0, c1, c2):
	if select == 1:
		x0_new = addGaussianNoise(x0, sample - c0)
		# print("Done for class 1 with generating: ", sample-c0)
		x1_new = addGaussianNoise(x1, sample - c1)
		# print("Done for class 2 with generating: ", sample-c1)
		x2_new = addGaussianNoise(x2, sample - c2)
		# print("Done for class 3 with generating: ", sample-c2)

	elif select == 2:
		x0_new = addFlipImage(x0, sample - c0)
		# print("Done for class 1 with generating: ", sample-c0)
		x1_new = addFlipImage(x1, sample - c1)
		# print("Done for class 2 with generating: ", sample-c1)
		x2_new = addFlipImage(x2, sample - c2)
		# print("Done for class 3 with generating: ", sample-c2)

	elif select == 3:
		x0_new = addRotateImage(x0, sample - c0)
		# print("Done for class 1 with generating: ", sample-c0)
		x1_new = addRotateImage(x1, sample - c1)
		# print("Done for class 2 with generating: ", sample-c1)
		x2_new = addRotateImage(x2, sample - c2)
		# print("Done for class 3 with generating: ", sample-c2)

	elif select == 4:
		x0_new = addScaleImage(x0, sample - c0)
		# print("Done for class 1 with generating: ", sample-c0)
		x1_new = addScaleImage(x1, sample - c1)
		# print("Done for class 2 with generating: ", sample-c1)
		x2_new = addScaleImage(x2, sample - c2)
		# print("Done for class 3 with generating: ", sample-c2)

	elif select == 5:
		x0_new = addTranslateImage(x0, sample - c0)
		x1_new = addTranslateImage(x1, sample - c1)
		# print("Done for class 2 with generating: ", sample-c1)
		x2_new = addTranslateImage(x2, sample - c2)
		# print("Done for class 3 with generating: ", sample-c2)

	return [x0_new, x1_new, x2_new]


# ====================================================================================================================
# function: randomly select samples from a set
def randomlySelectSamples(x, sample_added):
	random.seed(10)
	randIndex = random.sample(range(len(x)), sample_added)
	randIndex.sort()
	new_x = [x[i] for i in randIndex]

	return new_x


# ====================================================================================================================
# function: add samples to each classes to balance all the classes
def addSamplesToBalanceClasses(c0, c1, c2, max_sample, x0, x1, x2):
	x0_added, x1_added, x2_added, y0_added, y1_added, y2_added = ([] for i in range(6))

	if c0 < max_sample:  # for class 0
		sample_added = max_sample - c0
		x0_added = randomlySelectSamples(x0, sample_added)
		y0_added = []
		for i in range(sample_added):  # adding the class label
			y0_added.append(0)
		y0_added = np.asarray(y0_added)
	if c1 < max_sample:  # for class 1
		sample_added = max_sample - c1
		x1_added = randomlySelectSamples(x1, sample_added)
		y1_added = []
		for i in range(sample_added):  # adding the class label
			y1_added.append(1)
		y1_added = np.asarray(y1_added)
	if c2 < max_sample:  # for class 2
		sample_added = max_sample - c2
		x2_added = randomlySelectSamples(x2, sample_added)
		y2_added = []
		for i in range(sample_added):  # adding the class label
			y2_added.append(2)
		y2_added = np.asarray(y2_added)

	# convert the list to array
	x0_added = np.asarray(x0_added)
	x1_added = np.asarray(x1_added)
	x2_added = np.asarray(x2_added)
	y0_added = np.asarray(y0_added)
	y1_added = np.asarray(y1_added)
	y2_added = np.asarray(y2_added)

	x_added = np.concatenate((x0_added, x1_added, x2_added), axis=0)
	y_added = np.concatenate((y0_added, y1_added, y2_added), axis=0)

	return [x_added, y_added]

# ====================================================================================================================
# function: concatenate equal samples from each class to prepare the final dataset
def assembleAddSamplesToBalanceClasses(x0, x1, x2):
	# convert the list to array
	x0 = np.asarray(x0)
	x1 = np.asarray(x1)
	x2 = np.asarray(x2)

	y_added = []
	for i in range(x0.shape[0]):  # adding the class label 0
		y_added.append(0)
	for i in range(x1.shape[0]):  # adding the class label 1
		y_added.append(1)
	for i in range(x2.shape[0]):  # adding the class label 2
		y_added.append(2)
	y_added = np.asarray(y_added)

	if x0.shape[0] > 0 and x1.shape[0] > 0 and x2.shape[0] > 0:
		x_added = np.concatenate((x0, x1, x2), axis=0)
	elif x0.shape[0] == 0 and x1.shape[0] > 0 and x2.shape[0] > 0:
		x_added = np.concatenate((x1, x2), axis=0)
	elif x0.shape[0] > 0 and x1.shape[0] == 0 and x2.shape[0] > 0:
		x_added = np.concatenate((x0, x2), axis=0)
	elif x0.shape[0] > 0 and x1.shape[0] > 0 and x2.shape[0] == 0:
		x_added = np.concatenate((x0, x1), axis=0)

	return [x_added, y_added]


# ====================================================================================================================
# function: comcatenate the original unbalanced dataset with the oversampled samples
# and shuffle them thouroughly to generate the final balanced dataset
def augmentAndShuffle(x_old, y_old, x_added, y_added):
	# y_added = y_added.reshape(y_added.shape[0], 1)

	x = np.concatenate((x_old, x_added), axis=0)
	y = np.concatenate((y_old, y_added), axis=0)

	##shuffling the set
	comb = list(zip(x, y))
	random.seed(10)
	random.shuffle(comb)
	x, y = zip(*comb)

	x = np.asarray(x)
	y = np.asarray(y)

	return [x, y]


# ====================================================================================================================
# function: Balance the dataset by oversampling technique
def balanceClassesByOversampling(x_train1, x_valid1, x_test1, y_train1, y_valid1, y_test1, sel):
	# count the instances of each classes
	[c0_tr, c1_tr, c2_tr] = countClasses(y_train1)
	train_sample = max(c0_tr, c1_tr, c2_tr)
	[c0_vl, c1_vl, c2_vl] = countClasses(y_valid1)
	valid_sample = max(c0_vl, c1_vl, c2_vl)
	[c0_ts, c1_ts, c2_ts] = countClasses(y_test1)
	test_sample = max(c0_ts, c1_ts, c2_ts)

	# find the indices of the corresponding classes
	[i0_tr, i1_tr, i2_tr] = findClassIndices(y_train1)
	[i0_vl, i1_vl, i2_vl] = findClassIndices(y_valid1)
	[i0_ts, i1_ts, i2_ts] = findClassIndices(y_test1)

	# separate the samples of each class
	[x0_tr, x1_tr, x2_tr] = separateSamplesClasswise(x_train1, i0_tr, i1_tr, i2_tr)
	[x0_vl, x1_vl, x2_vl] = separateSamplesClasswise(x_valid1, i0_vl, i1_vl, i2_vl)
	[x0_ts, x1_ts, x2_ts] = separateSamplesClasswise(x_test1, i0_ts, i1_ts, i2_ts)

	# technique to generate the more data by any one of the data augmentation technique

	[x0_train_new, x1_train_new, x2_train_new] = selectOversamplingMethod(x0_tr, x1_tr, x2_tr, train_sample, sel, c0_tr,
																		  c1_tr, c2_tr)
	[x0_valid_new, x1_valid_new, x2_valid_new] = selectOversamplingMethod(x0_vl, x1_vl, x2_vl, valid_sample, sel, c0_vl,
																		  c1_vl, c2_vl)
	[x0_test_new, x1_test_new, x2_test_new] = selectOversamplingMethod(x0_ts, x1_ts, x2_ts, test_sample, sel, c0_ts,
																	   c1_ts, c2_ts)

	# assemble the added classes with class labels
	[x_train_added, y_train_added] = assembleAddSamplesToBalanceClasses(x0_train_new, x1_train_new, x2_train_new)
	[x_valid_added, y_valid_added] = assembleAddSamplesToBalanceClasses(x0_valid_new, x1_valid_new, x2_valid_new)
	[x_test_added, y_test_added] = assembleAddSamplesToBalanceClasses(x0_test_new, x1_test_new, x2_test_new)

	# # making equal number of samples for all the classes in training, valid, and test sets
	# # [x_train_added, y_train_added] = addSamplesToBalanceClasses(c0_tr, c1_tr, c2_tr, train_sample, x0_train_new, x1_train_new, x2_train_new)
	# # [x_valid_added, y_valid_added] = addSamplesToBalanceClasses(c0_vl, c1_vl, c2_vl, valid_sample, x0_valid_new, x1_valid_new, x2_valid_new)
	# # [x_test_added, y_test_added] = addSamplesToBalanceClasses(c0_ts, c1_ts, c2_ts, test_sample, x0_test_new, x1_test_new, x2_test_new)

	# adding the data augmentation with original data and shuffling them

	[x_train, y_train] = augmentAndShuffle(x_train1, y_train1, x_train_added, y_train_added)
	[x_valid, y_valid] = augmentAndShuffle(x_valid1, y_valid1, x_valid_added, y_valid_added)
	[x_test, y_test] = augmentAndShuffle(x_test1, y_test1, x_test_added, y_test_added)

	return [x_train, x_valid, x_test, y_train, y_valid, y_test]

# ===================================================================================================================
# function: train the CNN model for the classification task
def trainCNNModel(x_train, y_train, x_valid, y_valid, bs, lr, epochs):
	# Model
	model = Sequential()
	model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=x_train.shape[1:], data_format='channels_first'))
	model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
	model.add(Dropout(0.6))

	model.add(Conv2D(filters=64, kernel_size=(3, 3), data_format='channels_first'))
	model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
	model.add(Dropout(0.6))


	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	model.add(Dense(16, activation='relu'))
	model.add(Dense(1, activation='linear'))

	opt = Adam(lr=lr)
	model.compile(optimizer=opt, loss='mean_squared_error')
	model.summary()

	# I added class_weight = 'auto' since the data was unbalanced
	history = model.fit(x_train, y_train, batch_size=bs, epochs=epochs, validation_data=(x_valid, y_valid),
						verbose=2)  # ,class_weight='auto')

	return [model, history]

# ===================================================================================================================
# function: plot the model losses for training and validation period
def plotModelAccuracy(history, filename):
	plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Var')
	plt.xlabel('Epoch')
	plt.legend(['train_loss', 'validation_loss'], loc='upper right')
	plt.savefig(path + 'results/' + results_folder + '/' + filename + '.png')
	plt.close()

def NomalizeCmatrixRow(cmatrix):
	row_sum = cmatrix.sum(axis=1, dtype='float').T
	cmatrix = cmatrix / row_sum[:, None]

	return cmatrix
# ===================================================================================================================
# function: providing different classification performance measures
def modelPerformanceClassificationMetrics(y_test_nor, y_pred_nor, filename, normflag):
	file = open(path + 'results/'+ results_folder + '/' + filename + ".txt", 'w')
	file.write("\nClassification report:\n")
	file.write(classification_report(y_test_nor, y_pred_nor))
	file.write("\n\nConfusion matrix:\n")

	# Normalizing
	cmatrix = confusion_matrix(y_test_nor, y_pred_nor)
	if normflag == 1:
		cmatrix = NomalizeCmatrixRow(cmatrix)

	file.write(np.array2string(cmatrix, separator=', '))
	file.write("\n\nBalanced accuracy score:")
	file.write(str(balanced_accuracy_score(y_test_nor, y_pred_nor)))
	file.write("\n\nAccuracy:")
	file.write(str(accuracy_score(y_test_nor, y_pred_nor)))

# function: providing different regression performance measures
def modelPerformanceRegressionMetrics(y_test, y_pred, filename):
	file = open(path + 'results/'+ results_folder + '/' + filename + ".txt", 'w')

	error = mean_squared_error(y_test, y_pred)
	file.write("\nMean Squred Error:\n")
	file.write(str(error))

	y_pred = y_pred.astype(float).reshape(y_pred.shape[0],)
	y_test = y_test.astype(float).reshape(y_pred.shape[0],)

	r_coeff = np.corrcoef(y_pred, y_test)
	file.write("\n\nPearson Coefficient:\n")
	file.write(str(r_coeff[0, 1]))

# ===================================================================================================================
def getFromSingleVariable(lead, file_x, file_y, flag_pres):
	# Importing Datasets (in form of image: where the axis corresponds to lattitude and longitude)
	Dataset = np.load(path + file_x)
	print("$$$$$",Dataset.shape)

	if flag_pres == 1:  # datasets with multiple pressure levels
		# considering the lowest three levels for air temperature from Earth surface
		Dataset = Dataset[:, 0:pres_levels, :, :]
	else:  # datasets only with one pressure levels or surface
		Dataset = Dataset[:, :, :]
		[r1, r2, r3] = Dataset.shape
		Dataset = np.reshape(Dataset, (r1, 1, r2, r3))

	# Reading the rainfall classes along with added 999 for the lead days
	# (Before June-- required for adjustment while considering the history)
	Labels = np.load(path + file_y)

	# creating the dataset and labels with lead and history
	[x, y] = createDataSetWithLeadandHistory(Dataset, Labels, lead, hist)

	return [x, y]

# ======================================================================================================================

def addRainfallHistoryInCNNImage(file_x, row, col, rainfall_hist):
	# Importing Datasets (in form of image: where the axis corresponds to lattitude and longitude)
	Dataset = np.load(path + file_x, allow_pickle=True)

	newImageDataset = []
	for elt in Dataset:
		temp = [[elt]*row,]*col
		newImageDataset.append(temp)

	newImageDataset = np.asarray(newImageDataset)

	#putting the channel axis as second instead of last as the others
	newImageDataset = np.swapaxes(newImageDataset,1,3)

	# adding the history.......
	climate_data_new = newImageDataset
	for i in range(1, rainfall_hist):
		climate_data_hist = np.roll(newImageDataset, i, axis=0)
		climate_data_new = np.concatenate((climate_data_new, climate_data_hist), axis=1)
	print(climate_data_new.shape)

	return climate_data_new

# ======================================================================================================================

def removeGarbageRowsWithRainfall(x, x6, y):

	# Concatenate Labels and Features
	# finding the indices corresponding to false rainfall rows(9999)
	indices = []
	# selecting the indices whose labels class corresponds to 9999
	for i in range(0, len(y)):
		if y[i] == 9999:
			indices.append(i)

	#selecting the indices whose rainfall history values coresponds to 9999
	# (all the row and cols values are same as the same identical value is repeated for creating the whole image)
	for i in range(0, x6.shape[0]):
		flag = 0
		for j in range(0,x6.shape[1]):
			if x6[i,j,0,0]==9999:
				flag = 1
				break
		if flag == 1:
			indices.append(i)

	indices = np.asarray(indices)
	indices = np.unique(indices)
	indices = np.sort(indices)

	# Delete Images samples and rainfall labels corresponding to false row (9999)
	x = np.delete(x, indices, axis=0)
	y = np.delete(y, indices, axis=0)

	return [x, y]

# ======================================================================================================================

def removeGarbageRows(x, y):

	# Concatenate Labels and Features
	# finding the indices corresponding to false rainfall rows(9999)
	indices = []
	# selecting the indices whose labels class correspondds to 9999
	for i in range(0, len(y)):
		if y[i] == 9999:
			indices.append(i)

	indices = np.asarray(indices)
	indices = np.sort(indices)

	# Delete Images samples and rainfall labels corresponding to false row (9999)
	x = np.delete(x, indices, axis=0)
	y = np.delete(y, indices, axis=0)

	return [x, y]

def ImportVariables(lead):
	if s_flag == 1:
		# path = 'DataSets/dataset_rainfall/SpellOutputFile.npy'
		path = 'DataSets/dataset_rainfall/SpellOutputFile1_padded_9999_oneD.npy'
	elif s_flag == 2:
		path = 'DataSets/dataset_rainfall/RainAvgOutputFile_padded_9999.npy'
	elif s_flag == 3:
		path = 'DataSets/dataset_rainfall/rainfall_class_withproxy9999_new.npy'
	else:
		print('Error, not a valid target file')
		exit()

	[x3, y3] = getFromSingleVariable(lead, 'DataSets/dataset_CNN/CNN_new_bigger_region/uwnd_multi_norm.npy', path, flag_pres=1)
	[x4, y4] = getFromSingleVariable(lead, 'DataSets/dataset_CNN/CNN_new_bigger_region/vwnd_multi_norm.npy', path, flag_pres=1)
	[x5, y5] = getFromSingleVariable(lead, 'DataSets/dataset_CNN/CNN_new_bigger_region/hgt_multi_norm.npy', path, flag_pres=1)

	[timestep, cha, row, col] = x3.shape

	x6 = addRainfallHistoryInCNNImage('DataSets/dataset_rainfall/rainfall_values_withproxy9999_new.npy', row, col, rainfall_hist)

	if addingrainfallflag == 1:
		x = np.concatenate((x3, x4, x5, x6), axis=1)
		y = y3
		[x, y] = removeGarbageRowsWithRainfall(x, x6, y)
	else:
		x = np.concatenate((x3, x4, x5), axis=1)
		y = y3
		[x, y] = removeGarbageRows(x, y)

	return x, y
def ImportMeanStdCases(y):
	mean_path = "C:/Users/user/Documents/Personal/Research/MachineLearningClimate19/DataSets/dataset_rainfall/Mean_pentad_longPeriod(1961-1990)_triples_samescale_rainfall.pkl"
	std_path = "C:/Users/user/Documents/Personal/Research/MachineLearningClimate19/DataSets/dataset_rainfall/Std_pentad_longPeriod(1961-1990)_triples_samescale_rainfall.pkl"

	mean = np.load(mean_path, allow_pickle=True)
	std = np.load(std_path, allow_pickle=True)

	n = int(len(y) / len(mean))
	mean = np.tile(mean, n).reshape(len(y), 1)
	std = np.tile(std, n).reshape(len(y), 1)

	# check axis operations!!
	case1 = y
	case2 = y - mean
	case3 = (y - mean) / std
	return case1, case2, case3

def ChangeCase(y, case_num):
	case1, case2, case3 = ImportMeanStdCases(y)
	if case_num == 1:
		result = case1
	elif case_num == 2:
		result = case2
	elif case_num == 3:
		result = case3

	return result

# ======================================================================================================================
# function: the main function ....
def main():

	for lead in present_lead:

		x, y = ImportVariables(lead)

		# Removing first and last 3 years in samples to match 1951 - 2011 period
		x = x[3*120:-3*120,:,:,:]
		y = y[3*120:-3*120,:]

		print('1:', y.shape)
		# Changing case for correlation coefficient
		y = ChangeCase(y, case_num)
		print('3:', y.shape)

		# # Splitting Data into training and test
		# Average Rainfall!!!
		if s_flag == 2:
			[x_train, x_valid, x_test, y_train, y_valid, y_test] = divideIntoSets(x, y, test_ratio)

		# Classification!!!
		if s_flag != 2:
			select = 2
			[x_train1, x_valid1, x_test1, y_train1, y_valid1, y_test1] = divideIntoSets(x, y, test_ratio)
			[x_train, x_valid, x_test, y_train, y_valid, y_test] = balanceClassesByOversampling(x_train1, x_valid1, x_test1,
																								y_train1, y_valid1, y_test1,
																								select)

		#  running the models for a number of gridSearch variables
		# applying grid search
		batch_size = [25]  # , 50, 75, 100, 150, 200]
		learn_rate = [0.00001]  # , 0.0001]
		# momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
		for bs in batch_size:
			for lr in learn_rate:
				# Training
				[model, history] = trainCNNModel(x_train, y_train, x_valid, y_valid, bs, lr,epochs)

				# Plotting
				plotModelAccuracy(history, 'case_' + str(case_num) + '_3V_loss_batch-size_' + str(bs) + '_learn-rate_' + str(lr) +
								'_varpressureLayers_' + str(pres_levels) + '_history_' + str(hist) + '_lead_' + str(
					lead) + 'with_rainfall_history_' + str(rainfall_hist))

				# Classification!!!
				if s_flag != 2:
					# Testing OVERSAMPLED =====================
					y_pred_over = model.predict(x_test)
					y_pred_nor_over = predictToNumeric(y_pred_over)

					# Confusion Matrix
					modelPerformanceClassificationMetrics(y_test, y_pred_nor_over,
														'over_3V_result_batch-size_' + str(bs) + '_learn-rate_' + str(lr) +
														'_varpressureLayers_' + str(pres_levels) + 'history_' + str(
															hist) + '_lead_' + str(lead)+'with_rainfall_history_'+str(rainfall_hist), normflag=0)
					# Testing NON-OVERSMAPLED ===============
					y_pred = model.predict(x_test1)
					y_pred_nor = predictToNumeric(y_pred)

					# Confusion Matrix
					modelPerformanceClassificationMetrics(y_test1, y_pred_nor,
														'3V_result_batch-size_' + str(bs) + '_learn-rate_' + str(lr) +
														'_varpressureLayers_' + str(pres_levels) + 'history_' + str(
															hist) + '_lead_' + str(lead)+'with_rainfall_history_'+str(rainfall_hist), normflag=1)
				# Regression (Average Rainfall)!!!
				if s_flag == 2:
					# Testing OVERSAMPLED =====================
					y_pred = model.predict(x_test)

					# Confusion Matrix
					modelPerformanceRegressionMetrics(y_test, y_pred,
														'case_' + str(case_num) + '_3V_result_batch-size_' + str(bs) + '_learn-rate_' + str(lr) +
														'_varpressureLayers_' + str(pres_levels) + 'history_' + str(
															hist) + '_lead_' + str(lead)+'with_rainfall_history_'+str(rainfall_hist))

# ===================================================================================================================

if __name__ == '__main__':
	main()

# ===================================================================================================================