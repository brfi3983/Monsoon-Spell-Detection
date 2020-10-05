import numpy as np

# Loading datasets
air = np.load("C:/Users/user/Documents/Personal/Research/MachineLearningClimate19/DataSets/dataset_extended_long/raw/air_raw_4d_dataset_ext.npy")
vwnd = np.load("C:/Users/user/Documents/Personal/Research/MachineLearningClimate19/DataSets/dataset_extended_long/raw/vwnd_raw_4d_dataset_ext.npy")
uwnd = np.load("C:/Users/user/Documents/Personal/Research/MachineLearningClimate19/DataSets/dataset_extended_long/raw/uwnd_raw_4d_dataset_ext.npy")
hgt = np.load("C:/Users/user/Documents/Personal/Research/MachineLearningClimate19/DataSets/dataset_extended_long/raw/hgt_raw_4d_dataset_ext.npy")
var_names = ['air','vwnd','uwnd', 'hgt']

air = np.swapaxes(air, 0, 1)
vwnd = np.swapaxes(vwnd, 0, 1)
uwnd = np.swapaxes(uwnd, 0, 1)
hgt = np.swapaxes(hgt, 0, 1)
variables = np.array((air, vwnd, uwnd, hgt))

lat_n = 25
long_n = 33

print(variables.shape)
print(air.shape)
print(vwnd.shape)
print(uwnd.shape)
print(hgt.shape)

# Classify month functions########
def months(index):
	if 0 <= index < 16:
		return 0
	if 16 <= index < 46:
		return 1
	if 46 <= index < 77:
		return 2
	if 77 <= index < 108:
		return 3
	if 108 <= index < 138:
		return 4

def month_class(index):
	day = index % 138
	if day == 0 & index != 0:
		return 4
	else:
		return months(day)

# Concatinating Monthly Datasets######
def month_data(num,arr,May,June,July,Aug,Sep):
	if num == 0:
		May = np.vstack((May,arr))
	if num == 1:
		June = np.vstack((June,arr))
	if num == 2:
		July = np.vstack((July,arr))
	if num == 3:
		Aug = np.vstack((Aug,arr))
	if num == 4:
		Sep = np.vstack((Sep,arr))
	return [May, June, July, Aug, Sep]

############Main#############
# This runs for each variable, stores the data into monthly datasets, calculates the mean/std, 
# and then loops back through the original datasets and normalizes it
# when j == 30*138 break
for i in range (0,len(variables)):

	for j in range (0,variables[i].shape[0]):
		
		print('Starting ' + var_names[i] + ' pressure level ' + str(j+1) + '...')
		May = np.zeros((0,lat_n,long_n))
		June = np.zeros((0,lat_n,long_n))
		July = np.zeros((0,lat_n,long_n))
		Aug = np.zeros((0,lat_n,long_n))
		Sep = np.zeros((0,lat_n,long_n))

		for k in range (0,variables[i].shape[1]):

			if k == 30*138:
				break
			day = variables[i][j][k]
			day = day[np.newaxis, :]
			month = month_class(k)
			[May, June, July, Aug, Sep] = month_data(month,day,May,June,July,Aug,Sep)

		May_mean = np.mean(May)
		May_std = np.std(May)

		June_mean = np.mean(June)
		June_std = np.std(June)

		July_mean = np.mean(July)
		July_std = np.std(July)

		Aug_mean = np.mean(Aug)
		Aug_std = np.std(Aug)

		Sep_mean = np.mean(Sep)
		Sep_std = np.std(Sep)

		print('Normalizing pressure level ' + str(j+1) + '\n')
		
		for k in range(0,variables[i].shape[1]):
			if month_class(k) == 0:
				variables[i][j][k] = (variables[i][j][k] - May_mean) / May_std
			if month_class(k) == 1:
				variables[i][j][k] = (variables[i][j][k] - June_mean) / June_std
			if month_class(k) == 2:
				variables[i][j][k] = (variables[i][j][k] - July_mean) / July_std
			if month_class(k) == 3:
				variables[i][j][k] = (variables[i][j][k] - Aug_mean) / Aug_std
			if month_class(k) == 4:
				variables[i][j][k] = (variables[i][j][k] - Sep_mean) / Sep_std
	np.save(str(var_names[i]) + '_mult_norm_ext',variables[i])

