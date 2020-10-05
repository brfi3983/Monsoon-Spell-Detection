import numpy as np

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

# =======================================
# Concatinating Monthly Datasets
def month_data(num,arr,June,July,Aug,Sep):
	if num == 0:
		June = np.vstack((June,arr))
	if num == 1:
		July = np.vstack((July,arr))
	if num == 2:
		Aug = np.vstack((Aug,arr))
	if num == 3:
		Sep = np.vstack((Sep,arr))
	return [June, July, Aug, Sep]

# ================================================
# Main
def main():

	# Loading datasets
	# air = np.load('C:/Users/user/Desktop/original_datasets/air_surface_raw.npy')

	vwnd = np.load('C:/Users/user/Desktop/original_datasets/vwnd.sig995_surface_cent_raw.npy')
	uwnd = np.load('C:/Users/user/Desktop/original_datasets/uwnd.sig995_surface_cent_raw.npy')
	variables = np.array((vwnd, uwnd))
	var_names = ['vwnd', 'uwnd']

	# Checking Shapes
	print(variables.shape)
	# print(air.shape)
	print(vwnd.shape)
	print(uwnd.shape)

	days = 122
	lat = 3
	long = 6

	for i in range (0,len(variables)):

		'''
		This runs for each variable, stores the data into monthly datasets, calculates the mean/std,
		and then loops back through the original datasets and normalizes it
		when j == 30*days break
		'''

		print('Starting ' + var_names[i])
		May = np.zeros((0,lat,long))
		June = np.zeros((0,lat,long))
		July = np.zeros((0,lat,long))
		Aug = np.zeros((0,lat,long))
		Sep = np.zeros((0,lat,long))

		for j in range (0,variables[i].shape[0]):

			if j == 30*days:
				break
			day = variables[i][j]
			day = day[np.newaxis, :]
			month = month_class(j)
			[June, July, Aug, Sep] = month_data(month,day,June,July,Aug,Sep)

		# May_mean = np.mean(May)
		# May_std = np.std(May)

		June_mean = np.mean(June)
		June_std = np.std(June)

		July_mean = np.mean(July)
		July_std = np.std(July)

		Aug_mean = np.mean(Aug)
		Aug_std = np.std(Aug)

		Sep_mean = np.mean(Sep)
		Sep_std = np.std(Sep)

		print('Normalizing...\n')

		for j in range(0,variables[i].shape[0]):
			# if month_class(j) == 0:
			# 	variables[i][j] = (variables[i][j] - May_mean) / May_std
			if month_class(j) == 0:
				variables[i][j] = (variables[i][j] - June_mean) / June_std
			if month_class(j) == 1:
				variables[i][j] = (variables[i][j] - July_mean) / July_std
			if month_class(j) == 2:
				variables[i][j] = (variables[i][j] - Aug_mean) / Aug_std
			if month_class(j) == 3:
				variables[i][j] = (variables[i][j] - Sep_mean) / Sep_std

		np.save(str(var_names[i]) + '_cent_single_norm',variables[i])

if __name__ == "__main__":
	main()
