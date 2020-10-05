from netCDF4 import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# ===================================================
# Function: Gets data from each year and concatonates it into single array

def GetData(year_index, VariableDataSet, var, lat_north, lat_south, long_min, long_max, time_min, time_max, year, levels):

	year = 1948 + year_index
	dataset = Dataset('C:/Users/user/Documents/Personal/Research/MachineLearningClimate19/original_dataset/' + var + '/' + var + '.' + str(year) + '.nc', 'r')

	# Leap Year
	if (dataset.dimensions['time'].size) == 366:
		time_min = 136
		time_max = 274
	else:
		time_min = 135
		time_max = 273
	
	# Creating Dataset
	dataset = dataset.variables[var][time_min:time_max,0:levels,lat_north:lat_south,long_min:long_max]
	dataset = np.asarray(dataset).astype(float)

	# Concatenating
	VariableDataSet = np.vstack((VariableDataSet, dataset))

	return VariableDataSet

# ====================================================
# Main
def main():

	variables = ['air', 'uwnd', 'hgt', 'vwnd']
	folders = variables

	# Indices that coorespond to lat/long in form: COOR(INDEX) -> 20S(45) to 40N(20) 60E(24) to 120E(49)
	lat_north = 20
	lat_south = 45
	long_min = 24
	long_max = 49
	time_min = 135
	time_max = 273
	years = 67
	levels = 17

	# Indices for extended region 10 deg to include madagascar
	long_min = 20
	long_max = 53

	lat_dpoints = abs(lat_south - lat_north)
	long_dpoints = abs(long_max - long_min)
	time_dpoints = abs(time_max - time_min)

	for var in range(0,len(variables)):

		print('Starting ' + variables[var] + ' variable...\n')
		VariableDataSet = np.zeros((0,levels,lat_dpoints,long_dpoints), dtype = float)
		output_file_name = variables[var] + '_raw_4d_dataset_ext'

		# Running the function for each year
		for year_index in range(0,years):
			year = year_index + 1948
			print(year)
			VariableDataSet = GetData(year_index, VariableDataSet, variables[var], lat_north, lat_south, long_min, long_max, time_min, time_max, year, levels)

		np.save(output_file_name, VariableDataSet)

if __name__ == "__main__":
	main()