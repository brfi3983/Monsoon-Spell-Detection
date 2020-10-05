from netCDF4 import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Ranges within dataset that correspond to 21N-27N, 72E-85E, and June 1 - Sep. 30
lat_north = 25
lat_south = 28
long_min = 28
long_max = 34
time_min = 151
time_max = 273
years = 67

lat_dpoints = abs(lat_south - lat_north)
long_dpoints = abs(long_max - long_min)
time_dpoints = abs(time_max - time_min)

file_extensions = ['uwnd.sig995', 'vwnd.sig995']
folders = file_extensions
variables = ['uwnd', 'vwnd']

# ======================================================
# Function: Gets data from each year and creates a single array
def GetData(year_index, VariableDataSet, var, extensions, variables):

	year = 1948 + year_index
	dataset = Dataset('C:/Users/user/Desktop/original_datasets/' + extensions[var]
	+ '/' + extensions[var] + '.' + str(year) + '.nc', 'r')

	# Leap Year
	if (dataset.dimensions['time'].size) == 366:
		time_min = 152
		time_max = 274
	else:
		time_min = 151
		time_max = 273
	
	# Creating Dataset
	dataset = dataset.variables[variables[var]][time_min:time_max,lat_north:lat_south,long_min:long_max]
	dataset = np.asarray(dataset).astype(float)

	# Concatenating
	VariableDataSet = np.vstack((VariableDataSet, dataset))

	return VariableDataSet

# =========================================================
# Main
# def main():

for var in range(0,len(variables)):

	print('Starting ' + variables[var] + ' variable...\n')
	VariableDataSet = np.zeros((0,lat_dpoints,long_dpoints), dtype = float)
	output_file_name = 'C:/Users/user/Desktop/original_datasets/' + file_extensions[var] + '_surface_cent_raw'

	for year_index in range(0,years):

		print(year_index + 1948)
		VariableDataSet = GetData(year_index, VariableDataSet, var, file_extensions, variables)

	np.save(output_file_name, VariableDataSet)

# ====================================================
# if __name__ == "__main__":
# 	main()
# # ====================================================
