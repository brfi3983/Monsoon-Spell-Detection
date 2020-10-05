from netCDF4 import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

file_extensions = ['air.2m.gauss','uwnd.10m.gauss','pr_wtr.eatm','slp','vwnd.10m.gauss']
folders = ['Air Temperature', 'Horizontal Wind','Precipitation Water', 'Sea Level Pressure','Vertical Wind']
Variables = ['air','uwnd','pr_wtr','slp','vwnd']

# Ranges within dataset that correspond to 21N-27N, 72E-85E, and May 20-Sep. 30
lat_max = 35
lat_min = 32
long_min = 38
long_max = 45
time_min = 141
time_max = 273

years = 67
days = 132
lat_dpoints = 3
long_dpoints = 7

def GetData (year_index, VariableDataSet):

	year = 1948 + year_index
	dataset = Dataset('C:/Users/user/Documents/Personal/Research/Climate Variable Data/' + folders[var] + '/' + file_extensions[var] + '.' + str(year) + '.nc', 'r')
	dataset = dataset.variables[Variables[var]][time_min:time_max,lat_min:lat_max,long_min:long_max]
	dataset = np.asarray(dataset).astype(float)
	
	# Concatenating
	VariableDataSet = np.vstack((VariableDataSet, dataset))

	return VariableDataSet

def SaveMeanStd (Array):
	mean_array = Array.mean(axis = 0)
	std_array = Array.std(axis = 0)
	print("Shape of Mean Array: ", mean_array.shape)
	print("Shape of Std Array: ", std_array.shape)

	np.save('Mean_Array_' + Variables[var] ,mean_array)
	np.save('Std_Array_' + Variables[var] ,std_array)

# ===============Main================
for var in range (0,len(Variables)):

	print('Starting ' + Variables[var] + ' variable...\n')
	VariableDataSet = np.zeros((0,3,7), dtype = float)
	output_file_name = Variables[var] + '_raw_3d_dataset'

	for year_index in range(0,30):
		print(year_index + 1948)
		VariableDataSet = GetData(year_index, VariableDataSet)

	np.save(output_file_name, VariableDataSet)
	SaveMeanStd(VariableDataSet)