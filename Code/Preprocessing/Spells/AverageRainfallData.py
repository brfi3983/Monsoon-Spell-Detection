import numpy as np
import pandas as pd

def IsSpell(arr):
	spell = 0
	if np.all(arr == arr[0], axis = 0):
		spell = 1
	
	return spell

def Merge2RainAverage(arr, spell_num):
	'''
	takes in 1) numpy array of size (years, days) AND 2) the number of days to create a spell,\
		 and returns a new array of shape (x,y) with each cell being the average of the spell length
	'''
	rain_arr = np.zeros((arr.shape[0], arr.shape[1] - spell_num + 1), dtype = float)
	for i in range(0, arr.shape[0]):
		for j in range(0, arr.shape[1] - spell_num + 1):
			temp = arr[i, j:j + spell_num]
			val = temp.mean()
			rain_arr[i, j] = val

	return rain_arr

def main():
	classes_raw = np.loadtxt("C:/Users/user/Documents/Personal/Research/MachineLearningClimate19/original_dataset/normalized_daily_JJAS_rainfall_central_India_1948_2014.csv", delimiter = ",")
	print(classes_raw.shape)
	spell_num = 3

	test = Merge2RainAverage(classes_raw, spell_num)
	print(test.shape, test)
	np.savetxt("C:/Users/user/Desktop/RainAvgOutputFile.csv", test, fmt='%f', delimiter=",")

if __name__ == "__main__":
	main()