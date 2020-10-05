import numpy as np
import pandas as pd

def IsSpell(arr):
	spell = 0
	if np.all(arr == arr[0], axis = 0):
		spell = 1
	
	return spell

def MergeDays2Spells(arr, spell_num):
	'''
	takes in 1) numpy array of size (years, days) AND 2) the number of days to create a spell, and returns a new array of shape (x,y)
	'''
	spell_arr = np.zeros((arr.shape[0], arr.shape[1] - spell_num + 1), dtype = int)
	for i in range(0, arr.shape[0]):
		for j in range(0, arr.shape[1] - spell_num + 1):
			temp = arr[i, j:j + spell_num]
			val = temp[0] - 1
			if IsSpell(temp):
				spell_arr[i, j] = val
			else:
				spell_arr[i, j] = 1

	return spell_arr

def main():
	classes_raw = np.loadtxt("C:/Users/user/Documents/Personal/Research/MachineLearningClimate19/original_dataset/class_daily_JJAS_rainfall_central_India_1948_2014.csv", delimiter = ",")
	spell_num = 3
	test = MergeDays2Spells(classes_raw, spell_num)
	np.savetxt("C:/Users/user/Desktop/SpellOutputFile.csv", test, fmt='%i', delimiter=",")

if __name__ == "__main__":
	main()