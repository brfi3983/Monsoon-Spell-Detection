import pandas as pd
import numpy as np
import csv

# path in computer and clusters
path_comp_moumita = "/media/moumita/Research/Files/University_Colorado/Work/work4/Spells_data_results/"
path_comp_brandon = ""
path_cluster = "/projects/mosa2108/spells/"

path = path_cluster

def main():

    # rainfall classes.....................................................................
    labels = pd.read_csv(path+"DataSets/dataset_rainfall/9999_Rainfall_Class_SC.csv")
    print(labels.shape)
    labels = np.asarray(labels)

    # removing the 9999 and only taking the rainfall classes from June to Sepetember
    newlabel = []
    for ind in range(labels.shape[0]):
        if labels[ind,0]!=9999:
            newlabel.append(labels[ind,0])

    print(len(newlabel))

    # adding the 9999 for May 15 to May 31 of each year and make the dataset as from May 15 to Sep 30 of each year
    number9999 = 16
    rainfall = 122 # days of june 1 to September 30

    finallabel = []
    ind = 0
    while ind < len(newlabel):
        for i in range(number9999):   #adding proxy values for May 15 to May 31
            finallabel.append(9999)

        for i in range(rainfall):     #adding rainfall for June 1 to Sep 30
            finallabel.append(newlabel[ind])
            ind = ind +1

    finallabel = np.transpose(np.asarray(finallabel))
    finallabel = np.reshape(finallabel,(finallabel.shape[0],1))
    print(finallabel.shape)

    np.save(path+"DataSets/dataset_rainfall/rainfall_class_withproxy9999_new.npy", finallabel)

    # rainfall values.......................................................................
    rainfall = pd.read_csv(path+"DataSets/dataset_rainfall/rainfall_data_SC.csv")
    print(rainfall.shape)
    rainfall = np.asarray(rainfall)

    # adding the 9999 for May 15 to May 31 of each year and make the dataset as from May 15 to Sep 30 of each year
    number9999 = 16
    numberrainfall = 122 # days of june 1 to September 30

    finalrainfall = []
    ind = 0
    while ind < rainfall.shape[0]:
        for i in range(number9999):   #adding proxy values for May 15 to May 31
            finalrainfall.append(9999)

        for i in range(numberrainfall):     #adding rainfall for June 1 to Sep 30
            finalrainfall.append(rainfall[ind])
            ind = ind +1

    finalrainfall = np.transpose(np.asarray(finalrainfall))
    finalrainfall = np.reshape(finalrainfall,(finalrainfall.shape[0],1))
    print(finalrainfall.shape)

    np.save(path+"DataSets/dataset_rainfall/rainfall_values_withproxy9999_new.npy", finalrainfall)


if __name__=='__main__':
    main()