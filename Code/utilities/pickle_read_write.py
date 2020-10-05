import pickle
import os
p = os.path.abspath('..')

# Read the data from pickle file
def read_data_from_pickle(file_name):
    file = open('dataset/'+file_name, 'rb')
    data = pickle.load(file)
    return data

# Write the data from pickle file
def write_data_to_pickle(file_name, data):
    file = open('dataset/'+file_name, 'wb')
    pickle.dump(data, file)
