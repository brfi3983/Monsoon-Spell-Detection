from netCDF4 import Dataset
import numpy as np
import csv
import os
p = os.path.abspath('..')


def load_dataset(file_name):
    sst_data = Dataset(file_name, 'r')
    return sst_data


def select_lat_lon(sst_data, latvar, lonvar, s_lat, e_lat, s_lon, e_lon):
    selected_lat = np.array(sst_data.variables[latvar][s_lat:e_lat])
    selected_lon = np.array(sst_data.variables[lonvar][s_lon:e_lon])

    return selected_lat, selected_lon


def reduced_grid_locations(selected_lat, selected_lon, grid_size, grid_resolution):
    # numbering on grid
    lat_lon_location = []
    reduction_factor = int(grid_size/grid_resolution)

    # print(reduction_factor, len(selected_lat), len(selected_lon))
    reduced_selected_lat = []
    for i in range(0, len(selected_lat)-1, reduction_factor):
        lat_sum = 0
        for x in range(reduction_factor):
            lat_sum += selected_lat[i+x]
        reduced_selected_lat.append(lat_sum/reduction_factor)

    reduced_selected_lon = []
    for j in range(0, len(selected_lon)-1, reduction_factor):
        lon_sum = 0
        for y in range(reduction_factor):
            lon_sum += selected_lon[j+y]
        reduced_selected_lon.append(lon_sum/reduction_factor)

    k = 0
    for lat in reduced_selected_lat:
        for lon in reduced_selected_lon:
            lat_lon_location.append([k, lat, lon])
            k += 1
    return lat_lon_location


def write_csv_file(file_name, lat_lon_location):
    file = open('dataset/'+file_name, 'w', newline='')
    file.fieldnames = ['grid_number', 'lat', 'lon']
    writer = csv.writer(file, delimiter=',')
    for row in lat_lon_location:
        writer.writerow(row)

