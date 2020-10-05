import numpy as np
from utilities.prepare_data_for_selected_grid import *
from utilities.pickle_read_write import *
from utilities.process_data_seasonality_reduction import *

def main():

    #=================================================================================================
    # load data from the data file
    # data = load_dataset('data.nc')
    data = load_dataset('dataset/air.2m.gauss.1948.nc')
    print(data.dimensions.keys())
    print(data.dimensions)
    print(data.variables['lat'][:])
    print(data.variables['lon'][:])
    print(data.dimensions['lat'])
    print(data.dimensions['lon'])
    print(data.dimensions['time'])
    print(data.variables['air'][24,45,:])

    # #=================================================================================================
    # # all parameters for at data
    # grid_size = 2.5
    # grid_resolution = 2.5
    # var = 'air'
    #
    # if var=='sst':
    #     latvar = 'Y'
    #     lonvar ='X'
    #     s_lat = 5
    #     e_lat = 85
    #     s_lon = 0
    #     e_lon = 180
    #     s_time = 1152
    #     e_time = 1980
    # else:
    #     latvar = 'lat'
    #     lonvar = 'lon'
    #     s_lat = 4
    #     e_lat = 69
    #     s_lon = 0
    #     e_lon = 144
    #     s_time = 24
    #     e_time = 852
    #
    # #at: (present 1854 to feb2019)--data selected from Jan 1950 to Dec 2018 (index: 1153:1980)
    # #selected latitude: 80 S to 80 N, and longitude: all
    # #(data is 2 degree resolution==index--lat: 5:85, lon:1:180)
    # # get selected lat {5, 85} lon {1,180} time{1152:1980}
    # # print(latvar,lonvar)
    # selected_lat, selected_lon = select_lat_lon(data, latvar, lonvar, s_lat, e_lat, s_lon, e_lon)
    # print(selected_lat)
    # print(selected_lon)

    # #=================================================================================================
    # #preparing the grid points...........................................
    # # combine the selected grid to a reduced grid {10x10} ie {5} grid points along x and y in 1 point
    # # 5 points combined will form a single point in reduced grid.
    # # pass the grid_size as 10 if you want 10x10 grid size.
    # # pass the grid_resolution as 2 if data is 2x2 grid format
    # # lat_lon_location = reduced_grid_locations(selected_lat, selected_lon, grid_size, grid_resolution)
    # #
    # # # write the lat lon location in csv file for future reference
    # # write_csv_file('vwnd_grid_point_reduced_degree'+str(grid_size)+'.csv', lat_lon_location)
    #
    # # =================================================================================================
    # # preparing the data at reduced grids ...........................................
    # # Generate dataset for Jan 1950 to Dec 2018
    # total_dataset, _30_years_dataset = generate_decade_dataset(data, var, s_time, e_time, s_lat, e_lat, s_lon, e_lon)
    #
    # # # Convert the selected dataset (180, 30, 80) dim data into (180, 15, 40) dim dataset combine {2} grids, {4x4}
    # # pass the grid_resolution as 2 if data is 2x2 grid format
    # total_dataset_reduced_grid = convert_data_to_reduced_grid_size(total_dataset, grid_size, grid_resolution)
    # _30_years_dataset_reduced_grid = convert_data_to_reduced_grid_size(_30_years_dataset, grid_size, grid_resolution)

if __name__=='__main__':
    main()