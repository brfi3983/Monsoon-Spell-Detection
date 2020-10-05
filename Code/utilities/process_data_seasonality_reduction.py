from utilities.prepare_data_for_selected_grid import *

# get dataset for 3 decades {1951-1965} {1977-1991} {2003-2018}
# 1951 = index 1164 in dataset
def generate_decade_dataset(data, var, s_time, e_time, s_lat, e_lat, s_lon, e_lon):

    if var=='sst': #sst data available from 1854
        total_dataset = data[var][s_time:e_time, 0, s_lat:e_lat, s_lon:e_lon]
        _30_years_dataset = data[var][1392:1752, 0, s_lat:e_lat, s_lon:e_lon]
    elif var == ('air' or 'slp'): #other data available from 1948
        total_dataset = data[var][s_time:e_time, s_lat:e_lat, s_lon:e_lon]
        _30_years_dataset = data[var][264:624, s_lat:e_lat, s_lon:e_lon]
    else:  #data have pressure level (we are selecting at 200 hecPascal pressure level
        total_dataset = data[var][s_time:e_time, 9, s_lat:e_lat, s_lon:e_lon]
        _30_years_dataset = data[var][264:624, 9, s_lat:e_lat, s_lon:e_lon]

    return total_dataset, _30_years_dataset

# Convert the selected dataset (180, 30, 80) dim data into (180, 15, 40) dim dataset combine {2} grids, {4x4}
# parameter grid size. to make 4x4 grid, provide 4.
# Since dataset is already 2x2, no need to worry about it.
# but it should be multiple of dataset grid size.
def convert_data_to_reduced_grid_size(decade_dataset, grid_size, grid_resolution):
    reduction_factor = int(grid_size/grid_resolution)
    grid_shape = decade_dataset.shape
    reduced_grid_shape = (grid_shape[0], int(grid_shape[1]/reduction_factor), int(grid_shape[2]/reduction_factor))
    decade_dataset_reduced_grid_size = np.zeros(shape=reduced_grid_shape)

    for i in range(0, grid_shape[1]-1, reduction_factor):
        for j in range(0, grid_shape[2]-1, reduction_factor):
            is_masked = False
            masked_list = [0] * grid_shape[0]
            for xi in range(reduction_factor):
                for xj in range(reduction_factor):
                    if decade_dataset[:,i+xi, j+xj].mask.all():
                        is_masked = True
                        masked_list = decade_dataset[:,i+xi, j+xj]

            if is_masked:
                decade_dataset_reduced_grid_size[:, int(i / reduction_factor), int(j / reduction_factor)] = masked_list
            else:
                series_sum = [0]*grid_shape[0]
                num_elem = 0
                for x in range(reduction_factor):
                    for y in range(reduction_factor):
                        # print(i+x, "::", j+y)
                        series_sum += decade_dataset[:, i+x, j+y]
                        num_elem += 1
                series_sum_np = np.array(series_sum)
                decade_dataset_reduced_grid_size[:, int(i/reduction_factor), int(j/reduction_factor)] = series_sum_np/num_elem

    return decade_dataset_reduced_grid_size


# # Get list of land locations
# def get_land_locations(decade_dataset_reduced_grid):
#     land_locations = []
#     grid_shape = decade_dataset_reduced_grid.shape
#     num_lats = grid_shape[1]
#     num_lons = grid_shape[2]
#     k = 0
#     for i in range(num_lats):
#         for j in range(num_lons):
#             if np.mean(decade_dataset_reduced_grid[:, i, j]) == -999.0:
#                 land_locations.append(k)
#             k += 1
#     return land_locations

# Get list of land locations
def get_land_locations(decade_dataset,count):
    land_locations = []
    grid_shape = decade_dataset.shape
    num_lats = grid_shape[1]
    num_lons = grid_shape[2]
    k = 0
    selected_point = 0
    # selected_grid_points = []
    actual_grid_point_locations = [0]*count
    for i in range(num_lats):
        for j in range(num_lons):
            if decade_dataset[:, i, j].mask.all():
                land_locations.append(k)
            else:
                # mapping = [selected_point, k]
                actual_grid_point_locations[selected_point] = k
                selected_point += 1
                # selected_grid_points.append(mapping)
            k += 1
    # print(selected_grid_points)
    # print(actual_grid_point_locations)
    return land_locations, actual_grid_point_locations


# Get list of land locations
def get_land_locations_for_reduced_grid(decade_dataset_reduced_grid):
    land_locations = []
    grid_shape = decade_dataset_reduced_grid.shape
    num_lats = grid_shape[1]
    num_lons = grid_shape[2]
    k = 0
    for i in range(num_lats):
        for j in range(num_lons):
            if np.mean(decade_dataset_reduced_grid[:, i, j]) == -999.0:
                land_locations.append(k)
            k += 1
    return land_locations

# Create monthly avg for all locations on grid and store in a 2D table
def func_30_years_monthly_avg_per_location(_30_years_dataset_reduced_grid, n_years=30):
    grid_shape = _30_years_dataset_reduced_grid.shape
    grid_size = grid_shape[1]*grid_shape[2]
    locations_timeseries_table = []*grid_size

    for lat in range(_30_years_dataset_reduced_grid.shape[1]):
        for lon in range(_30_years_dataset_reduced_grid.shape[2]):
            monthly_sum1 = [0]*12
            if np.mean(_30_years_dataset_reduced_grid) != -999.0 :
                for ii in range(_30_years_dataset_reduced_grid.shape[0]):
                    monthly_sum1[ii%12] = monthly_sum1[ii%12] + _30_years_dataset_reduced_grid[ii, lat, lon]
            monthly_avg1 = [x / n_years for x in monthly_sum1]
            locations_timeseries_table.append(monthly_avg1)
    return locations_timeseries_table


# subtract the average values from the time series and Create timeseries for all locations on grid and store in a 2D table
def reduce_seasonality_per_location(decade_dataset_reduced_grid, land_locations, locations_timeseries_table):
    grid_shape = decade_dataset_reduced_grid.shape
    grid_size = grid_shape[1] * grid_shape[2]
    decade_dataset_seasonalty_reduced = np.zeros(grid_shape)
    decade1_grid_seasonality_reduced_grid_reduced = []*grid_size

    i = 0
    for lat in range(decade_dataset_reduced_grid.shape[1]):
        for lon in range(decade_dataset_reduced_grid.shape[2]):
            if(i not in land_locations):
                time_series = [] * grid_shape[0]
                for ii in range(decade_dataset_reduced_grid.shape[0]):
                    decade_dataset_seasonalty_reduced[ii, lat, lon] = decade_dataset_reduced_grid[ii, lat, lon] - locations_timeseries_table[i][ii%12]
                    time_series.append(decade_dataset_seasonalty_reduced[ii, lat, lon])
                decade1_grid_seasonality_reduced_grid_reduced.append(time_series)
            i += 1
    return decade1_grid_seasonality_reduced_grid_reduced
