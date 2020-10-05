import numpy as np
Output_filename = "ImageDataSetUnbalanced"

# Loading Datasets
Air = np.load('air_raw_3d_dataset.npy')
Air_mean = np.load('Mean_Array_air.npy')
Air_std = np.load('Std_Array_air.npy')

P_water = np.load('pr_wtr_raw_3d_dataset.npy')
P_water_mean = np.load('Mean_Array_pr_wtr.npy')
P_water_std = np.load('Std_Array_pr_wtr.npy')

Slp = np.load('slp_raw_3d_dataset.npy')
Slp_mean = np.load('Mean_Array_slp.npy')
Slp_std = np.load('Std_Array_slp.npy')

Hor_Wind = np.load('uwnd_raw_3d_dataset.npy')
Hor_Wind_mean = np.load('Mean_Array_uwnd.npy')
Hor_wind_std = np.load('Std_Array_uwnd.npy')

Vert_Wind = np.load('vwnd_raw_3d_dataset.npy')
Vert_Wind_mean = np.load('Mean_Array_vwnd.npy')
Vert_Wind_std = np.load('Std_Array_vwnd.npy')

# Scaling
Air = (Air - Air_mean) / Air_std
P_water = (P_water - P_water_mean) / P_water_std
Slp = (Slp - Slp_mean) / Slp_std
Hor_Wind = (Hor_Wind - Hor_Wind_mean) / Hor_wind_std
Vert_Wind = (Vert_Wind - Vert_Wind_mean) / Vert_Wind_std

print("Shapes of datasets:\n")
print(Air.shape)
print(P_water.shape)
print(Slp.shape)
print(Hor_Wind.shape)
print(Vert_Wind.shape)

# Saving to File
MasterDataSet = np.stack((Air, P_water, Slp, Hor_Wind, Vert_Wind), axis = 3)
np.save(Output_filename, MasterDataSet)