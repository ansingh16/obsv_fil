import numpy as np


data = np.loadtxt('Data_all_mass.csv',delimiter=',',skiprows=1)
data_clusters = np.loadtxt('Groups_and_clusters.csv',delimiter=',',skiprows=1)
data_filaments = np.loadtxt('Filament_output.csv',delimiter=',',skiprows=1)
data_fields = np.loadtxt('Field_galaxies.csv',delimiter=',',skiprows=1)


print data.shape,data_clusters.shape,data_filaments.shape,data_fields.shape


