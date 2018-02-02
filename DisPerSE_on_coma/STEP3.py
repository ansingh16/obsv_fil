import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.constants import c
from astropy.cosmology import WMAP9 as cosmo
from scipy import spatial
from astropy.coordinates import search_around_3d
from sympy.geometry import *
from numpy.linalg import norm
# np.setdiff1d(array1, array2)
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import pandas as pd

plt.rcParams.update({'figure.max_open_warning': 0})

radius = 6.0
limit_log = 3.0


def clip_data(data):
    data = data[data[:, 3] > min(data[:, 3] + 5.0)]
    data = data[data[:, 3] < max(data[:, 3] - 5.0)]

    data = data[data[:, 4] > min(data[:, 4] + 5.0)]
    data = data[data[:, 4] < max(data[:, 4] - 5.0)]
    return data



data_coma = np.genfromtxt('Coma_large_smriti.csv', delimiter=',')

# file_RADEC_array = np.array([data_coma[:,3],data_coma[:,4]]).T
data_coma = clip_data(data_coma)
print data_coma.shape

file_RADEC_array = np.array([data_coma[:, 3], data_coma[:, 4]]).T
save_data_array = np.array([data_coma[:, 0], data_coma[:, 1], data_coma[:, 2]]).T


# READ THE LENGTHS OF THE FILAMENTS


# FOR THE FILAMENTS GREATER THAN THE 10MPC LENGTH CALCULATE THE GALAXIES WITHIN THE 3MPC DISTANCE FROM THE DATA
# HAVING NO CLUSTER

mean_z = 0.023
C = mean_z * (c.to('km/s')) / cosmo.H(0.0)




nearest_points_RA = []
nearest_points_Dec = []

lengths = np.loadtxt('Distances60.dat')

from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from shapely.geometry import MultiPoint

data_crit = np.loadtxt('Critical_Points.dat',delimiter=',')

#colorbar_ax = fig1.add_axes([0.92, 0.1, 0.01, 0.8])
#cbar = plt.colorbar(s_m,cax=colorbar_ax)
#cbar.set_label('Length(Mpc)')
data_crit = data_crit[data_crit[:,0]==2.0]





Sky_all_points = SkyCoord(ra=data_coma[:, 3] * u.degree, dec=data_coma[:, 4] * u.degree, distance=C)
maxima_all_points = SkyCoord(ra=data_crit[:, 1] * u.degree, dec=data_crit[:, 2] * u.degree,distance=C)


maxima_value, sky_values, sep, dist = search_around_3d(maxima_all_points, Sky_all_points,distlimit=radius * u.Mpc)



data = np.array([maxima_value, sky_values, sep, dist]).T
data = data[data[:, 3].argsort()]
data_max_ind = data[:, 0].astype(int)
data_gal_ind = data[:, 1].astype(int)
data_dist_gal = data[:, 3]
nearby_gal, indexes = np.unique(data_gal_ind, return_index=True)
near_fil_point = data_max_ind[indexes]
dist_max = data_dist_gal[indexes]



d0 = data_coma[nearby_gal,0]
d1 = data_coma[nearby_gal,1]
d2 = data_coma[nearby_gal,2]
dr = data_coma[nearby_gal,3]
dd = data_coma[nearby_gal,4]
ddist = dist_max

fdata = np.array([d0,d1,d2,dr,dd,dist_max]).T

np.savetxt('galaxies_within_6Mpc_from_maxima.csv',fdata,delimiter=',',fmt='%d,%d,%d,%f,%f,%f',header='#plate,mjd,fiberID,RA,Dec,dist(Mpc)')


fig,ax = plt.subplots(1,1,figsize=(8,6))


ax.scatter(data_coma[:,3],data_coma[:,4],s=1.5,color='black',alpha=0.4)

cm = ax.scatter(data_coma[nearby_gal,3],data_coma[nearby_gal,4],s=1.5,c=ddist,cmap='jet')

ax.scatter(data_crit[:,1],data_crit[:,2],s=20.0,color='red')


ax.set_xlim(max(data_coma[:,3]),min(data_coma[:,3]))
ax.set_ylim(min(data_coma[:,4]),max(data_coma[:,4]))
ax.set_xlabel('RA')
ax.set_ylabel('Dec')

#colorbar_ax = fig.add_axes([0.15, 0.02, 0.8, 0.02])
cbar1 = plt.colorbar(cm)
cbar1.set_label('distance(Mpc)')

print dr.shape,data_coma.shape

#fig.colorbar(cm)
fig.tight_layout()

fig.savefig('galaxies_within_6Mpc_from_maxima_color_distance.png')

plt.show()