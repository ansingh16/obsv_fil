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
#np.setdiff1d(array1, array2)
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({'figure.max_open_warning': 0})



radius = 3.0
limit_log = 3.0


def clip_data(data):
    data = data[data[:, 0] > min(data[:, 0] + 5.0)]
    data = data[data[:, 0] < max(data[:, 0] - 5.0)]

    data = data[data[:, 1] > min(data[:, 1] + 5.0)]
    data = data[data[:, 1] < max(data[:, 1] - 5.0)]

    return data


data_coma = np.genfromtxt('all_data_paraview.csv',delimiter=',',skip_header=1)


file_RADEC_array = np.array([data_coma[:,3],data_coma[:,4]]).T
file_RADEC_array = clip_data(file_RADEC_array)
#print min(file_RADEC_array[:,0]),max(file_RADEC_array[:,0])



file_data_pos = file_RADEC_array


def clustering(r,sam):

    dbsc = DBSCAN(eps = r, min_samples = sam).fit(file_RADEC_array)
    labels = dbsc.labels_
    core_samples = np.zeros_like(labels, dtype = bool)
    core_samples[dbsc.core_sample_indices_] = True
    labels = dbsc.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    return core_samples,labels,n_clusters_


R = []
Num_clus = []


#FINDING CLUSTERS AND GROUPS USING DBSCAN

minnum=20
for r in np.arange(0.1,2.0,0.1):
    fig1,ax1 = plt.subplots(1,1,figsize=(8,6))

    #print r
    R.append(r)
    core_samples1,labels1,n_clusters1_ = clustering(r,minnum)
    size_array=[]

    for i in range(n_clusters1_):
        size_array.append(file_RADEC_array[labels1==i].shape[0])

    #print min(size_array),max(size_array),n_clusters1_
    Num_clus.append(n_clusters1_)
    '''
    ax1.scatter(file_RADEC_array[core_samples1][:,0],file_RADEC_array[core_samples1][:,1],s=1,color='red')
    ax1.set_xlim(max(file_RADEC_array[:,0]),min(file_RADEC_array[:,0]))
    ax1.set_ylim(min(file_RADEC_array[:,1]),max(file_RADEC_array[:,1]))
    ax1.set_xlabel('RA')
    ax1.set_ylabel('DEC')
    ax1.set_title('eps='+ str(r)+'(deg)')
    fig1.savefig('cluster'+str(minnum)+'_'+str(r)+'.png')
    '''

'''
fig2,ax2 = plt.subplots(1,1,figsize=(8,6))
ax2.plot(R,Num_clus,'k')
ax2.set_xlabel('Maximum Radius(deg)')
ax2.set_ylabel('Number of groups detected')

fig2.savefig('eps_vs_Nclus'+str(minnum)+'.png',dpi=600)
'''

R = np.array(R)
Num_clus = np.array(Num_clus)

eps_r = R[np.where(Num_clus==max(Num_clus))][0]
core_samples, labels, n_clusters_ = clustering(eps_r, minnum)

'''
fig2,ax2 = plt.subplots(1,1,figsize=(8,6))
ax2.scatter(file_RADEC_array[np.logical_not(core_samples)][:,0],file_RADEC_array[np.logical_not(core_samples)][:,1],s=1,color='red')
ax2.scatter(file_RADEC_array[core_samples][:,0],file_RADEC_array[core_samples][:,1],s=1,color='green')
ax2.set_xlim(max(file_RADEC_array[:, 0]), min(file_RADEC_array[:, 0]))
ax2.set_ylim(min(file_RADEC_array[:, 1]), max(file_RADEC_array[:, 1]))
fig2.savefig('no_cluster1.png')
'''


data_no_cluster = file_RADEC_array[np.logical_not(core_samples)]


#READ THE LENGTHS OF THE FILAMENTS


lengths = np.loadtxt('Distances60.dat')

#FOR THE FILAMENTS GREATER THAN THE 10MPC LENGTH CALCULATE THE GALAXIES WITHIN THE 3MPC DISTANCE FROM THE DATA
#HAVING NO CLUSTER

mean_z = 0.023
C = mean_z*(c.to('km/s'))/cosmo.H(0.0)

import os
i=0
k=0
for file in os.listdir("."):
    if file.startswith("filament_"):

        if(lengths[i]>10.0):

            filament_data = np.loadtxt(file)


            #calculate the distances
            Sky_all_points = SkyCoord(ra=data_no_cluster[:,0] * u.degree, dec=data_no_cluster[:,1] * u.degree, distance=C)
            Filament_all_points = SkyCoord(ra=filament_data[:, 0] * u.degree, dec=filament_data[:, 1] * u.degree,distance=C)
            fil_value, sky_values, sep, dist = search_around_3d(Filament_all_points, Sky_all_points,distlimit=radius * u.Mpc)

            data = np.array([fil_value, sky_values, sep, dist]).T
            data = data[data[:, 3].argsort()]
            data_fil_ind = data[:, 0].astype(int)
            data_gal_ind = data[:, 1].astype(int)
            data_dist_gal = data[:, 3]
            nearby_gal, indexes = np.unique(data_gal_ind, return_index=True)
            near_fil_point = data_fil_ind[indexes]
            dist_fil = data_dist_gal[indexes]

            d = np.zeros((len(nearby_gal), 1))


            for j in range(len(nearby_gal)):

                if (near_fil_point[j] == 0):
                    # p2=np.array([RA_fil[0],RA_fil[1]])
                    P1 = SkyCoord(ra=filament_data[0,0] * u.degree, dec=filament_data[0,1] * u.degree, distance=C)
                    P2 = SkyCoord(ra=filament_data[1,0] * u.degree, dec=filament_data[1,1] * u.degree, distance=C)
                    p1 = np.array([P1.cartesian.x.value, P1.cartesian.y.value]).T
                    p2 = np.array([P2.cartesian.x.value, P2.cartesian.y.value]).T

                    near_halo_RA = data_no_cluster[nearby_gal[j],0]
                    near_halo_DEC = data_no_cluster[nearby_gal[j],1]

                    P3 = SkyCoord(ra=near_halo_RA * u.degree, dec=near_halo_DEC * u.degree, distance=C)
                    p3 = np.array([P3.cartesian.x.value, P3.cartesian.y.value]).T

                    d[j, 0] = np.abs(
                        (p2[1] - p1[1]) * p3[0] - (p2[0] - p1[0]) * p3[1] + p2[0] * p1[1] - p2[1] * p1[0]) / np.sqrt(
                        (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

                    if (d[j, 0] > dist_fil[j]):
                        d[j, 0] = dist_fil[j]





                elif (0 < near_fil_point[j] < (len(filament_data[:,0]) - 2)):

                    # print near_fil_point[i + 1],i,

                    P1 = SkyCoord(ra=filament_data[near_fil_point[j],0] * u.degree, dec=filament_data[near_fil_point[j],1] * u.degree,distance=C)
                    P2 = SkyCoord(ra=filament_data[near_fil_point[j] + 1,0] * u.degree,dec=filament_data[near_fil_point[j] + 1,1] * u.degree, distance=C)
                    p1 = np.array([P1.cartesian.x.value, P1.cartesian.y.value]).T
                    p2 = np.array([P2.cartesian.x.value, P2.cartesian.y.value]).T

                    near_halo_RA = data_no_cluster[nearby_gal[j],0]
                    near_halo_DEC = data_no_cluster[nearby_gal[j],1]

                    P3 = SkyCoord(ra=near_halo_RA * u.degree, dec=near_halo_DEC * u.degree, distance=C)
                    p3 = np.array([P3.cartesian.x.value, P3.cartesian.y.value]).T

                    tem_dist1 = np.abs(
                        (p2[1] - p1[1]) * p3[0] - (p2[0] - p1[0]) * p3[1] + p2[0] * p1[1] - p2[1] * p1[0]) / norm(p2 - p1)

                    P1 = SkyCoord(ra=filament_data[near_fil_point[j] - 1,0] * u.degree,dec=filament_data[near_fil_point[j] - 1,1] * u.degree, distance=C)
                    P2 = SkyCoord(ra=filament_data[near_fil_point[j],0] * u.degree, dec=filament_data[near_fil_point[j],1] * u.degree,distance=C)
                    p1 = np.array([P1.cartesian.x.value, P1.cartesian.y.value]).T
                    p2 = np.array([P2.cartesian.x.value, P2.cartesian.y.value]).T

                    near_halo_RA = data_no_cluster[nearby_gal[j],0]
                    near_halo_DEC = data_no_cluster[nearby_gal[j],1]

                    P3 = SkyCoord(ra=near_halo_RA * u.degree, dec=near_halo_DEC * u.degree, distance=C)
                    p3 = np.array([P3.cartesian.x.value, P3.cartesian.y.value]).T

                    tem_dist2 = np.abs(
                        (p2[1] - p1[1]) * p3[0] - (p2[0] - p1[0]) * p3[1] + p2[0] * p1[1] - p2[1] * p1[0]) / norm(p2 - p1)

                    # print d.shape, len(tem_dist2)


                    if (tem_dist1 < tem_dist2):
                        d[j, 0] = tem_dist1
                    else:
                        d[j, 0] = tem_dist2

                    if (dist_fil[j] < d[j]):
                        d[j, 0] = dist_fil[j]



                elif (near_fil_point[j] == (len(filament_data[:,0]) - 1)):

                    #print filament_data[near_fil_point[j],0],filament_data[near_fil_point[j],1],C,j

                    P2 = SkyCoord(ra=filament_data[near_fil_point[j] - 1,0] * u.degree,dec=filament_data[near_fil_point[j] - 1,1] * u.degree, distance=C)
                    P1 = SkyCoord(ra=filament_data[near_fil_point[j],0] * u.degree,dec=filament_data[near_fil_point[j],1] * u.degree, distance=C)
                    p1 = np.array([P1.cartesian.x.value, P1.cartesian.y.value]).T
                    p2 = np.array([P2.cartesian.x.value, P2.cartesian.y.value]).T

                    near_halo_RA = data_no_cluster[nearby_gal[j],0]
                    near_halo_DEC = data_no_cluster[nearby_gal[j],1]

                    P3 = SkyCoord(ra=near_halo_RA * u.degree, dec=near_halo_DEC * u.degree, distance=C)
                    p3 = np.array([P3.cartesian.x.value, P3.cartesian.y.value]).T

                    d[j, 0] = np.abs(
                        (p2[1] - p1[1]) * p3[0] - (p2[0] - p1[0]) * p3[1] + p2[0] * p1[1] - p2[1] * p1[0]) / norm(p2 - p1)

                    if (dist_fil[j] < d[j]):
                        d[j, 0] = dist_fil[j]



            fig2, ax2 = plt.subplots(1, 1)
            ax2.plot(filament_data[:,0],filament_data[:,1],linewidth=2.5, color='red')
            ax2.scatter(data_no_cluster[nearby_gal,0], data_no_cluster[nearby_gal,1], c=d[:,0], s=1.5, cmap='jet')

            ax2.set_xlim(max(file_RADEC_array[:, 0]), min(file_RADEC_array[:, 0]))
            ax2.set_ylim(min(file_RADEC_array[:, 1]), max(file_RADEC_array[:, 1]))
            fig2.savefig('FILAMENT_IMAGE'+str(i)+'.png',dpi=600)

        k=k+1

        i=i+1




