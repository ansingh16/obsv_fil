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
import pandas as pd


plt.rcParams.update({'figure.max_open_warning': 0})



radius = 3.0
limit_log = 3.0



def clip_data(data):
    data = data[data[:, 3] > min(data[:, 3] + 5.0)]
    data = data[data[:, 3] < max(data[:, 3] - 5.0)]

    data = data[data[:, 4] > min(data[:, 4] + 5.0)]
    data = data[data[:, 4] < max(data[:, 4] - 5.0)]
    return data


def clustering(r,sam):

    dbsc = DBSCAN(eps = r, min_samples = sam).fit(file_RADEC_array)
    labels = dbsc.labels_
    core_samples = np.zeros_like(labels, dtype = bool)
    core_samples[dbsc.core_sample_indices_] = True
    labels = dbsc.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    return core_samples,labels,n_clusters_



data_coma = np.genfromtxt('Coma_large_smriti.csv',delimiter=',')

#file_RADEC_array = np.array([data_coma[:,3],data_coma[:,4]]).T
data_coma = clip_data(data_coma)
print data_coma.shape



file_RADEC_array = np.array([data_coma[:,3],data_coma[:,4]]).T
save_data_array = np.array([data_coma[:,0],data_coma[:,1],data_coma[:,2]]).T


#print min(file_RADEC_array[:,0]),max(file_RADEC_array[:,0])
file_data_pos = file_RADEC_array


R = []
Num_clus = []

fil_file = open('FILAMENTS'+'.dat', 'r')
lines = fil_file.readlines()
total_filaments = int(lines[0])

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


R = np.array(R)
Num_clus = np.array(Num_clus)
eps_r = R[np.where(Num_clus==max(Num_clus))][0]
core_samples, labels, n_clusters_ = clustering(eps_r, minnum)


data_cluster = file_RADEC_array[core_samples]

print 'Total shape',save_data_array.shape
savedata_groups = np.array(save_data_array[core_samples],dtype=np.int32)

print 'Groups shape',savedata_groups.shape
#print len(core_samples),file_RADEC_array.shape,savedata_groups.shape,save_data_array.shape
np.savetxt('Groups_and_clusters.csv',savedata_groups,fmt='%d,%d,%d',header='plate,mjd,fiberID',comments='#')



data_no_cluster = file_RADEC_array[np.logical_not(core_samples)]
save_data_filaments = np.array(save_data_array[np.logical_not(core_samples)],dtype=np.int32)

print 'Without clusters',data_no_cluster.shape


#READ THE LENGTHS OF THE FILAMENTS


lengths = np.loadtxt('Distances60.dat')
print lengths.shape
#FOR THE FILAMENTS GREATER THAN THE 10MPC LENGTH CALCULATE THE GALAXIES WITHIN THE 3MPC DISTANCE FROM THE DATA
#HAVING NO CLUSTER

mean_z = 0.023
C = mean_z*(c.to('km/s'))/cosmo.H(0.0)

fig2, ax2 = plt.subplots(1, 1,figsize=(10,8))
ax2.set_xlim(max(file_RADEC_array[:, 0]), min(file_RADEC_array[:, 0]))
ax2.set_ylim(min(file_RADEC_array[:, 1]), max(file_RADEC_array[:, 1]))



import matplotlib
parameters = np.linspace(0,10,11)
# norm is a class which, when called, can normalize data into the
# [0.0, 1.0] interval.
norm = matplotlib.colors.Normalize(
    vmin=np.min(parameters),
    vmax=np.max(parameters))

# choose a colormap
c_m = matplotlib.cm.magma

# create a ScalarMappable and initialize a data structure
s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
s_m.set_array([])

i=0
k=0


from pathlib import Path

my_file = Path("./Filament_output.csv")
if my_file.is_file():
    import os

    os.remove("Filament_output.csv")

filament_out = open('Filament_output.csv','w')

filament_out.write('#plate,mjd,fiberID,RA,Dec,dist(Mpc)\n')


all_filament_gal=[]
all_filament_gal_dist=[]


nearest_points_RA=[]
nearest_points_Dec=[]

lengths = np.loadtxt('Distances60.dat')

from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from shapely.geometry import MultiPoint

for f in range(1,total_filaments+1):

            if(lengths[f-1]>=10.0):



                file = 'filament_' + str(f)+'.dat'
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


                RA_fil = filament_data[:,0]
                DEC_fil = filament_data[:, 1]


                list1 = [(x,y) for x,y in zip(RA_fil,DEC_fil)]

                filament = LineString(list1)         #geometry2


                #print len(list1)


                for j in range(len(nearby_gal)):

                    p = Point(data_no_cluster[nearby_gal[j],0],data_no_cluster[nearby_gal[j],1])

                    nearest_points_RA.append(p.x)
                    nearest_points_Dec.append(p.y)


                    #dist = p.distance(filament)

                    npoint = filament.interpolate(filament.project(p))


                    mp = MultiPoint(list1)
                    a =  nearest_points(mp, p)[0]  # POINT (19.124929 72.89317699999999)

                    


                    ra_fil_nearpoint = npoint.x *u.degree
                    dec_fil_nearpoint = npoint.y * u.degree
                    ra_gal = p.x*u.degree
                    dec_gal = p.y*u.degree

                    c1 = SkyCoord(ra=ra_gal, dec=dec_gal, distance=C)
                    c2 = SkyCoord(ra=ra_fil_nearpoint, dec=dec_fil_nearpoint, distance=C)
                    dist = c1.separation_3d(c2)

                    #print dist

                    d[j, 0] = dist.value


                    if (nearby_gal[j] in all_filament_gal):
                        ind = all_filament_gal.index(nearby_gal[j])
                        if(all_filament_gal_dist[ind]<dist.value):
                            continue
                        else:
                            all_filament_gal_dist[ind] = dist.value
                            all_filament_gal[ind] = nearby_gal[j]
                    else:
                        all_filament_gal.append(nearby_gal[j])
                        all_filament_gal_dist.append(dist.value)



                ind = int(lengths[f-1]) / 5
                # print ind
                temp = ax2.plot(filament_data[:, 0], filament_data[:, 1], linewidth=1.5, color=s_m.to_rgba(ind))
                gal = ax2.scatter(data_no_cluster[nearby_gal, 0], data_no_cluster[nearby_gal, 1], c=d[:, 0], s=1.5, cmap='jet')

                #filament_out.write('[FILAMENT:]'+str(f)+'\n')


                ax2.set_xlabel("Ra(deg)")
                ax2.set_ylabel("Dec(deg)")

                #ax2.plot(filament_data[:,0],filament_data[:,1],linewidth=2.5, color=s_m.to_rgba(ind))




#filament_out.close()

print len(all_filament_gal),len(np.unique(all_filament_gal))

np.savetxt(filament_out,np.array([save_data_filaments[all_filament_gal,0],\
                                              save_data_filaments[all_filament_gal,1],\
                                              save_data_filaments[all_filament_gal,2],\
                                              data_no_cluster[all_filament_gal, 0],\
                                              data_no_cluster[all_filament_gal, 1],all_filament_gal_dist]).T,\
                       delimiter=',',fmt='%d,%d,%d,%f,%f,%f')



save_data_field_plate=[]
save_data_field_mjd=[]
save_data_field_fibreID=[]
save_data_field_RA=[]
save_data_field_Dec=[]


for i in range(0,data_no_cluster.shape[0]):
    if i not in all_filament_gal:

        save_data_field_plate.append(save_data_filaments[i,0])
        save_data_field_mjd.append(save_data_filaments[i,1])
        save_data_field_fibreID.append(save_data_filaments[i, 2])
        save_data_field_RA.append(data_no_cluster[i,0])
        save_data_field_Dec.append(data_no_cluster[i, 1])



with open('Field_galaxies_clipped_10Mpc.csv','w') as outfile:

    outfile.write('#plate,mjd,fiberID,RA,Dec\n')

    for i in xrange(len(save_data_field_RA)):
        outfile.write('%d,%d,%d,%0.6f,%0.6f\n'%(save_data_field_plate[i],save_data_field_mjd[i],save_data_field_fibreID[i],save_data_field_RA[i],save_data_field_Dec[i]))



tot_dict = {'plate':data_coma[:,0].astype(int),'mjd':data_coma[:,1].astype(int),'fiberID':data_coma[:,2].astype(int),'RA':data_coma[:,3],'Dec':data_coma[:,4]}

tot_data_clipped = pd.DataFrame(tot_dict)

tot_data_clipped.to_csv('Total_galaxies.csv',index=False)

colorbar_ax = fig2.add_axes([0.92, 0.1, 0.01, 0.72])
#colorbar_ax1 = fig2.add_axes([0.15, 0.02, 0.8, 0.02])
cbar1 = plt.colorbar(gal,cax=colorbar_ax)
cbar1.set_label('distance(Mpc)')

'''
colorbar_ax1 = fig2.add_axes([0.15, 0.03, 0.72, 0.03])
cbar2 = plt.colorbar(s_m,cax=colorbar_ax1,orientation='horizontal')
cbar2.set_label('Length(Mpc)')
'''

fig2.savefig('FINAL_IMAGE'+'.png',dpi=600)





