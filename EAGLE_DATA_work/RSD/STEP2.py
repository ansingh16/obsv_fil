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
from scipy.spatial import distance
import eagleSqlTools as sql
from sympy import Point3D, Line3D
from sympy import N



plt.rcParams.update({'figure.max_open_warning': 0})



radius = 5.0
limit_log = 3.0


def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]

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



data_eagle = np.genfromtxt('Data_all_mass.csv',delimiter=',',skip_header=1)

#file_RADEC_array = np.array([data_coma[:,3],data_coma[:,4]]).T
#data_coma = clip_data(data_coma)
#print data_coma.shape



file_xyz_array = np.array([data_eagle[:,0],data_eagle[:,1],data_eagle[:,2]]).T
#save_data_array = np.array([data_eagle[:,0],data_eagle[:,1],data_eagle[:,2]]).T



R = []
Num_clus = []

fil_file = open('FILAMENTS'+'.dat', 'r')
lines = fil_file.readlines()
total_filaments = int(lines[0])

mySims = np.array([('RefL0100N1504', 100.)])

# This uses the eagleSqlTools module to connect to the database with your username and password.
# If the password is not given, the module will prompt for it.
con = sql.connect("asingh", password="GFB4ejc2")

for sim_name, sim_size in mySims:
    print sim_name

    # Construct and execute query for each simulation. This query returns the number of galaxies
    # for a given 30 pkpc aperture stellar mass bin (centered with 0.2 dex width).
    # MH.Group_M_Crit200 > 1.0e6 and \

    myQuery = "SELECT \
                        ape.Mass_Star as SM,\
                        M.Stars_Metallicity as Metal,\
                        (mg.u_nodust - mg.r_nodust) as u_minus_r,\
                        (mg.g_nodust - mg.r_nodust) as g_minus_r,\
                        M.CentreOfMass_x as x,\
                        M.CentreOfMass_y as y,\
                        M.CentreOfMass_z as z\
                FROM \
        			    %s_Subhalo as M, \
                        %s_Magnitudes as mg,\
                        %s_Aperture as ape,\
                        %s_FOF as G\
                WHERE \
        			    M.SnapNum = 27 \
                        and ape.Mass_Star >= 1.0e9\
                        and ape.ApertureSize = 30\
                        and M.GalaxyID = mg.GalaxyID\
                        and M.GalaxyID = ape.GalaxyID\
                        and M.GroupID = G.GroupID\
                        and G.GroupMass>=1.0e13\
                        and G.NumOfSubhalos>=4\
                ORDER BY \
        			    M.Mass " % (sim_name, sim_name, sim_name,sim_name)

    # Execute query.
    All_catalogue = sql.execute_query(con, myQuery)

    '''
    ax.scatter(myData_6[x], myData_6[y],myData_6[z], label=sim_name, s=1.0)
    ax.set_xlabel('x(cMpc)',fontweight='bold')
    ax.set_ylabel('y(cMpc)',fontweight='bold')
    ax.set_zlabel('z(cMpc)',fontweight='bold')
    '''

    Data = np.zeros((All_catalogue.shape[0], 7), dtype=np.float64)
    Data[:, 0] = All_catalogue['x']
    Data[:, 1] = All_catalogue['y']
    Data[:, 2] = All_catalogue['z']
    Data[:, 3] = All_catalogue['SM']
    Data[:, 4] = All_catalogue['Metal']
    Data[:, 5] = All_catalogue['u_minus_r']
    Data[:, 6] = All_catalogue['g_minus_r']

    np.savetxt("Data_Groups_and_cluster.csv", Data, header='x,y,z,SM,Metal,u_minus_r,g_minus_r', delimiter=',')



'''
#FINDING CLUSTERS AND GROUPS USING DBSCAN
import hdbscan

clusterer = hdbscan.HDBSCAN()

clusterer=hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5).fit(file_xyz_array)


#labels = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=10).fit_predict(file_RADEC_array)

labels=clusterer.labels_
prob = clusterer.probabilities_
'''

print "Total Galaxies:",data_eagle.shape
print 'Number of galaxies in groups:',Data.shape


df1 = pd.DataFrame(data_eagle)
df2 = pd.DataFrame(Data)


data_no_cluster = df1.loc[~df1.set_index(list(df1.columns)).index.isin(df2.set_index(list(df2.columns)).index)].values



print "Number of galaxies not in cluster",data_no_cluster.shape

#data_cluster = file_xyz_array[labels!=-1,:]

#print 'Total shape',save_data_array.shape
savedata_groups = Data

#print 'Groups shape',savedata_groups.shape
#print len(core_samples),file_RADEC_array.shape,savedata_groups.shape,save_data_array.shape
#np.savetxt('Groups_and_clusters.csv',savedata_groups)



#data_no_cluster = file_xyz_array[np.logical_not(labels!=-1),:]

#print 'Without clusters',data_no_cluster.shape


#READ THE LENGTHS OF THE FILAMENTS


lengths = np.loadtxt('Lengths.dat')
#print lengths.shape
#FOR THE FILAMENTS GREATER THAN THE 10MPC LENGTH CALCULATE THE GALAXIES WITHIN THE 3MPC DISTANCE FROM THE DATA
#HAVING NO CLUSTER


i=0
k=0


from pathlib import Path

my_file = Path("./Filament_output.csv")
if my_file.is_file():
    import os

    os.remove("Filament_output.csv")

filament_out = open('Filament_output.csv','w')

filament_out.write('#x,y,z,d_per(Mpc),d_long(Mpc),length_fil(Mpc),SM,Metal,u_minus_r,g_minus_r\n')


all_filament_gal=[]
all_filament_gal_dist=[]
all_filament_gal_dist_fil=[]
all_filament_gal_dist_tot_fil=[]


nearest_points_RA=[]
nearest_points_Dec=[]

lengths = np.loadtxt('Lengths.dat')

from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from shapely.geometry import MultiPoint

for f in range(1,total_filaments+1):

            if(lengths[f-1]>=5.0):



                file = 'filament_' + str(f)+'.dat'
                filament_data = np.loadtxt(file)

                print file

                #calculate the distances

                #Sky_all_points = SkyCoord(ra=data_no_cluster[:,0] * u.degree, dec=data_no_cluster[:,1] * u.degree, distance=C)

                Sky_all_points = SkyCoord(u = data_no_cluster[:,0], v = data_no_cluster[:,1], w = data_no_cluster[:,2], frame = 'galactic', unit = 'Mpc', representation = 'cartesian')

                Filament_all_points = SkyCoord(u = filament_data[:,0], v = filament_data[:,1], w = filament_data[:,2], frame = 'galactic', unit = 'Mpc', representation = 'cartesian')

                fil_value, sky_values, sep, dist = search_around_3d(Filament_all_points, Sky_all_points,distlimit=radius*u.Mpc)




                data = np.array([fil_value, sky_values, sep, dist]).T
                data = data[data[:, 3].argsort()]
                data_fil_ind = data[:, 0].astype(int)
                data_gal_ind = data[:, 1].astype(int)
                data_dist_gal = data[:, 3]

                nearby_gal, indexes = np.unique(data_gal_ind, return_index=True)
                near_fil_point = data_fil_ind[indexes]
                dist_fil = data_dist_gal[indexes]



                d_per = np.zeros((len(nearby_gal), 1))
                d_clus = np.zeros((len(nearby_gal), 1))
                d_totlen = np.zeros((len(nearby_gal), 1))




                x_fil = filament_data[:,0]
                y_fil = filament_data[:,1]
                z_fil = filament_data[:,2]

                #list1 = [(x,y,z) for x,y,z in zip(x_fil,y_fil,z_fil)]

                list1=[]

                for m in range(filament_data.shape[0]):
                    list1.append((x_fil[m],y_fil[m],z_fil[m]))


                print list1,filament_data


                for j in range(len(nearby_gal)):

                    p = Point(data_no_cluster[nearby_gal[j],0],data_no_cluster[nearby_gal[j],1],data_no_cluster[nearby_gal[j],2])

                    p1 = (data_no_cluster[nearby_gal[j], 0], data_no_cluster[nearby_gal[j], 1],data_no_cluster[nearby_gal[j], 2])

                    #dist = p.distance(filament)

                    #npoint = filament.interpolate(filament.project(p))

                    D = radius

                    for a in range(filament_data.shape[0] - 1):
                        P1, P2 = Point3D(x_fil[a], y_fil[a], z_fil[a]), Point3D(x_fil[a + 1], y_fil[a + 1],
                                                                                z_fil[a + 1])
                        l1 = Line3D(P1, P2)

                        P3 = Point3D(data_no_cluster[nearby_gal[j],0],data_no_cluster[nearby_gal[j],1],data_no_cluster[nearby_gal[j],2])

                        s1 = l1.perpendicular_segment(P3)



                        P = l1.intersect(s1)

                        P_list = list(P)

                        P_per = P_list[0]
                        C1 = float(N(P_per.x) - N(P3.x))
                        C2 = float(N(P_per.y) - N(P3.y))
                        C3 = float(N(P_per.z) - N(P3.z))

                        D1 = np.sqrt(C1**2 + C2**2 + C3**2)

                        #print D1

                        if(D1<D):
                            D=D1
                            x_fil_nearpoint=  float(N(P_per.x))
                            y_fil_nearpoint = float(N(P_per.y))
                            z_fil_nearpoint = float(N(P_per.z))

                    #print D

                    #mp = MultiPoint(list1)

                    #print mp ,p
                    nodes = np.array(list1)
                    #print "shape of nodes",nodes.shape

                    #perpen_point =  nearest_points(mp, p)[0] #POINT ON FILAMENT NEAR PERPENDICULAR POINT

                    perpen_point = closest_node(p1,nodes)
                    #print perpen_point
                    #print nearest_points(mp, p)

                    indp= list1.index((perpen_point[0],perpen_point[1],perpen_point[2]))

                    if(indp<(indp-len(list1))):

                        fildist=0.0

                        x_gal_np = x_fil_nearpoint
                        y_gal_np = y_fil_nearpoint
                        z_gal_np = z_fil_nearpoint


                        x_fil_np = perpen_point[0]
                        y_fil_np = perpen_point[0]
                        z_fil_np = perpen_point[0]


                        c1 = SkyCoord(u=x_gal_np, v=y_gal_np,w=z_gal_np, frame = 'galactic', unit = 'Mpc', representation = 'cartesian')
                        c2 = SkyCoord(u=x_fil_np, v=y_fil_np,w=z_fil_np, frame = 'galactic', unit = 'Mpc', representation = 'cartesian')
                        dist1 = c1.separation_3d(c2)
                        fildist = dist1


                        for m in range(0,indp):

                            x_along_fil1 = x_fil[m]
                            y_along_fil1 = y_fil[m]
                            z_along_fil1 = z_fil[m]

                            x_along_fil2 = x_fil[m+1]
                            y_along_fil2 = y_fil[m+1]
                            z_along_fil2 = z_fil[m + 1]


                            c1 = SkyCoord(u=x_along_fil1, v=y_along_fil1,w=z_along_fil1, frame = 'galactic', unit = 'Mpc', representation = 'cartesian')
                            c2 = SkyCoord(u=z_along_fil2, v=y_along_fil2, w=z_along_fil2, frame = 'galactic', unit = 'Mpc', representation = 'cartesian')
                            dist1 = c1.separation_3d(c2)

                            fildist+=dist1


                    elif(indp>(indp-len(list1))):
                        fildist = 0.0

                        x_gal_np = x_fil_nearpoint
                        y_gal_np = y_fil_nearpoint
                        z_gal_np = z_fil_nearpoint

                        x_fil_np = perpen_point[0]
                        y_fil_np = perpen_point[1]
                        z_fil_np = perpen_point[2]


                        c1 = SkyCoord(u=x_gal_np, v=y_gal_np, w=z_gal_np, frame = 'galactic', unit = 'Mpc', representation = 'cartesian')
                        c2 = SkyCoord(u=x_fil_np, v=y_fil_np, w=z_fil_np, frame = 'galactic', unit = 'Mpc', representation = 'cartesian')
                        dist1 = c1.separation_3d(c2)
                        fildist = dist1

                        for m in range(indp,len(list1)-1):

                            x_along_fil1 = x_fil[m]
                            y_along_fil1 = y_fil[m]
                            z_along_fil1 = z_fil[m]

                            x_along_fil2 = x_fil[m + 1]
                            y_along_fil2 = y_fil[m + 1]
                            z_along_fil2 = z_fil[m + 1]


                            c1 = SkyCoord(u=x_along_fil1, v=y_along_fil1,w=z_along_fil1, frame = 'galactic', unit = 'Mpc', representation = 'cartesian')
                            c2 = SkyCoord(u=z_along_fil2,v=y_along_fil2,w=z_along_fil2, frame = 'galactic', unit = 'Mpc', representation = 'cartesian')
                            dist1 = c1.separation_3d(c2)

                            fildist += dist1


                    elif (indp == len(list1)/2):
                        fildist = 0.0

                        x_gal_np = x_fil_nearpoint
                        y_gal_np = y_fil_nearpoint
                        z_gal_np = z_fil_nearpoint

                        x_fil_np = perpen_point[0]
                        y_fil_np = perpen_point[0]
                        z_fil_np = perpen_point[0]


                        c1 = SkyCoord(u=x_gal_np, v=y_gal_np,w=z_gal_np, frame = 'galactic', unit = 'Mpc', representation = 'cartesian')
                        c2 = SkyCoord(u=x_fil_np, v=y_fil_np,w=z_fil_np, frame = 'galactic', unit = 'Mpc', representation = 'cartesian')
                        dist1 = c1.separation_3d(c2)
                        fildist = dist1

                        for m in range(indp, len(list1) - 1):
                            x_along_fil1 = x_fil[m]
                            y_along_fil1 = y_fil[m]
                            z_along_fil1 = z_fil[m]

                            x_along_fil2 = x_fil[m + 1]
                            y_along_fil2 = y_fil[m + 1]
                            z_along_fil2 = z_fil[m + 1]


                            c1 = SkyCoord(u=x_along_fil1, v=y_along_fil1,w=z_along_fil1, frame = 'galactic', unit = 'Mpc', representation = 'cartesian')
                            c2 = SkyCoord(u=x_along_fil2, v=y_along_fil2,w=z_along_fil2, frame = 'galactic', unit = 'Mpc', representation = 'cartesian')
                            dist1 = c1.separation_3d(c2)

                            fildist += dist1

                    

                    d_clus[j,0] = fildist.value

                    d_totlen[j,0] = lengths[f-1]

                    #x_fil_nearpoint = npoint.x
                    #y_fil_nearpoint = npoint.y
                    #z_fil_nearpoint = npoint.z


                    x_gal = p.x
                    y_gal = p.y
                    z_gal = p.z

                    c1 = SkyCoord(u=x_gal, v=y_gal,w=z_gal, frame = 'galactic', unit = 'Mpc', representation = 'cartesian')
                    c2 = SkyCoord(u=x_fil_nearpoint, v=y_fil_nearpoint , w=z_fil_nearpoint, frame = 'galactic', unit = 'Mpc', representation = 'cartesian')
                    dist = c1.separation_3d(c2)

                    print dist

                    d_per[j, 0] = dist.value


                    if (nearby_gal[j] in all_filament_gal):
                        ind = all_filament_gal.index(nearby_gal[j])
                        if(all_filament_gal_dist[ind]<dist.value):
                            continue
                        else:
                            all_filament_gal_dist[ind] = dist.value
                            all_filament_gal_dist_fil[ind] = fildist.value
                            all_filament_gal_dist_tot_fil[ind] = lengths[f-1]

                            all_filament_gal[ind] = nearby_gal[j]
                    else:
                        all_filament_gal.append(nearby_gal[j])
                        all_filament_gal_dist.append(dist.value)

                        all_filament_gal_dist_fil.append(fildist.value)
                        all_filament_gal_dist_tot_fil.append(lengths[f-1])





#filament_out.close()

print len(all_filament_gal),len(np.unique(all_filament_gal))

np.savetxt(filament_out,np.array([            data_no_cluster[all_filament_gal, 0],\
                                              data_no_cluster[all_filament_gal, 1],\
                                              data_no_cluster[all_filament_gal, 2],\
                                              all_filament_gal_dist,\
                                              all_filament_gal_dist_fil,\
                                              all_filament_gal_dist_tot_fil,\
                                              data_no_cluster[all_filament_gal, 3],\
                                              data_no_cluster[all_filament_gal, 4],\
                                              data_no_cluster[all_filament_gal, 5],
                                              data_no_cluster[all_filament_gal, 6]]).T,\
                                              delimiter=',',fmt='%f,%f,%f,%f,%f,%f,%f,%f,%f,%f')



save_data_field_x=[]
save_data_field_y=[]
save_data_field_z=[]

save_data_field_Mass=[]
save_data_field_SM=[]
save_data_field_GALEX_FUV=[]
save_data_field_GALEX_NUV=[]
save_data_field_SDSS_r=[]
save_data_field_g_minus_r=[]
save_data_field_u_minus_r=[]


save_data_field_Metal=[]




for i in range(0,data_no_cluster.shape[0]):
    if i not in all_filament_gal:

        save_data_field_x.append(data_no_cluster[i,0])
        save_data_field_y.append(data_no_cluster[i, 1])
        save_data_field_z.append(data_no_cluster[i, 2])
        save_data_field_SM.append(data_no_cluster[i, 3])
        save_data_field_Metal.append(data_no_cluster[i, 4])

        save_data_field_u_minus_r.append(data_no_cluster[i, 5])
        save_data_field_g_minus_r.append(data_no_cluster[i, 6])



with open('Field_galaxies.csv','w') as outfile:

    outfile.write('#x,y,z,SM,Metal,u_minus_r,g_minus_r\n')

    for i in xrange(len(save_data_field_x)):
        outfile.write('%f,%f,%f,%f,%f,%f,%f\n'%(save_data_field_x[i],\
                                                   save_data_field_y[i],\
                                                   save_data_field_z[i],\
                                                   save_data_field_SM[i], \
                                                   save_data_field_Metal[i],\
                                                   save_data_field_u_minus_r[i],\
                                                   save_data_field_g_minus_r[i]))



tot_dict = {'x':data_eagle[:,0],'y':data_eagle[:,1],'z':data_eagle[:,2]}

tot_data_clipped = pd.DataFrame(tot_dict)

tot_data_clipped.to_csv('Total_galaxies.csv',index=False)

#colorbar_ax = fig2.add_axes([0.92, 0.1, 0.01, 0.72])
#colorbar_ax1 = fig2.add_axes([0.15, 0.02, 0.8, 0.02])
#cbar1 = plt.colorbar(gal,cax=colorbar_ax)
#cbar1.set_label('distance(Mpc)')

'''
colorbar_ax1 = fig2.add_axes([0.15, 0.03, 0.72, 0.03])
cbar2 = plt.colorbar(s_m,cax=colorbar_ax1,orientation='horizontal')
cbar2.set_label('Length(Mpc)')
'''

#fig2.savefig('FINAL_IMAGE'+'.png',dpi=600)



#plt.show()




