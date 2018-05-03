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


def Environment_classifier(Data,K):


    file_RADEC_array = np.array([Data[:,0],Data[:,1],Data[:,2]]).T


    fil_file = open('FILAMENTS'+'.dat', 'r')
    lines = fil_file.readlines()
    total_filaments = int(lines[0])

    #FINDING CLUSTERS AND GROUPS USING DBSCAN
    import hdbscan

    clusterer = hdbscan.HDBSCAN()

    clusterer=hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10).fit(file_RADEC_array)

    labels=clusterer.labels_

    data_no_cluster = file_RADEC_array[np.logical_not(labels!=-1),:]



    print 'Without clusters',data_no_cluster.shape


    #READ THE LENGTHS OF THE FILAMENTS


    lengths = np.loadtxt('Lengths.dat')
    print lengths.shape
    #FOR THE FILAMENTS GREATER THAN THE 10MPC LENGTH CALCULATE THE GALAXIES WITHIN THE 3MPC DISTANCE FROM THE DATA
    #HAVING NO CLUSTER


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



    from pathlib import Path

    my_file = Path("./Filament_output.csv")
    if my_file.is_file():
        import os

        os.remove("Filament_output.csv")

    filament_out = open('Filament_output.csv','w')

    filament_out.write('#plate,mjd,fiberID,RA,Dec,d_per(Mpc),d_long(Mpc),length_fil(Mpc)\n')


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
    from sympy import Point, Line
    from sympy import *

    for f in range(1,total_filaments + 1):

        if (lengths[f - 1] >= 0.0):

            file = 'filament_' + str(f) + '.dat'
            filament_data = np.loadtxt(file)

            #print file

            # calculate the distances

            # Sky_all_points = SkyCoord(ra=data_no_cluster[:,0] * u.degree, dec=data_no_cluster[:,1] * u.degree, distance=C)

            Sky_all_points = SkyCoord(u=data_no_cluster[:, 0], v=data_no_cluster[:, 1], w=0.0,
                                      frame='galactic', unit='Mpc', representation='cartesian')

            Filament_all_points = SkyCoord(u=filament_data[:, 0], v=filament_data[:, 1], w=0.0,
                                           frame='galactic', unit='Mpc', representation='cartesian')

            fil_value, sky_values, sep, dist = search_around_3d(Filament_all_points, Sky_all_points,
                                                                distlimit=radius * u.Mpc)

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

            x_fil = filament_data[:, 0]
            y_fil = filament_data[:, 1]

            # list1 = [(x,y,z) for x,y,z in zip(x_fil,y_fil,z_fil)]

            list1 = []

            for m in range(filament_data.shape[0]):
                list1.append((x_fil[m], y_fil[m]))

            #print list1, filament_data

            for j in range(len(nearby_gal)):

                p = Point(data_no_cluster[nearby_gal[j], 0], data_no_cluster[nearby_gal[j], 1])

                p1 = (data_no_cluster[nearby_gal[j], 0], data_no_cluster[nearby_gal[j], 1])

                # dist = p.distance(filament)

                # npoint = filament.interpolate(filament.project(p))

                D = radius

                for a in range(filament_data.shape[0] - 1):
                    P1, P2 = Point(x_fil[a], y_fil[a]), Point(x_fil[a + 1], y_fil[a + 1])
                    l1 = Line(P1, P2)

                    P3 = Point(data_no_cluster[nearby_gal[j], 0], data_no_cluster[nearby_gal[j], 1])

                    s1 = l1.perpendicular_segment(P3)

                    P = l1.intersect(s1)

                    P_list = list(P)

                    P_per = P_list[0]
                    C1 = float(N(P_per.x) - N(P3.x))
                    C2 = float(N(P_per.y) - N(P3.y))

                    D1 = np.sqrt(C1 ** 2 + C2 ** 2 )

                    # print D1

                    if (D1 < D):
                        D = D1
                        x_fil_nearpoint = float(N(P_per.x))
                        y_fil_nearpoint = float(N(P_per.y))

                # print D

                # mp = MultiPoint(list1)

                # print mp ,p
                nodes = np.array(list1)
                # print "shape of nodes",nodes.shape

                # perpen_point =  nearest_points(mp, p)[0] #POINT ON FILAMENT NEAR PERPENDICULAR POINT

                perpen_point = closest_node(p1, nodes)
                # print perpen_point
                # print nearest_points(mp, p)

                indp = list1.index((perpen_point[0], perpen_point[1]))

                if (indp < (indp - len(list1))):

                    fildist = 0.0

                    x_gal_np = x_fil_nearpoint
                    y_gal_np = y_fil_nearpoint

                    x_fil_np = perpen_point[0]
                    y_fil_np = perpen_point[1]

                    c1 = SkyCoord(u=x_gal_np, v=y_gal_np, w=0.0, frame='galactic', unit='Mpc',
                                  representation='cartesian')
                    c2 = SkyCoord(u=x_fil_np, v=y_fil_np, w=0.0, frame='galactic', unit='Mpc',
                                  representation='cartesian')
                    dist1 = c1.separation_3d(c2)
                    fildist = dist1

                    for m in range(0, indp):
                        x_along_fil1 = x_fil[m]
                        y_along_fil1 = y_fil[m]

                        x_along_fil2 = x_fil[m + 1]
                        y_along_fil2 = y_fil[m + 1]

                        c1 = SkyCoord(u=x_along_fil1, v=y_along_fil1, w=0.0, frame='galactic', unit='Mpc',
                                      representation='cartesian')
                        c2 = SkyCoord(u=z_along_fil2, v=y_along_fil2, w=0.0, frame='galactic', unit='Mpc',
                                      representation='cartesian')
                        dist1 = c1.separation_3d(c2)

                        fildist += dist1


                elif (indp > (indp - len(list1))):
                    fildist = 0.0

                    x_gal_np = x_fil_nearpoint
                    y_gal_np = y_fil_nearpoint

                    x_fil_np = perpen_point[0]
                    y_fil_np = perpen_point[1]

                    c1 = SkyCoord(u=x_gal_np, v=y_gal_np, w=0.0, frame='galactic', unit='Mpc',
                                  representation='cartesian')
                    c2 = SkyCoord(u=x_fil_np, v=y_fil_np, w=0.0, frame='galactic', unit='Mpc',
                                  representation='cartesian')
                    dist1 = c1.separation_3d(c2)
                    fildist = dist1

                    for m in range(indp, len(list1) - 1):
                        x_along_fil1 = x_fil[m]
                        y_along_fil1 = y_fil[m]

                        x_along_fil2 = x_fil[m + 1]
                        y_along_fil2 = y_fil[m + 1]

                        c1 = SkyCoord(u=x_along_fil1, v=y_along_fil1, w=0.0, frame='galactic', unit='Mpc',
                                      representation='cartesian')
                        c2 = SkyCoord(u=x_along_fil2, v=y_along_fil2, w=0.0, frame='galactic', unit='Mpc',
                                      representation='cartesian')
                        dist1 = c1.separation_3d(c2)

                        fildist += dist1


                elif (indp == len(list1) / 2):
                    fildist = 0.0

                    x_gal_np = x_fil_nearpoint
                    y_gal_np = y_fil_nearpoint

                    x_fil_np = perpen_point[0]
                    y_fil_np = perpen_point[1]

                    c1 = SkyCoord(u=x_gal_np, v=y_gal_np, w=0.0, frame='galactic', unit='Mpc',
                                  representation='cartesian')
                    c2 = SkyCoord(u=x_fil_np, v=y_fil_np, w=0.0, frame='galactic', unit='Mpc',
                                  representation='cartesian')
                    dist1 = c1.separation_3d(c2)
                    fildist = dist1

                    for m in range(indp, len(list1) - 1):
                        x_along_fil1 = x_fil[m]
                        y_along_fil1 = y_fil[m]

                        x_along_fil2 = x_fil[m + 1]
                        y_along_fil2 = y_fil[m + 1]

                        c1 = SkyCoord(u=x_along_fil1, v=y_along_fil1, w=0.0, frame='galactic', unit='Mpc',
                                      representation='cartesian')
                        c2 = SkyCoord(u=x_along_fil2, v=y_along_fil2, w=0.0, frame='galactic', unit='Mpc',
                                      representation='cartesian')
                        dist1 = c1.separation_3d(c2)

                        fildist += dist1

                d_clus[j, 0] = fildist.value

                d_totlen[j, 0] = lengths[f - 1]

                # x_fil_nearpoint = npoint.x
                # y_fil_nearpoint = npoint.y
                # z_fil_nearpoint = npoint.z


                x_gal = float(N(p.x))
                y_gal = float(N(p.y))


                c1 = SkyCoord(u=x_gal, v=y_gal, w=0.0, frame='galactic', unit='Mpc', representation='cartesian')
                c2 = SkyCoord(u=x_fil_nearpoint, v=y_fil_nearpoint, w=0.0, frame='galactic', unit='Mpc',
                              representation='cartesian')
                dist = c1.separation_3d(c2)

                #print dist

                d_per[j, 0] = dist.value

                if (nearby_gal[j] in all_filament_gal):
                    ind = all_filament_gal.index(nearby_gal[j])
                    if (all_filament_gal_dist[ind] < dist.value):
                        continue
                    else:
                        all_filament_gal_dist[ind] = dist.value
                        all_filament_gal_dist_fil[ind] = fildist.value
                        all_filament_gal_dist_tot_fil[ind] = lengths[f - 1]

                        all_filament_gal[ind] = nearby_gal[j]
                else:
                    all_filament_gal.append(nearby_gal[j])
                    all_filament_gal_dist.append(dist.value)

                    all_filament_gal_dist_fil.append(fildist.value)
                    all_filament_gal_dist_tot_fil.append(lengths[f - 1])

            ind = int(lengths[f - 1]) / 5

            temp = ax2.plot(filament_data[:, 0], filament_data[:, 1], linewidth=1.5, color=s_m.to_rgba(ind))
            gal = ax2.scatter(data_no_cluster[nearby_gal, 0], data_no_cluster[nearby_gal, 1], color='black',
                                  alpha=0.7, s=5.0)



            ax2.set_xlabel("Ra(deg)")
            ax2.set_ylabel("Dec(deg)")

    #filament_out.close()

    #print len(all_filament_gal),len(np.unique(all_filament_gal))

    np.savetxt(filament_out,np.array([data_no_cluster[all_filament_gal, 0],
                                      data_no_cluster[all_filament_gal, 1],
                                      all_filament_gal_dist,
                                      all_filament_gal_dist_fil,
                                      all_filament_gal_dist_tot_fil]).T,
                           delimiter=',',fmt='%f,%f,%f,%f,%f')



    save_data_field_RA=[]
    save_data_field_Dec=[]
    save_data_field_z=[]

    for i in range(0,data_no_cluster.shape[0]):
        if i not in all_filament_gal:

            save_data_field_RA.append(data_no_cluster[i,0])
            save_data_field_Dec.append(data_no_cluster[i, 1])
            save_data_field_z.append(data_no_cluster[i, 2])



    with open('Field_galaxies_clipped_10Mpc.csv','w') as outfile:

        outfile.write('#plate,mjd,fiberID,RA,Dec\n')

        for i in xrange(len(save_data_field_RA)):
            outfile.write('%0.6f,%0.6f, %0.6f\n'%(save_data_field_RA[i],save_data_field_Dec[i],save_data_field_z[i]))






    data_groups = Data[labels!=-1,:]



    data_field = np.array([save_data_field_RA, save_data_field_Dec,save_data_field_z]).T

    data_filament = np.array([data_no_cluster[all_filament_gal, 0],
                                      data_no_cluster[all_filament_gal, 1],
                                      data_no_cluster[all_filament_gal, 2],
                                      all_filament_gal_dist,
                                      all_filament_gal_dist_fil,
                                      all_filament_gal_dist_tot_fil]).T


    ax2.scatter(data_groups[:,0],data_groups[:,1],color='red',s=5.0)
    ax2.scatter(data_field[:,0],data_field[:,1],color='blue',s=5.0)


    fig2.savefig('FINAL_IMAGE'+str(K)+'.png',dpi=600)

    return data_groups,data_filament,data_field,Data





#plt.show()




