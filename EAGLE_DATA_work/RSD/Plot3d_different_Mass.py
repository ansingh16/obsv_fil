import eagleSqlTools as sql
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from subprocess import call
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.axes_grid import AxesGrid
from matplotlib.offsetbox import AnchoredText
#from adjustText import adjust_text
import csv
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.coordinates import search_around_3d
import astropy.units as u
from matplotlib.widgets import SpanSelector
from checking_random_slicing import get_slice

import pandas as pd
#from halotools.mock_observables import return_xyz_formatted_array

from cross_match import cross_matching

def onpick3(event):
    ind = event.ind
    print('onpick3 scatter:', ind, np.take(x, ind), np.take(y, ind))




#ax = fig.add_subplot(111, projection='2d')

# Array of chosen simulations. Entries refer to the simulation name and comoving box length.
mySims		= np.array([('RefL0100N1504', 100.)])

# This uses the eagleSqlTools module to connect to the database with your username and password.
# If the password is not given, the module will prompt for it.
con     	= sql.connect("asingh", password="GFB4ejc2")



for sim_name, sim_size in mySims:

        print sim_name
    
	    # Construct and execute query for each simulation. This query returns the number of galaxies
	    # for a given 30 pkpc aperture stellar mass bin (centered with 0.2 dex width).
        #MH.Group_M_Crit200 > 1.0e6 and \

        myQuery = "SELECT \
                        ape.Mass_Star as SM,\
                        M.Stars_Metallicity as Metal,\
                        (mg.u_nodust - mg.r_nodust) as u_minus_r,\
                        (mg.g_nodust - mg.r_nodust) as g_minus_r,\
                        M.CentreOfMass_x as x,\
                        M.CentreOfMass_y as y,\
                        M.CentreOfMass_z as z,\
                        M.Velocity_x as Velx,\
                        M.Velocity_y as Vely,\
                        M.Velocity_z as Velz\
                FROM \
        			    %s_Subhalo as M, \
                        %s_Magnitudes as mg,\
                        %s_Aperture as ape\
                WHERE \
        			    M.SnapNum = 27 \
                        and M.Spurious=0\
                        and ape.Mass_Star >= 1.0e9\
                        and ape.ApertureSize = 30\
                        and M.GalaxyID = mg.GalaxyID\
                        and M.GalaxyID = ape.GalaxyID\
        ORDER BY \
        			    M.Mass " % (sim_name, sim_name, sim_name)

        # Execute query.
        All_catalogue 	= sql.execute_query(con, myQuery)

        Data = np.zeros((All_catalogue.shape[0],7),dtype=np.float64)
        Data[:,0] = All_catalogue['x']
        Data[:,1] = All_catalogue['y']
        Data[:,2] = All_catalogue['z']
        #Data[:,3] = All_catalogue['mass']
        Data[:,3] = All_catalogue['SM']
        Data[:,4] = All_catalogue['Metal']
        #Data[:,6] = All_catalogue['g_minus_r']
        Data[:,5] = All_catalogue['u_minus_r']
        Data[:, 6] = All_catalogue['g_minus_r']
        #Data = Data[Data[:,3]>1.0e8]

        Data1 = np.zeros((All_catalogue.shape[0],6),dtype=np.float64)
        Data1[:, 0:3] = Data[:,0:3]
        Data1[:,3] = All_catalogue['Velx']
        Data1[:, 4] = All_catalogue['Vely']
        Data1[:, 5] = All_catalogue['Velz']

        #Data1 = Data[(float(sim_size)/2-50.0<Data[:,0]) & (Data[:,0]<float(sim_size)/2+50.0)]
        #Data1 = Data1[(float(sim_size) / 2 - 50.0 < Data1[:, 1]) & (Data1[:,0] < float(sim_size) / 2 + 50.0)]
        #Data1 = Data1[(float(sim_size) / 2 - 50.0 < Data1[:, 2]) & (Data1[:,0] < float(sim_size) / 2 + 50.0)]

        np.savetxt("Data_all_mass.csv", Data,header='x,y,z,SM,Metal,u_minus_r,g_minus_r',delimiter=',')
        np.savetxt('Data_xyz_vxyz.csv', Data1,header='px py pz vx vy vz',delimiter=',')
        #np.savetxt('Data_slice.csv', Data1[:, 0:2],header='px py',delimiter=',')


        print sim_name

        # Construct and execute query for each simulation. This query returns the number of galaxies
        # for a given 30 pkpc aperture stellar mass bin (centered with 0.2 dex width).
        # MH.Group_M_Crit200 > 1.0e6 and \

        myQuery = "SELECT \
                                G.GroupCentreOfPotential_x as x,\
                                G.GroupCentreOfPotential_y as y,\
                                G.GroupCentreOfPotential_z as z,\
                                G.Group_M_Crit200 as M,\
                                G.Group_R_Crit200 as R\
                        FROM \
                                %s_FOF as G\
                        WHERE \
                			    G.SnapNum = 27 \
                                and G.GroupMass>=1.0e14\
                                and G.NumOfSubhalos>=4\
                        ORDER BY \
                			    G.Group_M_Crit200 " % (sim_name)

        # Execute query.
        All_catalogue = sql.execute_query(con, myQuery)

        
        Data = np.zeros((All_catalogue.shape[0], 5), dtype=np.float64)
        Data[:, 0] = All_catalogue['x']
        Data[:, 1] = All_catalogue['y']
        Data[:, 2] = All_catalogue['z']
        Data[:, 3] = All_catalogue['M']
        Data[:, 4] = All_catalogue['R']

        np.savetxt("Data_Cluster.csv", Data, header='x,y,z,M,R', delimiter=',')

#bin_list = np.arange(0.5,max(Data6[:,2]),4.0)
Data1 = np.loadtxt('Data_xyz_vxyz.csv',delimiter=',')



data = np.loadtxt('Data_Cluster.csv', delimiter=',', skiprows=1)
Data = np.loadtxt('Data_all_mass.csv',delimiter=',',skiprows=1)

#Data = np.loadtxt('Test_data.csv',delimiter=',',skiprows=1)


from itertools import combinations

comb = combinations(range(10), 2)
distlist = []
between1 = []
between2 = []
DMAX = 50.0
for i in comb:
        Cluster1 = SkyCoord(u=data[i[0], 0], v=data[i[0], 1], w=data[i[0], 2], frame='galactic', unit='Mpc',
                            representation='cartesian')
        Cluster2 = SkyCoord(u=data[i[1], 0], v=data[i[1], 1], w=data[i[1], 2], frame='galactic', unit='Mpc',
                            representation='cartesian')

        dist = Cluster1.separation_3d(Cluster2)

        if (dist.value < DMAX):
                distlist.append(dist.value)
                between1.append(i[0])
                between2.append(i[1])

print len(distlist), between1, between2



call(['mkdir','SLICED_DATA'])



k=0

import h5py

DB = h5py.File('Total_data.hdf5','w')




for i,j in zip(between1,between2):

        dset = DB.create_group('Pair'+str(k))

        cluster1 = np.array([data[i,0],data[i,1],data[i,2]])
        cluster2 = np.array([data[j,0],data[j,1],data[j,2]])

        dset.attrs['positions1'] = cluster1
        dset.attrs['positions2'] = cluster2

        origin_shift = (cluster1+cluster2)/2.0

        rad = 12.0#10* max(data[i,4],data[j,4])/1000.0

        dset.attrs['width'] = rad

        origin_shiftx, origin_shifty, origin_shiftz = origin_shift[0], origin_shift[1], origin_shift[2]
        #origin_shift = np.array([origin_shiftx,origin_shifty,origin_shiftz])


        cluster_vec = cluster2 - cluster1

        norx = np.random.rand()
        nory = np.random.rand()
        norz = -(cluster_vec[0]*norx + cluster_vec[1]*nory)/cluster_vec[2]

        normal = np.array([norx,nory,norz])

        dset.attrs['normal'] = normal




        DG,DF,DFi,total_Data,zl,zu = get_slice(origin_shift,normal,Data[:,0:3].T,rad,k)

        print "Main shape: ",total_Data.shape,DG.shape,DF.shape,DFi.shape

        Data_Groups = cross_matching(Data,DG,k)

        Data_Filaments = cross_matching(Data, DF, k)
        Data_Filaments['d_per'] = DF[:,3]
        Data_Filaments['d_long'] = DF[:,4]
        Data_Filaments['d_total'] = DF[:,5]

        Data_Fields = cross_matching(Data, DFi, k)



        #SLICED_DATA.to_csv('./SLICED_DATA/SLICE'+str(k)+'.csv',header=SLICED_DATA.columns)



        df_to_nparray = Data_Groups.as_matrix()
        #dset['Groups'] = df_to_nparray


        grpG = dset.create_dataset("Group", data = df_to_nparray , dtype=np.float64)
        #grpG = df_to_nparray

        df_to_nparray = Data_Filaments.as_matrix()
        grpF = dset.create_dataset("Filament", data = df_to_nparray, dtype=np.float64)


        df_to_nparray = Data_Fields.as_matrix()
        #dset['Fields'] = df_to_nparray

        grpFi = dset.create_dataset("Field", data = df_to_nparray, dtype=np.float64)
        #grpFi = df_to_nparray


        xx, yy = np.meshgrid([float(i) for i in range(100)], [float(i) for i in range(100)])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Data[:,0],Data[:,1],Data[:,2],s=2.0)
        ax.scatter(origin_shiftx,origin_shifty,origin_shiftz,s=18,color='red')
        #ax.plot_surface(xx, yy, zl, alpha=0.2)
        #ax.plot_surface(xx, yy, zu, alpha=0.2)
        ax.set_xlim(0.0,100.0)
        ax.set_ylim(0.0,100.0)
        ax.set_zlim(0.0,100.0)
        ax.set_title('Full_box'+str(k))
        fig.savefig('Full_box'+str(k)+'.png')

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.scatter(total_Data[:,0],total_Data[:,1],total_Data[:,2],s=2.0)
        #ax2.plot_surface(xx, yy, zl, alpha=0.2)
        #ax2.plot_surface(xx, yy, zu, alpha=0.2)
        ax2.set_xlim(0.0,100.0)
        ax2.set_ylim(0.0,100.0)
        ax2.set_zlim(0.0,100.0)
        ax2.set_title('Slice : '+str(k))
        fig2.savefig('Sliced.'+str(k)+'.png')


        #plt.show()
        #break

        print "Main loop ",k

        call(['rm', 'filament_*'])

        k=k+1



