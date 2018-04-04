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

from matplotlib.widgets import SpanSelector


def onpick3(event):
    ind = event.ind
    print('onpick3 scatter:', ind, np.take(x, ind), np.take(y, ind))




fig,ax = plt.subplots()
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
                        M.CentreOfMass_x as x,\
                        M.CentreOfMass_y as y,\
                        M.CentreOfMass_z as z\
                FROM \
        			    %s_Subhalo as M, \
                        %s_Magnitudes as mg,\
                        %s_Aperture as ape\
                WHERE \
        			    M.SnapNum = 27 \
                        and ape.Mass_Star >= 1.0e9\
                        and ape.ApertureSize = 30\
                        and M.GalaxyID = mg.GalaxyID\
                        and M.GalaxyID = ape.GalaxyID\
        ORDER BY \
        			    M.Mass " % (sim_name, sim_name, sim_name)

        # Execute query.
        All_catalogue 	= sql.execute_query(con, myQuery)

        '''
                                and M.Spurious=0\

        and G.NumOfSubhalos>0\
                        and G.GroupMass>0.0\
                        and M.GroupID = G.GroupID\
                        
                        
        ax.scatter(myData_6[x], myData_6[y],myData_6[z], label=sim_name, s=1.0)
        ax.set_xlabel('x(cMpc)',fontweight='bold')
        ax.set_ylabel('y(cMpc)',fontweight='bold')
        ax.set_zlabel('z(cMpc)',fontweight='bold')
        '''



        Data = np.zeros((All_catalogue.shape[0],6),dtype=np.float64)
        Data[:,0] = All_catalogue['x']
        Data[:,1] = All_catalogue['y']
        Data[:,2] = All_catalogue['z']
        #Data[:,3] = All_catalogue['mass']
        Data[:,3] = All_catalogue['SM']
        Data[:,4] = All_catalogue['Metal']
        #Data[:,6] = All_catalogue['g_minus_r']
        Data[:,5] = All_catalogue['u_minus_r']

        #Data = Data[Data[:,3]>1.0e8]

        #Data1 = Data[(float(sim_size)/2-50.0<Data[:,0]) & (Data[:,0]<float(sim_size)/2+50.0)]
        #Data1 = Data1[(float(sim_size) / 2 - 50.0 < Data1[:, 1]) & (Data1[:,0] < float(sim_size) / 2 + 50.0)]
        #Data1 = Data1[(float(sim_size) / 2 - 50.0 < Data1[:, 2]) & (Data1[:,0] < float(sim_size) / 2 + 50.0)]

        np.savetxt("Data_all_mass.csv", Data,header='x,y,z,SM,Metal,u_minus_r',delimiter=',')
        np.savetxt('Data_xyz.csv', Data[:,0:3],header='px py pz',delimiter=',')
        #np.savetxt('Data_slice.csv', Data1[:, 0:2],header='px py',delimiter=',')

#bin_list = np.arange(0.5,max(Data6[:,2]),4.0)


im = ax.scatter(np.log10(Data[:,3]),Data[:,5],c = np.log10(Data[:,4]/0.012),s=2)

cbar = plt.colorbar(im)
cbar.set_label(r'Metallicity(Z/$Z_{\odot}$)')


ax.set_xlabel(r'Stellar Mass($M_{\odot}$)')
ax.set_ylabel(r'u - r')
fig.savefig('SM_vs_uminusr.png')
plt.show()


'''

FILAMENT_FILE = SKLfile+'.S010.a.NDskl'


with open(FILAMENT_FILE) as infile, open('FILAMENTS'+str(i)+'.dat','w') as outfile:
            copy = False
            for line in infile:
                if line.strip() == "[FILAMENTS]":
                    copy = True
                elif line.strip() == "[CRITICAL POINTS DATA]":
                    copy = False
                elif copy:
                    outfile.write(line)

outfile.close()

fil_file = open('FILAMENTS'+'.dat', 'r')

lines = fil_file.readlines()

        #print len(lines)

        total_filaments = lines[0]

        count = 1
        k = 0
        for j in range(len(lines)):
            if lines[j].startswith(' '):
                k = k + 1
            else:
                count = count + 1

        #print count

        file_list = []
        for j in range(count - 1):
            file_list.append('filament'+str(i)+'_'+ str(j)+'.dat')
            #file_list[j] = open('filament' + str(i)+'_'+str(j)+'.dat', 'r')

        for l in range(len(file_list)):
            with open(file_list[l], 'w') as file:
                for j in range(1, len(lines)):

                    if lines[j].startswith(' '):
                        # print l,i,len(file_list),len(lines)
                            #print l,count-1
                            file.write(lines[j])
                    else:
                        l = l + 1


        fig,ax = plt.subplots(figsize=(8, 8))


        for fname in file_list[1:]:
            with open(fname) as infile:
                data = np.loadtxt(infile)
                # FT.write('%f,%f')
                ax.scatter(data[:, 0], data[:, 1],color='black', s=1)
                ax.set_ylabel(r"X (Mpc)", fontweight='bold', fontsize=16)
                ax.set_xlabel(r"Y (Mpc)", fontweight='bold', fontsize=16)
                ax.minorticks_on()
                ax.tick_params(axis='both', which='minor', length=5, width=2, labelsize=15)
                ax.tick_params(axis='both', which='major', length=8, width=2, labelsize=15)

        with open('FILAMENT_TOTAL'+str(i)+'.csv', 'w') as outfile:
            o = csv.DictWriter(outfile, fieldnames=["fx", "fy"])
            o.writeheader()
            o = csv.writer(outfile)

            for fname in file_list[1:]:
                with open(str(fname)) as infile:
                    for line in infile:
                        o.writerow(line.split())


        data  = np.genfromtxt('FILAMENT_TOTAL'+str(i)+'.csv',delimiter=',')

        np.savetxt('filament_unique'+str(i)+'.csv',np.unique(data,axis=0),delimiter=',',header='fx,fy')
        filfile.writelines('filament_unique'+str(i)+'.csv' +'\n')

        fig.savefig('SLICE_PLOT' + str(i)+'.png',dpi=600)

filfile.close()
'''