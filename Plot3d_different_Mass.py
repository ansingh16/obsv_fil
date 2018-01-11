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




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Array of chosen simulations. Entries refer to the simulation name and comoving box length.
mySims		= np.array([('RefL0050N0752', 50.)])

# This uses the eagleSqlTools module to connect to the database with your username and password.
# If the password is not given, the module will prompt for it.
con     	= sql.connect("asingh", password="GFB4ejc2")



for sim_name, sim_size in mySims:

        print sim_name
    
	    # Construct and execute query for each simulation. This query returns the number of galaxies
	    # for a given 30 pkpc aperture stellar mass bin (centered with 0.2 dex width).
        #MH.Group_M_Crit200 > 1.0e6 and \

        myQuery = "SELECT \
                MH.GroupCentreOfPotential_x as x,\
        MH.GroupCentreOfPotential_y as y,\
        MH.GroupCentreOfPotential_z as z,\
        MH.Group_M_Crit200 as mass\
			FROM \
			    %s_FOF as MH \
            WHERE \
			    MH.SnapNum = 27 \
		    ORDER BY \
			    MH.Group_M_Crit200 "%(sim_name)
	
	    # Execute query.
        All_catalogue 	= sql.execute_query(con, myQuery)

        '''
        ax.scatter(myData_6[x], myData_6[y],myData_6[z], label=sim_name, s=1.0)
        ax.set_xlabel('x(cMpc)',fontweight='bold')
        ax.set_ylabel('y(cMpc)',fontweight='bold')
        ax.set_zlabel('z(cMpc)',fontweight='bold')
        '''


        Data6 = np.zeros((All_catalogue.shape[0],4),dtype=np.float64)
        Data6[:,0] = All_catalogue['x']
        Data6[:,1] = All_catalogue['y']
        Data6[:,2] = All_catalogue['z']
        Data6[:,3] = All_catalogue['mass']
        Data6 = Data6[Data6[:,3]>1.0e6]


        np.savetxt("Data_M6.csv", Data6,header='X,Y,Z,Mass',delimiter=',')

        Data13 = np.zeros((All_catalogue.shape[0], 4), dtype=np.float64)
        Data13[:, 0] = All_catalogue['x']
        Data13[:, 1] = All_catalogue['y']
        Data13[:, 2] = All_catalogue['z']
        Data13[:, 3] = All_catalogue['mass']
        Data13 = Data13[Data13[:, 3] > 1.0e13]

        np.savetxt("Data_M13.csv", Data6,header='X,Y,Z,Mass',delimiter=',')
        print Data13.shape,Data6.shape


bin_list = np.arange(0.5,max(Data6[:,2]),4.0)


filelist = []
filfile = open('fil_unique.txt','w')
for i in range(len(bin_list)-1):


    slice_Data6 = Data6[(Data6[:,2]>=bin_list[i])&(Data6[:,2]<=bin_list[i+1])]
    slice_Data13 = Data13[(Data13[:,2]>=bin_list[i])&(Data13[:,2]<=bin_list[i+1])]

    #print i,bin_list[i],max(Data6[:,2])
    #print slice_Data13.shape,slice_Data6.shape

    if(slice_Data13.shape[0]>1):
        file = 'SLICE_DATA'+str(i)+'M6'+'.dat'
        file1 = 'SLICE_DATA' + str(i) + 'M13' + '.dat'

        np.savetxt(file,slice_Data6[:,0:2],header='px py')
        np.savetxt(file1, slice_Data13[:, 0:2], header='px py')

        fig2 = plt.figure()
        plt.scatter(slice_Data6[:,0],slice_Data6[:,1], label=sim_name, s=1.0)
        plt.scatter(slice_Data13[:,0],slice_Data13[:,1], color='r', label=sim_name, s=25.0)
        plt.savefig('sliceno'+str(i)+'.png')
        fig2.clear()

        call(['../DisPerSE/bin/delaunay_2D',file,'-btype', 'periodic'])

        NDfile = file + '.NDnet'

        call(['../DisPerSE/bin/mse',NDfile,'-nsig','3','-upSkl','-dumpManifolds' ,'J0a'])

        SKLfile = NDfile + '_s3.up.NDskl'
        #print SKLfile

        call(['../DisPerSE/bin/skelconv',SKLfile,'-smooth','10','-to','vtp'])

        call(['../DisPerSE/bin/skelconv',SKLfile,'-smooth','10','-to','NDskl_ascii'])

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

        fil_file = open('FILAMENTS'+str(i)+'.dat', 'r')

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
