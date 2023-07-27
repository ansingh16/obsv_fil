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
mySims		= np.array([('RefL0100N1504', 100.)])

# This uses the eagleSqlTools module to connect to the database with your username and password.
# If the password is not given, the module will prompt for it.
con     	= sql.connect("asingh", password="GFB4ejc2")


for sim_name, sim_size in mySims:
    print sim_name

    # Construct and execute query for each simulation. This query returns the number of galaxies
    # for a given 30 pkpc aperture stellar mass bin (centered with 0.2 dex width).
    # MH.Group_M_Crit200 > 1.0e6 and \

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
			    MH.Group_M_Crit200 " % (sim_name)

    # Execute query.
    All_catalogue = sql.execute_query(con, myQuery)

    '''
    ax.scatter(myData_6[x], myData_6[y],myData_6[z], label=sim_name, s=1.0)
    ax.set_xlabel('x(cMpc)',fontweight='bold')
    ax.set_ylabel('y(cMpc)',fontweight='bold')
    ax.set_zlabel('z(cMpc)',fontweight='bold')
    '''

    Data6 = np.zeros((All_catalogue.shape[0], 4), dtype=np.float64)
    Data6[:, 0] = All_catalogue['x']
    Data6[:, 1] = All_catalogue['y']
    Data6[:, 2] = All_catalogue['z']
    Data6[:, 3] = All_catalogue['mass']
    Data6 = Data6[Data6[:, 3] > 1.0e6]

    np.savetxt("Data_all.dat", Data6[:, 0:3], header='px py pz')


    Data13 = np.zeros((All_catalogue.shape[0], 4), dtype=np.float64)
    Data13[:, 0] = All_catalogue['x']
    Data13[:, 1] = All_catalogue['y']
    Data13[:, 2] = All_catalogue['z']
    Data13[:, 3] = All_catalogue['mass']
    Data13 = Data13[Data13[:, 3] > 1.0e13]

    np.savetxt("Data_M13.dat", Data6[:,0:3], header='px py pz')
    print Data13.shape, Data6.shape

    file = "Data_all.dat"


    call(['/home/ankit/Python_Environments/EAGLE/DisPerSE/bin/delaunay_3D', file, '-btype', 'periodic'])

    NDfile = file + '.NDnet'

    call(['/home/ankit/Python_Environments/EAGLE/DisPerSE/bin/mse', NDfile,'-upSkl', '-forceLoops','-nsig','3.0'])

    SKLfile = NDfile + '_s3.up.NDskl'
    # print SKLfile

    call(['/home/ankit/Python_Environments/EAGLE/DisPerSE/bin/skelconv', SKLfile, '-to', 'vtp'])

    call(['/home/ankit/Python_Environments/EAGLE/DisPerSE/bin/skelconv', SKLfile, '-to', 'NDskl_ascii'])

    FILAMENT_FILE = SKLfile + '.a.NDskl'

    with open(FILAMENT_FILE) as infile, open('FILAMENTS' + '.dat', 'w') as outfile:
        copy=False
        for line in infile:
            if line.strip() == "[FILAMENTS]":
                copy = True
            elif line.strip() == "[CRITICAL POINTS DATA]":
                copy = False
            elif copy:
                outfile.write(line)

    outfile.close()
