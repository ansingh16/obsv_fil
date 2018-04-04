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
                        M.CentreOfMass_z as z,\
                        M.Velocity_x as Velx,\
                        M.Velocity_y as Vely,\
                        M.Velocity_z as Velz,\
                        M.SF_Hydrogen as SFH,\
                        M.NSF_Hydrogen as NSFH\
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

    All_catalogue = sql.execute_query(con, myQuery)

    Data = np.zeros((All_catalogue.shape[0], 9), dtype=np.float64)
    Data[:, 0] = All_catalogue['x']
    Data[:, 1] = All_catalogue['y']
    Data[:, 2] = All_catalogue['z']
    # Data[:,3] = All_catalogue['mass']
    Data[:, 3] = All_catalogue['SM']
    Data[:, 4] = All_catalogue['Metal']
    # Data[:,6] = All_catalogue['g_minus_r']
    Data[:, 5] = All_catalogue['u_minus_r']
    Data[:, 6] = All_catalogue['g_minus_r']
    Data[:, 7] = All_catalogue['SFH']
    Data[:, 8] = All_catalogue['NSFH']


np.savetxt('HI_data.csv',Data,delimiter=',',header='x,y,z,SM,Metal,u_minus_r,g_minus_r,SFH,NSFH')
