import eagleSqlTools as sql
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Array of chosen simulations. Entries refer to the simulation name and comoving box length.
mySims		= np.array([('RefL0050N0752', 50.)])

# This uses the eagleSqlTools module to connect to the database with your username and password.
# If the password is not given, the module will prompt for it.
con     	= sql.connect("asingh", password="GFB4ejc2")



for sim_name, sim_size in mySims:

        print sim_name
    
        #Construct and execute query for each simulation. This query returns the number of galaxies
        #for a given 30 pkpc aperture stellar mass bin (centered with 0.2 dex width).
        myQuery = "SELECT \
			GroupCentreOfPotential_x as x, \
                        GroupCentreOfPotential_y as y, \
                        GroupCentreOfPotential_z as z \
                   FROM \
			%s_FOF as MH \
                   WHERE \
			MH.Group_M_Crit200 > 1e7 and \
			MH.SnapNum = 27 \
		        "%(sim_name)

# Execute query.
myData_10 	= sql.execute_query(con, myQuery)

print myData_10.shape

np.savetxt("DataM7.csv", myData_10[1:], delimiter=",")
