import eagleSqlTools as sql
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111)

# Array of chosen simulations. Entries refer to the simulation name and comoving box length.
mySims	= np.array([('RefL0050N0752', 50.)])

# This uses the eagleSqlTools module to connect to the database with your username and password.
# If the password is not given, the module will prompt for it.
con = sql.connect("asingh", password="GFB4ejc2")



for sim_name, sim_size in mySims:

        print sim_name
    
        # Construct and execute query for each simulation. This query returns the number of galaxies
        # for a given 30 pkpc aperture stellar mass bin (centered with 0.2 dex width).


        # Normalize by volume and bin width.
        myQuery = "SELECT \
        			0.1+floor(log10(AP.Mass_Star)/0.2)*0.2 as mass, \
        			count(*) as num \
                           FROM \
        			%s_SubHalo as SH, \
        			%s_Aperture as AP \
                    WHERE \
                        SH.GalaxyID = AP.GalaxyID and \
                        AP.ApertureSize =  30 and \
                        AP.Mass_Star > 1e8 and \
                        SH.SnapNum = 27 \
        		   GROUP BY \
        			    0.1+floor(log10(AP.Mass_Star)/0.2)*0.2 \
        		   ORDER BY \
        			    mass" % (sim_name, sim_name)

        # Execute query.
        myData = sql.execute_query(con, myQuery)

        # Normalize by volume and bin width.
        hist = myData['num'][:] / float(sim_size) ** 3.
        hist = hist / 0.2

        plt.plot(myData['mass'], np.log10(hist), label=sim_name, linewidth=2)

# Label plot.
plt.xlabel(r'log$_{10}$ M$_{*}$ [M$_{\odot}$]', fontsize=20)
plt.ylabel(r'log$_{10}$ dn/dlog$_{10}$(M$_{*}$) [cMpc$^{-3}$]', fontsize=20)
plt.tight_layout()
plt.legend()

plt.savefig('GSMF.png')
plt.show()
