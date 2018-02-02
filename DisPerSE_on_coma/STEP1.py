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




def clip_data(data):
    data = data[data[:, 0] > min(data[:, 0] + 5.0)]
    data = data[data[:, 0] < max(data[:, 0] - 5.0)]

    data = data[data[:, 1] > min(data[:, 1] + 5.0)]
    data = data[data[:, 1] < max(data[:, 1] - 5.0)]

    return data


data_coma = np.genfromtxt('all_data_paraview.csv',delimiter=',',skip_header=1)


file_RADEC_array = np.array([data_coma[:,3],data_coma[:,4]]).T
file_RADEC_array = clip_data(file_RADEC_array)



FILAMENT_FILE = 'COMA_LARGE_RA_DEC.dat.NDnet_s3.up.NDskl.BRK.ASMB.a.NDskl'

#np.loadtxt('filament_83.dat')


with open(FILAMENT_FILE) as infile, open('FILAMENTS'+'.dat','w') as outfile:
    copy = False
    for line in infile:
        if line.strip() == "[FILAMENTS]":
            copy = True
        elif line.strip() == "[CRITICAL POINTS DATA]":
            copy = False
        elif copy:
            outfile.write(line)
            
        #outfile.close()
        #print line

with open(FILAMENT_FILE) as infile, open('Critical_Points' + '.dat', 'w') as outfile:
    copy = False
    for line in infile:
        if line.strip() == "[CRITICAL POINTS]":
            copy = True
        elif line.strip() == "[FILAMENTS]":
            copy = False
        elif copy:
            #print line.split(' ')
            if (not line.startswith(' ')):
                if(len(line)>10):
                    outfile.write('%d,%0.6f,%0.6f\n'%(int(line.split(' ')[0]),float(line.split(' ')[1]),float(line.split(' ')[2])))


outfile.close()

fil_file = open('FILAMENTS'+'.dat', 'r')

lines = fil_file.readlines()

#print len(lines)

total_filaments = lines[0]


k = 0

mean_z = 0.026
C = mean_z*(c.to('km/s'))/cosmo.H(0.0)


l=0
filRA = []
filDEC = []
for j in range(1, len(lines)):
    #print l
    if lines[j].startswith(' '):
        fp.write(lines[j])

    else :
            l=l+1
            fp = open('filament_'+str(l)+'.dat','w')

fp.close()

fig1,ax1 = plt.subplots(1,1,figsize=(10,6))
ax1.set_xlim(max(file_RADEC_array[:,0]),min(file_RADEC_array[:,0]))
ax1.set_ylim(min(file_RADEC_array[:, 1]), max(file_RADEC_array[:, 1]))
ax1.set_xlabel('RA')
ax1.set_ylabel('DEC')
ax1.scatter(file_RADEC_array[:,0],file_RADEC_array[:,1],s=1.5,color='black',alpha=0.3)

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

my_file = Path("./Distances60.dat")
if my_file.is_file():
    import os

    os.remove("Distances60.dat")



c_m = matplotlib.cm.magma

# create a ScalarMappable and initialize a data structure
s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
s_m.set_array([])


for f in range(1,int(lines[0])+1):
        file = 'filament_' + str(f) + '.dat'
        print file
        data = np.genfromtxt(file,delimiter=' ')
        #plt.plot(data[:,0],data[:,1])
        #print file
        RA_fil = data[:,0]
        DEC_fil = data[:,1]
        lenght_dist=[]
        length = 0.0
        for i in range(len(RA_fil)-1):
            c1 = SkyCoord(ra=RA_fil[i] * u.degree, dec=DEC_fil[i] * u.degree, distance=C)
            c2 = SkyCoord(ra=RA_fil[i+1] * u.degree, dec=DEC_fil[i+1] * u.degree, distance=C)

            length += c1.separation_3d(c2)

        lenght_dist.append(length.value)

        with open("Distances60.dat",'a') as dist_file:
            dist_file.write(str(length.value)+'\n')


        if length.value>10.0:

            ind = int(length.value)/5

            temp=ax1.plot(data[:, 0], data[:, 1], linewidth=1.5, color=s_m.to_rgba(ind))


data_crit = np.loadtxt('Critical_Points.dat',delimiter=',')

#colorbar_ax = fig1.add_axes([0.92, 0.1, 0.01, 0.8])
#cbar = plt.colorbar(s_m,cax=colorbar_ax)
#cbar.set_label('Length(Mpc)')
for i in range(data_crit.shape[0]):
    if(data_crit[i,0]==2.0):
        ax1.scatter(data_crit[i,1],data_crit[i,2],s=25,color='red')

    if (data_crit[i, 0] == 3.0):
        ax1.scatter(data_crit[i, 1], data_crit[i, 2], s=25, color='blue', label='minima')


            #if (data_crit[i, 0] == 3.0):
        #ax1.scatter(data_crit[i, 1], data_crit[i, 2], s=25, color='blue',label='BP')


plt.minorticks_on()

#plt.tight_layout()

L = np.loadtxt('Distances60.dat')



fig1.savefig('filaments_with_galaxies.png',dpi=600)

plt.show()