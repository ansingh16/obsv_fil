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

fig2,ax2 = plt.subplots(1,1,figsize=(8,6))
ax2.set_xlim(max(file_RADEC_array[:,0]),min(file_RADEC_array[:,0]))
ax2.set_ylim(min(file_RADEC_array[:, 1]), max(file_RADEC_array[:, 1]))
ax2.set_xlabel('RA')
ax2.set_ylabel('DEC')

ax2.scatter(file_RADEC_array[:,0],file_RADEC_array[:,1],s=1.5,color='red',alpha=0.6)

with open('clipped_file.dat','w') as cf:
    for f in range(1, int(lines[0]) + 1):
        file = 'filament_' + str(f) + '.dat'
        data = np.loadtxt(file)

        ax2.plot(data[:,0],data[:,1],linewidth=2.5,color='black')

        if(max(data[:,0])<max(file_RADEC_array[:,0]) and min(data[:,0])>min(file_RADEC_array[:,0]) and \
                       max(data[:,1])<max(file_RADEC_array[:,1]) and min(data[:,1])>min(file_RADEC_array[:,1])):

            cf.write(file+'\n')


clip_files = np.loadtxt('clipped_file.dat',dtype = np.string_)


fig1,ax1 = plt.subplots(1,1,figsize=(8,6))
ax1.set_xlim(max(file_RADEC_array[:,0]),min(file_RADEC_array[:,0]))
ax1.set_ylim(min(file_RADEC_array[:, 1]), max(file_RADEC_array[:, 1]))
ax1.set_xlabel('RA')
ax1.set_ylabel('DEC')
ax1.scatter(file_RADEC_array[:,0],file_RADEC_array[:,1],s=1.5,color='red',alpha=0.6)



for f in range(0,len(clip_files)):
        file = clip_files[f]
        print file
        data = np.genfromtxt(file,delimiter=' ')
        #plt.plot(data[:,0],data[:,1])
        lenght_dist=[]
        #print file
        RA_fil = data[:,0]
        DEC_fil = data[:,1]
        
        length = 0.0
        for i in range(len(RA_fil)-1):
            c1 = SkyCoord(ra=RA_fil[i] * u.degree, dec=DEC_fil[i] * u.degree, distance=C)
            c2 = SkyCoord(ra=RA_fil[i+1] * u.degree, dec=DEC_fil[i+1] * u.degree, distance=C)

            lenght_dist.append(c1.separation_3d(c2).value)
            length += c1.separation_3d(c2)
        
        with open("Distances60.dat",'a') as dist_file:
            dist_file.write(str(length.value)+'\n')



        ax1.plot(data[:,0],data[:,1],linewidth=2.5,color='black')



fig1.savefig('clipped_filaments.png',dpi=600)
fig2.savefig('all_filaments.png',dpi=600)

plt.show()