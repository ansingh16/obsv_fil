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
from subprocess import call




def clip_data(data):
    data = data[data[:, 0] > min(data[:, 0] + 5.0)]
    data = data[data[:, 0] < max(data[:, 0] - 5.0)]

    data = data[data[:, 1] > min(data[:, 1] + 5.0)]
    data = data[data[:, 1] < max(data[:, 1] - 5.0)]

    return data




#file_RADEC_array = np.array([data_coma[:,3],data_coma[:,4]]).T
#file_RADEC_array = clip_data(file_RADEC_array)

filelist = []
filfile = open('fil_unique.txt','w')

file = 'Data_xyz.csv'

call(['/home/ankit/Python_Environments/EAGLE/DisPerSE/bin/delaunay_3D',file,'-btype', 'periodic'])

NDfile = file + '.NDnet'

call(['/home/ankit/Python_Environments/EAGLE/DisPerSE/bin/mse',NDfile,'-nsig','3','-upSkl','-forceLoops'])

SKLfile = NDfile + '_s3.up.NDskl'

call(['/home/ankit/Python_Environments/EAGLE/DisPerSE/bin/skelconv',SKLfile,'-breakdown','-assemble','60','-to','vtp'])

call(['/home/ankit/Python_Environments/EAGLE/DisPerSE/bin/skelconv',SKLfile,'-to','NDskl_ascii'])


FILAMENT_FILE = 'Data_xyz.csv.NDnet_s3.up.NDskl.a.NDskl'

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


from pathlib import Path

my_file = Path("./Lengths.dat")
if my_file.is_file():
    import os

    os.remove("Lengths.dat")




for f in range(1,int(lines[0])+1):
        file = 'filament_' + str(f) + '.dat'
        print file
        data = np.genfromtxt(file,delimiter=' ')
        #plt.plot(data[:,0],data[:,1])
        #print file
        X_fil = data[:,0]
        Y_fil = data[:,1]
        Z_fil = data[:,2]
        lenght_dist=[]
        length = 0.0
        for i in range(len(X_fil)-1):

            c1 = SkyCoord(w=X_fil[i], u=Y_fil[i], v=Z_fil[i],frame='galactic' ,unit='Mpc', representation='cartesian')
            c2 = SkyCoord(w=X_fil[i+1], u=Y_fil[i+1], v=Z_fil[i+1],frame='galactic', unit='Mpc', representation='cartesian')



            length += c1.separation_3d(c2)

        lenght_dist.append(length.value)

        with open("Lengths.dat",'a') as dist_file:
            dist_file.write(str(length.value)+'\n')

'''
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
'''