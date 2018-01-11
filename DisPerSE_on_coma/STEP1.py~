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
import Analysis


FILAMENT_FILE = 'COMA_LARGE_RA_DEC.dat.NDnet_s3.up.NDskl.BRK.ASMB.NDskl.BRK.ASMB.a.NDskl'


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
        file_list.append('filament'+'_'+ str(j)+'.dat')
        #file_list[j] = open('filament' + str(i)+'_'+str(j)+'.dat', 'r')

#print file_list

l=0

        
for j in range(1, len(lines)):
    if lines[j].startswith(' '):
        with open(file_list[l], 'a') as file:
            # print l,i,len(file_list),len(lines)
            #print l,count-1
            file.write(lines[j])
            
    else:
        l=l+1
            
    if(l>(count-1)):
        break



import os
for file in os.listdir("."):
    if file.startswith("filament_"):
        data = np.loadtxt(file)
        #plt.plot(data[:,0],data[:,1])
        lenght_dist=[]

        RA_fil = data[:,0]
        DEC_fil = data[:,1]
        
        length = 0.0
        for i in range(len(RA_fil)-1):
            c1 = SkyCoord(ra=RA_fil[i] * u.degree, dec=DEC_fil[i] * u.degree, distance=C)
            c2 = SkyCoord(ra=RA_fil[i+1] * u.degree, dec=DEC_fil[i+1] * u.degree, distance=C)

            lenght_dist.append(c1.separation_3d(c2).value)
            length += c1.separation_3d(c2)
        
