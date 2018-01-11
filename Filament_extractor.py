import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eagleSqlTools as sql
import fileinput

binsize = 2.0   # Mpc

from numpy import genfromtxt

#filelist = np.loadtxt('fil_unique.txt', dtype=np.string_)
filelist = ['filament_unique19.csv']

fig,ax = plt.subplots()

M6_Data = np.genfromtxt('Data_M6.csv',delimiter=',')

print M6_Data.shape

bin_list = np.arange(0.5,max(M6_Data[:,2]),2.0)

bins = range(len(bin_list)-1)



df = pd.DataFrame()
d={}
for file in filelist:
    #file1 = 'SLICE_DATA' + file.split('.')[0][15:] + 'M13.dat'
    data = np.genfromtxt('LINE' + file.split('.')[0][15:] + '.dat')
    #data = np.genfromtxt(file,delimiter=',')
    zval = bins[int(file.split('.')[0][15:])]
    #print zval

    DataZclip=M6_Data[(M6_Data[:, 2] >= bin_list[int(file.split('.')[0][15:])]) & (M6_Data[:, 2] <= bin_list[int(file.split('.')[0][15:]) + 1])]

    dat = np.zeros((DataZclip.shape[0],4))

    print bin_list[int(file.split('.')[0][15:])],bin_list[int(file.split('.')[0][15:])+1]

    for j in range(data.shape[0]):

        for i in range(DataZclip.shape[0]):
            #df.append([DataZclip[i, 0], DataZclip[i, 1], DataZclip[i, 2]])

            a = np.array((data[j,0],data[j,1],0.0))
            b = np.array((DataZclip[i,0],DataZclip[i,1],0.0))
            dist = np.linalg.norm(a-b)
            #print dist
            if(dist<=2.0):
                    #print "M here"
                    dat[i,0],dat[i,1],dat[i,2],dat[i,3] = DataZclip[i,0],DataZclip[i,1],DataZclip[i,2],DataZclip[i,3]

    #condition = np.mod(dat) != 0


    dat = dat[dat[:,0]!=0]

    np.savetxt('Halo' + file.split('.')[0][15:] + '.dat',dat)

    print dat.shape



