import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eagleSqlTools as sql
import fileinput

def onpick3(event):
    ind = event.ind
    xy = event.artist.get_offsets()

    if len(ind) > 1:
        #print ind.shape
        datax = np.array(xy[ind])[:,0]
        datay = np.array(xy[ind])[:, 1]
        msx, msy = event.mouseevent.xdata, event.mouseevent.ydata

        dist = np.sqrt((np.array(datax) - msx) ** 2 + (np.array(datay) - msy) ** 2)
        ind = [ind[np.argmin(dist)]]

    #s = event.artist.get_gid()
    xs.append(xy[ind][0][0])
    ys.append(xy[ind][0][1])
    line.set_data(xs,ys)
    line.figure.canvas.draw()
    linefile.writelines(str(xy[ind][0][0])+'\t'+str(xy[ind][0][1])+'\n')
    print('near data:',xy[ind][0])



filelist = np.loadtxt('fil_unique.txt', dtype=np.string_)

#fig,ax = plt.subplots()

M6_Data = np.genfromtxt('Data_M6.csv',delimiter=',')

#print M6_Data.shape

bin_list = np.arange(0.5,max(M6_Data[:,2]),2.0)

bins = range(len(bin_list)-1)



df = pd.DataFrame()
d={}
for file in filelist:
    data = np.genfromtxt(file,delimiter=',')

    cluster_dat = np.loadtxt('SLICE_DATA' + file.split('.')[0][15:] + 'M13' + '.dat')


    with open('LINE' + file.split('.')[0][15:] + '.dat','w') as linefile:
        fig, ax = plt.subplots()
        #ax.scatter(cluster_dat[:,0],cluster_dat[:,1],c = 'r',s=30.0)
        col = ax.scatter(data[:,0], data[:,1],s=1 ,picker=True)
        ax.scatter(cluster_dat[:,0],cluster_dat[:,1],c = 'r',s=30.0)
        line, = ax.plot(np.NaN, np.NaN,'r')
        xs = list(line.get_xdata())
        ys = list(line.get_ydata())
        fig.canvas.mpl_connect('pick_event', onpick3)

        plt.show()
