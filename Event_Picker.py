import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
import eagleSqlTools as sql
import fileinput




class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        #print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()



def onpick3(event):
    ind = event.ind
    xy = event.artist.get_offsets()
    print xy[ind].shape
    min_ind=0
    if len(xy[ind]) > 1:
        msx, msy = event.mouseevent.xdata, event.mouseevent.ydata
        distance =  np.sqrt((xy[ind][:,0]-msx)**2 + (xy[ind][:,1]-msy)**2)
        min_ind = np.argmin(distance)

    #s = event.artist.get_gid()
    outfile.writelines(str(xy[ind][min_ind,0])+' '+str(xy[ind][min_ind,1])+'\n')
    print xy[ind][min_ind,0],xy[ind][min_ind,1]

from numpy import genfromtxt


filelist = np.loadtxt('fil_unique.txt',dtype=np.string_)

for file in filelist:

        file1 = 'SLICE_DATA'+file.split('.')[0][15:]+'M13.dat'
        outfile = open('LINE'+file.split('.')[0][15:]+'.dat','w')

        clus_data = np.genfromtxt(file1)
        print clus_data.shape
        data = genfromtxt(file,delimiter=',')
        fig, ax = plt.subplots()
        col = ax.scatter(data[:,0], data[:,1],color='k',s=1,picker=True)
        ax.scatter(clus_data[:,0],clus_data[:,1],color='b',s=30,picker=True)
        fig.canvas.mpl_connect('pick_event', onpick3)
        line, = ax.plot(np.NaN, np.NaN, '-', color='r', label='')  # empty line
        linebuilder = LineBuilder(line)

        plt.show()

