from mayavi import mlab
import numpy as np


data=np.loadtxt('Halo19.dat')
x, y, z,mass = data[:,0],data[:,1],data[:,2],np.log10(data[:,3])
mlab.points3d(x, y, z,mass)

mlab.show()
