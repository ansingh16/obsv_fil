from pyevtk.hl import pointsToVTK 

import numpy as np  

npoints = 3000  

data= np.loadtxt('Data_xyz.csv',delimiter=',',skiprows=1)


x = data[:,0]
y= data[:,1]
z  = data[:,2]
mass = np.zeros((data.shape[0],3),dtype=np.float64)

x = np.ascontiguousarray(x, dtype=np.float64)
y = np.ascontiguousarray(y, dtype=np.float64)
z = np.ascontiguousarray(z, dtype=np.float64)
mass = np.ascontiguousarray(mass, dtype=np.float64)

pressure = np.random.rand(data.shape[0])  

temp = np.random.rand(npoints)  

pointsToVTK("./EAGLE_TOTAL_SAMPLE", x,y,z, data = {"mass" : pressure})






'''

from pyevtk.hl import pointsToVTK
import numpy as np

data= np.loadtxt('Halo19.dat')


x = data[:,0]
y = data[:,1]
z = data[:,2]

d={}

d['mass'] = data[:3]

print x.size,y.size,z.size

pointsToVTK('./PYENV_test',x,y,z,data=None)

'''
