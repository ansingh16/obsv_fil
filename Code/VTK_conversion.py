from pyevtk.hl import pointsToVTK 

import numpy as np  

npoints = 3000  

data= np.loadtxt('Hal.dat')


x = data[:,0]
y= data[:,1]
z  = data[:,2]
mass = data[:,3]

x = np.ascontiguousarray(x, dtype=np.float64)
y = np.ascontiguousarray(y, dtype=np.float64)
z = np.ascontiguousarray(z, dtype=np.float64)
mass = np.ascontiguousarray(mass, dtype=np.float64)

pressure = np.random.rand(npoints)  

temp = np.random.rand(npoints)  

pointsToVTK("./points", x,y,z, data = {"mass" : np.log10(mass)})






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
