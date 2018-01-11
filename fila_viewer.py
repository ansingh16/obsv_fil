from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


data= np.loadtxt('Halo19.dat')

print data.shape


p=ax.scatter(data[:,0],data[:,1],data[:,2],c=np.log10(data[:,3]),s=np.log10(data[:,3]))
ax.set_zlim(0.0,50.0)
ax.set_xlim(0.0,50.0)
ax.set_ylim(0.0,50.0)
fig.colorbar(p)

plt.show()