import numpy as np
import h5py
import sys

directory = sys.argv[1]


def read_dataset_dm_mass():
    """ Special case for the mass of dark matter particles. """
    f           = h5py.File(directory+'/snap_027_z000p101.0.hdf5', 'r')
    h           = f['Header'].attrs.get('HubbleParam')
    a           = f['Header'].attrs.get('Time')
    dm_mass     = f['Header'].attrs.get('MassTable')[1]
    n_particles = f['Header'].attrs.get('NumPart_Total')[1]

    # Create an array of length n_particles each set to dm_mass.
    m = np.ones(n_particles, dtype='f8') * dm_mass

    # Use the conversion factors from the mass entry in the gas particles.
    cgs  = f['PartType0/Mass'].attrs.get('CGSConversionFactor')
    aexp = f['PartType0/Mass'].attrs.get('aexp-scale-exponent')
    hexp = f['PartType0/Mass'].attrs.get('h-scale-exponent')
    f.close()

    # Convert to physical.
    #m = np.multiply(m, cgs * a**aexp * h**hexp, dtype='f8')

    return m




def read_dataset(itype, att, nfiles=16):
    """ Read a selected dataset, itype is the PartType and att is the attribute name. """

    # Output array.
    data = []

    # Loop over each file and extract the data.
    for i in range(nfiles):
        f = h5py.File(directory+'/snap_027_z000p101.%i.hdf5'%i, 'r')
        tmp = f['PartType%i/%s'%(itype, att)][...]
        data.append(tmp)

        # Get conversion factors.
        cgs     = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
        aexp    = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
        hexp    = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')

        # Get expansion factor and Hubble parameter from the header.
        a       = f['Header'].attrs.get('Time')
        h       = f['Header'].attrs.get('HubbleParam')

        f.close()

    # Combine to a single array.
    if len(tmp.shape) > 1:
        data = np.vstack(data)
    else:
        data = np.concatenate(data)

    # Convert to physical.
    if data.dtype != np.int32 and data.dtype != np.int64:
        data = np.multiply(data, cgs * a**aexp * h**hexp, dtype='f8')
        #data = np.multiply(data,1.0, dtype='f8')

    return data

import astropy.units as u

con = u.cm.to('Mpc')

Pos_data= read_dataset(1,'Coordinates')
#Gas_density_data= read_dataset(1,'Density')

#print 12.0<Pos_data[:,2]*con<12.5

Data = np.zeros((Pos_data.shape[0],3),dtype=np.float64)

print Data.shape
Data[:,:3] = Pos_data

#Data[:,3] = Dark_density_data

Data = Data[(12.0<Data[:,2]*con) & (Data[:,2]*con<12.1)]


#Data[:,3]  = np.average(Data,axis=1, weights=Data[:,3])

np.savetxt('Data.dat',Data)

from scipy.stats import gaussian_kde

print con

#xy = np.vstack([Data[:,0],Data[:,1]])
#z = gaussian_kde(xy)(xy)

#idx = z.argsort()
#x, y, z = Data[:,0][idx], Data[:,1][idx], z[idx]


import matplotlib.pyplot as plt

fig  = plt.figure(figsize=(8,6))
from matplotlib  import cm

plt.scatter(Data[:,0]*con,Data[:,1]*con, marker='.', s=1, linewidths=0,cmap=cm.jet)
plt.xlabel('x(Mpc)')
plt.ylabel('y(Mpc)')
plt.title('Slice Plot with 0.1 Mpc width')

#plt.hist2d(Data[:,0]*con, Data[:,1]*con, bins=(10, 10), cmap=plt.cm.jet)

#plt.colorbar(sc, label="Density")

fig.savefig('Plot.png')




