import yt
from yt.units import Mpc

yt.enable_parallelism()

ds = yt.load('/home/ankit/Python_Environments/EAGLE/DisPerSE/TESTING/RefL0025N0376/snapshot_027_z000p101/snap_027_z000p101.0.hdf5')


prj = yt.ProjectionPlot(ds, 2, 'density', width=2*Mpc)
