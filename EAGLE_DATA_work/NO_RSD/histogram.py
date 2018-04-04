import numpy as np
import matplotlib.pyplot as plt



data = np.loadtxt('Lengths.dat')


from scipy.stats import ks_2samp

#print ks_2samp(data45[data45>10.0],data60[data60>10.0])



fig1,(ax1) = plt.subplots(1,1,figsize=(8,6))
ax1.hist(data[data>0.0],bins='auto',histtype='step' )
ax1.set_ylabel('N')
ax1.set_xlabel('Lenth(Mpc)')
fig1.legend(loc=1)
plt.savefig('Angle_historgram.png',dpi=600)


y1,binEdges1=np.histogram(data[data>0.0],bins='auto')
bincenters1 = 0.5*(binEdges1[1:]+binEdges1[:-1])


plt.figure(2, figsize=(8,6))
plt.plot(binEdges1[:-1], y1,'-o', color='m', linewidth=2, ms=8, label='Angle = 60.0', alpha=0.5)
plt.ylabel('N')
plt.xlabel('Lenth(Mpc)')

plt.savefig('Angle_historgram_line.png',dpi=600)

plt.show()

