import numpy as np
import matplotlib.pyplot as plt


def clip_data(data):
    
    data = data[data[:,0]>min(data[:,0] + 5.0)]
    data = data[data[:,0]<max(data[:,0] - 5.0)]
    
    data = data[data[:,1]>min(data[:,1] + 5.0)]
    data = data[data[:,1]<max(data[:,1] - 5.0)]

    return data


#data0 = np.loadtxt('Distances0.dat')
#data45 = np.loadtxt('Distances45.dat')
data60 = np.loadtxt('Distances60.dat')


from scipy.stats import ks_2samp

#print ks_2samp(data45[data45>10.0],data60[data60>10.0])



fig1,(ax1) = plt.subplots(1,1,figsize=(8,6))
#ax1.hist(data0[data0>10.0],bins='auto',histtype='step'   ,label='Angle = 0.0')
#ax1.hist(data45[data45>10.0],bins='auto',histtype='step' ,label='Angle = 45.0')
ax1.hist(data60[data60>10.0],bins='auto',histtype='step' ,label='Angle = 60.0)')
ax1.set_ylabel('N')
ax1.set_xlabel('Lenth(Mpc)')
fig1.legend(loc=1)
plt.savefig('Angle_historgram.png',dpi=600)


#y1,binEdges1=np.histogram(data0[data0>10.0],bins='auto')
#bincenters1 = 0.5*(binEdges1[1:]+binEdges1[:-1])
#y2,binEdges2=np.histogram(data45[data45>10.0],bins='auto')
#bincenters1 = 0.5*(binEdges2[1:]+binEdges2[:-1])
y3,binEdges3=np.histogram(data60[data60>10.0],bins='auto')
bincenters3 = 0.5*(binEdges3[1:]+binEdges3[:-1])


plt.figure(2, figsize=(8,6))
#plt.plot(binEdges1[:-1], y1,'-o', color='m', linewidth=2, ms=8, label='Angle = 0.0', alpha=0.5)
#plt.plot(binEdges2[:-1], y2,'-o', color='peru', linewidth=2, ms=8, label='Angle = 45.0', alpha=0.5)
plt.plot(binEdges3[:-1], y3,'-o', color='r', linewidth=2, ms=8   ,label='Angle = 60.0)', alpha=0.5)  
plt.ylabel('N')
plt.xlabel('Lenth(Mpc)')

plt.legend()
plt.savefig('Angle_historgram_line.png',dpi=600)

plt.show()

