import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors

plt.style.use('seaborn-talk')


data_all_mass = pd.read_csv('Data_all_mass.csv',header=1,names=['x','y','z','Mass','SM','GALEX_FUV','GALEX_NUV','SDSS_r','g_minus_r','u_minus_r'])


data_Groups = pd.read_csv('Data_Groups_and_cluster.csv',header=1,names=['x','y','z','Mass','SM','GALEX_FUV','GALEX_NUV','SDSS_r','g_minus_r','u_minus_r'])

data_Field = pd.read_csv('Field_galaxies.csv',header=1,names=['x','y','z','Mass','SM','GALEX_FUV','GALEX_NUV','SDSS_r','g_minus_r','u_minus_r'])

data_Filament = pd.read_csv('Filament_output.csv',header=1,names=['x','y','z','d_per','d_long','length_fil','Mass','SM','GALEX_FUV','GALEX_NUV','SDSS_r','g_minus_r','u_minus_r'])




#fig1,(ax1) = plt.subplots(1,1,figsize=(8,6))

#ax1.hist(data_Filament['g_minus_r'],weights=np.zeros_like(data_Filament['g_minus_r']) + 1. / data_Filament['g_minus_r'].size)
# Now we format the y-axis to display percentage


#ax1.set_ylabel('fraction')
#ax1.set_xlabel(r'g - r')
#fig1.legend(loc=1)
#plt.savefig('Angle_historgram.png',dpi=600)


y1,binEdges1=np.histogram(data_Filament['g_minus_r'],weights=np.zeros_like(data_Filament['g_minus_r']) + 1. / data_Filament['g_minus_r'].size)
bincenters1 = 0.5*(binEdges1[1:]+binEdges1[:-1])
y2,binEdges2=np.histogram(data_Groups['g_minus_r'],weights=np.zeros_like(data_Groups['g_minus_r']) + 1. / data_Groups['g_minus_r'].size)
bincenters2 = 0.5*(binEdges1[1:]+binEdges1[:-1])
y3,binEdges3=np.histogram(data_Field['g_minus_r'],weights=np.zeros_like(data_Field['g_minus_r']) + 1. / data_Field['g_minus_r'].size)
bincenters3 = 0.5*(binEdges1[1:]+binEdges1[:-1])


fig1,ax1 = plt.subplots(1,1, figsize=(8,6))
ax1.plot(binEdges1[:-1], y1,'-o', color='m', linewidth=2, ms=8, label='Filaments', alpha=0.5)
ax1.plot(binEdges2[:-1], y2,'-o', color='peru', linewidth=2, ms=8, label='Groups', alpha=0.5)
ax1.plot(binEdges3[:-1], y3,'-o', color='r', linewidth=2, ms=8   ,label='Field', alpha=0.5)
ax1.set_ylabel('Fraction of Galaxies')
ax1.set_xlabel(r'g - r')
plt.legend(loc=1)




y1,binEdges1=np.histogram(data_Filament['GALEX_NUV']-data_Filament['SDSS_r']\
                          ,weights=np.zeros_like(data_Filament['GALEX_NUV']-data_Filament['SDSS_r']) + 1. / (data_Filament['GALEX_NUV']-data_Filament['SDSS_r']).size)
bincenters1 = 0.5*(binEdges1[1:]+binEdges1[:-1])
y2,binEdges2=np.histogram(data_Groups['GALEX_NUV']-data_Groups['SDSS_r'],\
                          weights=np.zeros_like(data_Groups['GALEX_NUV']-data_Groups['SDSS_r']) + 1. / (data_Groups['GALEX_NUV']-data_Groups['SDSS_r']).size)
bincenters2 = 0.5*(binEdges1[1:]+binEdges1[:-1])
y3,binEdges3=np.histogram(data_Field['GALEX_NUV']-data_Field['SDSS_r'],\
                          weights=np.zeros_like(data_Field['GALEX_NUV']-data_Field['SDSS_r']) + 1. / (data_Field['GALEX_NUV']-data_Field['SDSS_r']).size)
bincenters3 = 0.5*(binEdges1[1:]+binEdges1[:-1])


fig2,ax2 = plt.subplots(1,1, figsize=(8,6))
ax2.plot(binEdges1[:-1], y1,'-o', color='m', linewidth=2, ms=8, label='Filaments', alpha=0.5)
ax2.plot(binEdges2[:-1], y2,'-o', color='peru', linewidth=2, ms=8, label='Groups', alpha=0.5)
ax2.plot(binEdges3[:-1], y3,'-o', color='r', linewidth=2, ms=8   ,label='Field', alpha=0.5)
ax2.set_ylabel('Fraction of Galaxies')
ax2.set_xlabel(r'NUV - r')


fig3,ax3 = plt.subplots(1,1, figsize=(8,6))

ax3.scatter(np.log10(data_Field['SM']),data_Field['u_minus_r'],alpha=0.8,s=5.0,label='Field')
ax3.scatter(np.log10(data_Filament['SM']),data_Filament['u_minus_r'],alpha=0.8,s=5.0,label='Filament')
ax3.scatter(np.log10(data_Groups['SM']),data_Groups['u_minus_r'],alpha=0.8,s=5.0,label='Groups')
ax3.set_ylabel('u - r')
ax3.set_xlabel(r'log Stellar Mass($M_{\odot}$)')


fig3.legend(loc=1)




'''
y1,binEdges1=np.histogram(data0[data0>0.0],bins='auto')
bincenters1 = 0.5*(binEdges1[1:]+binEdges1[:-1])
y2,binEdges2=np.histogram(data45[data45>0.0],bins='auto')
bincenters2 = 0.5*(binEdges2[1:]+binEdges2[:-1])
y3,binEdges3=np.histogram(data60[data60>0.0],bins='auto')
bincenters3 = 0.5*(binEdges3[1:]+binEdges3[:-1])


plt.figure(2, figsize=(8,6))
plt.plot(binEdges1[:-1], y1,'-o', color='m', linewidth=2, ms=8, label='Angle = 0.0', alpha=0.5)
plt.plot(binEdges2[:-1], y2,'-o', color='peru', linewidth=2, ms=8, label='Angle = 45.0', alpha=0.5)   
plt.plot(binEdges3[:-1], y3,'-o', color='r', linewidth=2, ms=8   ,label='Angle = 60.0)', alpha=0.5)  
plt.ylabel('N')
plt.xlabel('Lenth(Mpc)')

plt.legend()
plt.savefig('Angle_historgram_line.png',dpi=600)
'''
plt.show()

