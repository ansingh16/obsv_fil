import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors

plt.style.use('seaborn-talk')


data_all_mass = pd.read_csv('Data_all_mass.csv',header=1,names=['x','y','z','SM','Metal','u_minus_r','g_minus_r'])


data_Groups = pd.read_csv('Data_Groups_and_cluster.csv',header=1,names=['x','y','z','SM','Metal','u_minus_r','g_minus_r'])

data_Field = pd.read_csv('Field_galaxies.csv',header=1,names=['x','y','z','SM','Metal','u_minus_r','g_minus_r'])

data_Filament = pd.read_csv('Filament_output.csv',header=1,names=['x','y','z','d_per','d_long','length_fil','SM','Metal','u_minus_r','g_minus_r'])




#fig1,(ax1) = plt.subplots(1,1,figsize=(8,6))

#ax1.hist(data_Filament['g_minus_r'],weights=np.zeros_like(data_Filament['g_minus_r']) + 1. / data_Filament['g_minus_r'].size)
# Now we format the y-axis to display percentage


#ax1.set_ylabel('fraction')
#ax1.set_xlabel(r'g - r')
#fig1.legend(loc=1)
#plt.savefig('Angle_historgram.png',dpi=600)


y1,binEdges1=np.histogram(data_Filament['u_minus_r'],weights=np.zeros_like(data_Filament['u_minus_r']) + 1. / data_Filament['u_minus_r'].size)
bincenters1 = 0.5*(binEdges1[1:]+binEdges1[:-1])
y2,binEdges2=np.histogram(data_Groups['u_minus_r'],weights=np.zeros_like(data_Groups['u_minus_r']) + 1. / data_Groups['u_minus_r'].size)
bincenters2 = 0.5*(binEdges1[1:]+binEdges1[:-1])
y3,binEdges3=np.histogram(data_Field['u_minus_r'],weights=np.zeros_like(data_Field['u_minus_r']) + 1. / data_Field['u_minus_r'].size)
bincenters3 = 0.5*(binEdges1[1:]+binEdges1[:-1])


fig1,ax1 = plt.subplots(1,1, figsize=(8,6))
ax1.plot(binEdges1[:-1], y1,'-o', color='m', linewidth=2, ms=8, label='Filaments', alpha=0.5)
ax1.plot(binEdges2[:-1], y2,'-o', color='peru', linewidth=2, ms=8, label='Groups', alpha=0.5)
ax1.plot(binEdges3[:-1], y3,'-o', color='r', linewidth=2, ms=8   ,label='Field', alpha=0.5)
ax1.set_ylabel('Fraction of Galaxies')
ax1.set_xlabel(r'u - r')
plt.legend(loc=1)
fig1.savefig('Fraction_gal_vs_ur.png')







i=0





data_Fil = data_Filament[data_Filament['d_per']<=5.0]

new_field = data_Filament[data_Filament['d_per']>5.0]

new_Field_gal = new_field[['x', 'y', 'z','SM','Metal','u_minus_r','g_minus_r']].copy()

All_field_gal = data_Field.append(new_Field_gal)



print data_Fil.shape,All_field_gal.shape


y1,binEdges1=np.histogram(data_Filament['g_minus_r'],weights=np.zeros_like(data_Filament['g_minus_r']) + 1. / data_Filament['g_minus_r'].size)
bincenters1 = 0.5*(binEdges1[1:]+binEdges1[:-1])
y2,binEdges2=np.histogram(data_Groups['g_minus_r'],weights=np.zeros_like(data_Groups['g_minus_r']) + 1. / data_Groups['g_minus_r'].size)
bincenters2 = 0.5*(binEdges1[1:]+binEdges1[:-1])
y3,binEdges3=np.histogram(data_Field['g_minus_r'],weights=np.zeros_like(data_Field['g_minus_r']) + 1. / data_Field['g_minus_r'].size)
bincenters3 = 0.5*(binEdges1[1:]+binEdges1[:-1])


fig2,ax2 = plt.subplots(1,1, figsize=(8,6))
ax2.plot(binEdges1[:-1], y1,'-o', color='m', linewidth=2, ms=8, label='Filaments', alpha=0.5)
ax2.plot(binEdges2[:-1], y2,'-o', color='peru', linewidth=2, ms=8, label='Groups', alpha=0.5)
ax2.plot(binEdges3[:-1], y3,'-o', color='r', linewidth=2, ms=8   ,label='Field', alpha=0.5)
ax2.set_ylabel('Fraction of Galaxies')
ax2.set_xlabel(r'g - r')
plt.legend(loc=1)
fig2.savefig('Fraction_gal_vs_gr.png')











fig4,(axa,axb,axc) = plt.subplots(1,3, figsize=(14,6),sharey=True)
l1 = axb.scatter(np.log10(data_Fil['SM']),data_Fil['u_minus_r'],c=np.log10(data_Fil['Metal']/0.012),s=5.0,label='Filament')
l2 =axc.scatter(np.log10(All_field_gal['SM']),All_field_gal['u_minus_r'],c=np.log10(All_field_gal['Metal']/0.012),s=5.0,label='Field')
l3 = axa.scatter(np.log10(data_Groups['SM']),data_Groups['u_minus_r'],c=np.log10(data_Groups['Metal']/0.012),s=5.0,label='Groups and Clusters')
axa.set_xlabel(r'log $M_{*} (M_{\odot})$')
axa.set_ylabel(r'u - r')
axa.set_title('Groups')
axb.set_xlabel(r'log $M_{*} (M_{\odot})$')
#axb.set_ylabel(r'u - r')
axb.set_title('Filament')
axc.set_xlabel(r'log $M_{*} (M_{\odot})$')
#axc.set_ylabel(r'u - r')
axc.set_title('Field')
#cbar_ax = fig4.add_axes([0.85, 0.15, 0.05, 0.7])
fig4.tight_layout()
fig4.subplots_adjust(wspace=0, hspace=0)
cbar = fig4.colorbar(l3,ax=[axa,axb,axc])
cbar.set_label(r'$Z/Z_{\odot}$')
fig4.savefig('color_vs_staller_mass')




from scipy import stats

#bin_median, bin_edges, binnumber = stats.binned_statistic(data_Fil['d_per'],data_Fil['g_minus_r'],statistic='median',bins=50)

#print bin_median

#bin_width = (bin_edges[1] - bin_edges[0])
#bin_centers = bin_edges[1:] - bin_width/2

'''
fig5,ax5 = plt.subplots(1,1,figsize=(8,6))
ax5.plot(bin_centers, bin_median,'-o', color='m', linewidth=2, ms=8, label='Filaments', alpha=0.5)
ax5.set_xlabel(r'$d_{per} (Mpc)$')
ax5.set_ylabel(r'g - r')
fig5.savefig('d_vs_dper.png')



fig6,ax6 = plt.subplots(1,1,figsize=(8,6))
y1,binEdges1=np.histogram(data_Filament['d_per'])
bincenters1 = 0.5*(binEdges1[1:]+binEdges1[:-1])

ax6.plot(bincenters1, y1,'-o', color='m', linewidth=2, ms=8, label='Filaments', alpha=0.5)
'''

sort_data = data_Fil.sort_values('d_per')

rolling = sort_data.rolling(window=800)
rolling_median = rolling.median()
fig6,ax6 = plt.subplots(1,1,figsize=(8,6))
rolling_median.plot.scatter(x='d_per',y='g_minus_r')

'''
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

