import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.constants import c
from astropy.cosmology import WMAP9 as cosmo
from scipy import spatial
from astropy.coordinates import search_around_3d
from sympy.geometry import *
from numpy.linalg import norm




class Analysis:
     

     def __init__(self,infile):
         radius = 3.0
         limit_log = 3.0
         data_fil = np.loadtxt(infile)
         data_coma = np.genfromtxt('all_data_paraview.csv',delimiter=',',skip_header=1)


         
         RA_fil = data_fil[:,0]
         DEC_fil = data_fil[:,1]



         RA = data_coma[:,3]
         DEC = data_coma[:,4]
         
         
         log_field_values = data_coma[:,1]
         
         mean_z = 0.026
         C = mean_z*(c.to('km/s'))/cosmo.H(0.0)
         
         lenght_dist=[]
         
        

         data_fil = np.array([RA_fil,DEC_fil]).T
         
         data_fil = data_fil[data_fil[:,0].argsort()]
         
         #print data_fil



         Sky_all_points = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, distance=C)
         Filament_all_points = SkyCoord(ra=data_fil[:,0]*u.degree,dec=data_fil[:,1]*u.degree, distance=C)
         fil_value,sky_values,sep,dist = search_around_3d(Filament_all_points,Sky_all_points,distlimit=radius*u.Mpc)
         
         
         
         
         #get the unique values of the points
         data = np.array([fil_value,sky_values,sep,dist]).T
         data = data[data[:,3].argsort()]
         
         
         print data.shape
         
         data_fil_ind = data[:,0].astype(int)
         data_gal_ind = data[:,1].astype(int)
         data_dist_gal = data[:,3]
         
         
         nearby_gal,indexes = np.unique(data_gal_ind,return_index=True)
         near_fil_point= data_fil_ind[indexes]
         dist_fil=data_dist_gal[indexes]
         
         
         
         '''

         #plt.figure(1)
         fig1,ax1 = plt.subplots(1,1,figsize=(8,6))
         ax1.scatter(RA_fil,DEC_fil,s=2,color='k')
         cs = ax1.scatter(RA[nearby_gal],DEC[nearby_gal],s=1,c=dist_fil,cmap='jet')
         ax1.set_xlim(min(RA),max(RA))
         ax1.set_ylim(min(DEC),max(DEC))
         ax1.set_xlabel('RA')
         ax1.set_ylabel('DEC')
         cbar = plt.colorbar(cs)
         cbar.set_label('Distance(Mpc)')
         fig1.savefig('Filament_colored_dist_value'+'radius'+str(radius)+'.png',dpi=600)
         #fig1.show()


         #plt.figure(1)
         fig3,ax3 = plt.subplots(1,1,figsize=(8,6))
         ax3.scatter(RA_fil,DEC_fil,s=2,color='k')
         cs = ax3.scatter(RA[nearby_gal],DEC[nearby_gal],s=1,c=log_field_values[nearby_gal],cmap='jet')
         ax3.set_xlim(min(RA),max(RA))
         ax3.set_ylim(min(DEC),max(DEC))
         ax3.set_xlabel('RA')
         ax3.set_ylabel('DEC')
         cbar = plt.colorbar(cs)
         cbar.set_label('log Field Value')
         fig3.savefig('Filament_colored_log_f_value'+'.png',dpi=600)
         #fig1.show()
         '''


         RA_F = RA[nearby_gal]
         DEC_F = DEC[nearby_gal]
         
         fil_log = log_field_values[nearby_gal]
         
         indx1, = np.where(fil_log<limit_log)
         
         
         
         
         Filtered_Sky_all_points = SkyCoord(ra=RA_F[indx1]*u.degree, dec=DEC_F[indx1]*u.degree, distance=C)
         
         #Filtered_Sky_all_points = SkyCoord(ra=RA_F*u.degree, dec=DEC_F*u.degree, distance=C)
         
         Filament_all_points = SkyCoord(ra=RA_fil*u.degree,dec=DEC_fil*u.degree, distance=C)
         fil_value,sky_values,sep,dist = search_around_3d(Filament_all_points,Filtered_Sky_all_points,distlimit=radius*u.Mpc)
         
         
         data = np.array([fil_value,sky_values,sep,dist]).T
         data = data[data[:,3].argsort()]
         
         
         print data.shape
         
         data_fil_ind = data[:,0].astype(int)
         data_gal_ind = data[:,1].astype(int)
         data_dist_gal = data[:,3].astype(int)
         
         
         nearby_gal,indexes = np.unique(data_gal_ind,return_index=True)
         near_fil_point= data_fil_ind[indexes]
         dist_fil=data_dist_gal[indexes]
         
         #print nearby_gal,near_fil_point,dist_fil
         
         #calculating the distance of all these points from the filaments
         
         
         d=np.zeros((len(nearby_gal),1))
         #print len(dist[indexes])
         
         for i in range(len(nearby_gal)):
             
             if(near_fil_point[i] == 0):
                        
                 P1 = SkyCoord(ra=RA_fil[0] * u.degree, dec=DEC_fil[0] * u.degree, distance=C)
                 P2 = SkyCoord(ra=RA_fil[1] * u.degree, dec=DEC_fil[1] * u.degree, distance=C)
                 p1 = np.array([P1.cartesian.x.value, P1.cartesian.y.value]).T
                 p2 = np.array([P2.cartesian.x.value, P2.cartesian.y.value]).T
                 near_halo_RA = RA[nearby_gal[i]]
                 near_halo_DEC = DEC[nearby_gal[i]]
                 P3 = SkyCoord(ra=near_halo_RA * u.degree, dec=near_halo_DEC * u.degree, distance=C)
                 p3 = np.array([P3.cartesian.x.value, P3.cartesian.y.value]).T
                 d[i,0] = np.abs((p2[1] - p1[1]) * p3[0] - (p2[0] - p1[0]) * p3[1] + p2[0] * p1[1] - p2[1] * p1[0]) / np.sqrt((p1[0] - p2[0])**2 + (p1[1]-p2[1])**2)
                 if(d[i,0]>dist_fil[i]):
                      d[i,0] = dist_fil[i]
                     
             elif (0 < near_fil_point[i] < (len(RA_fil)-2)):
                    P1 = SkyCoord(ra=RA_fil[near_fil_point[i]]*u.degree,dec=DEC_fil[near_fil_point[i]]*u.degree,distance=C)
                    P2 = SkyCoord(ra=RA_fil[near_fil_point[i] + 1]*u.degree,dec=DEC_fil[near_fil_point[i]+1]*u.degree,distance=C)
                    p1 = np.array([P1.cartesian.x.value, P1.cartesian.y.value]).T
                    p2 = np.array([P2.cartesian.x.value, P2.cartesian.y.value]).T
                    near_halo_RA = RA[nearby_gal[i]]
                    near_halo_DEC = DEC[nearby_gal[i]]
                    P3 = SkyCoord(ra=near_halo_RA * u.degree, dec=near_halo_DEC * u.degree, distance=C)
                    p3 = np.array([P3.cartesian.x.value, P3.cartesian.y.value]).T
                    tem_dist1 = np.abs((p2[1]-p1[1])*p3[0]-(p2[0]-p1[0])*p3[1]+p2[0]*p1[1]-p2[1]*p1[0])/norm(p2 - p1)
                    P1 = SkyCoord(ra=RA_fil[near_fil_point[i]-1]*u.degree,dec=DEC_fil[near_fil_point[i]-1]*u.degree,distance=C)
                    P2 = SkyCoord(ra=RA_fil[near_fil_point[i]]*u.degree,dec=DEC_fil[near_fil_point[i]]*u.degree,distance=C)
                    p1 = np.array([P1.cartesian.x.value, P1.cartesian.y.value]).T
                    p2 = np.array([P2.cartesian.x.value, P2.cartesian.y.value]).T
                    near_halo_RA = RA[nearby_gal[i]]
                    near_halo_DEC = DEC[nearby_gal[i]]
                    P3 = SkyCoord(ra=near_halo_RA * u.degree, dec=near_halo_DEC * u.degree, distance=C)
                    p3 = np.array([P3.cartesian.x.value, P3.cartesian.y.value]).T
                    
                    tem_dist2 = np.abs((p2[1]-p1[1])*p3[0]-(p2[0]-p1[0])*p3[1]+p2[0]*p1[1]-p2[1]*p1[0])/norm(p2 - p1)
               
                    if (tem_dist1 < tem_dist2):
                         d[i, 0] = tem_dist1
                    else:
                         d[i, 0] = tem_dist2
                         
                         
                    if (dist_fil[i]<d[i]):
                         d[i,0] = dist_fil[i]
                         
             elif (near_fil_point[i] == (len(RA_fil) - 1)):

                    P1 = SkyCoord(ra=RA_fil[near_fil_point[i]] * u.degree, dec=DEC_fil[near_fil_point[i]] * u.degree, distance=C)
                    P2 = SkyCoord(ra=RA_fil[near_fil_point[i] - 1] * u.degree, dec=DEC_fil[near_fil_point[i] - 1] * u.degree, distance=C)
                    p1 = np.array([P1.cartesian.x.value, P1.cartesian.y.value]).T
                    p2 = np.array([P2.cartesian.x.value, P2.cartesian.y.value]).T
                    near_halo_RA = RA[nearby_gal[i]]
                    near_halo_DEC = DEC[nearby_gal[i]]
                    P3 = SkyCoord(ra=near_halo_RA * u.degree, dec=near_halo_DEC * u.degree, distance=C)
                    p3 = np.array([P3.cartesian.x.value, P3.cartesian.y.value]).T
                    d[i, 0] = np.abs((p2[1]-p1[1])*p3[0]-(p2[0]-p1[0])*p3[1]+p2[0]*p1[1]-p2[1]*p1[0])/norm(p2 - p1)
                    
                    if (dist_fil[i]<d[i]):
                        d[i,0] = dist_fil[i]
             self._ra = RA_F[indx1]
             self._dec = DEC_F[indx1]
             self._dist = d

             self._ra_fil = RA_fil
             self._dec_fil = DEC_fil


     def plotting(self,ax):

          
                        
          #ax.scatter(self._ra_fil,self._dec_fil,s=2,color='k')
          #cs = ax2.scatter(RA_F,DEC_F,c=dist_fil,s=1.5,cmap='jet')
          #cs = ax.scatter(self._ra,self._dec,c=self._dist,s=1.5,cmap='jet')
          #ax2.set_xlim(min(RA),max(RA))
          #ax2.set_ylim(min(DEC),max(DEC))
          #ax.set_xlabel('RA')
          #ax.set_ylabel('DEC')
          #cbar = plt.colorbar(cs)
          #cbar.set_label('distance(Mpc)')
          #fig2.savefig('Filament_with_logf_lt_'+str(limit_log)+'radius'+str(radius)+'.png',dpi=600)
          #fig1.show()

          a = np.array([self._ra,self._dec,self._dist[:,0]])

          print a.shape

          with open("Filament_galaxy.dat", "a") as myfile:
               np.savetxt(myfile,np.array([self._ra,self._dec,self._dist[:,0]]).T)
               myfile.write('\n')

          #return cs
                        
                        
                        
