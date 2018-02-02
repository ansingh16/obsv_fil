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
#np.setdiff1d(array1, array2)
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import pandas as pd



def clip_data(data):
    data = data[data[:, 3] > min(data[:, 3] + 5.0)]
    data = data[data[:, 3] < max(data[:, 3] - 5.0)]

    data = data[data[:, 4] > min(data[:, 4] + 5.0)]
    data = data[data[:, 4] < max(data[:, 4] - 5.0)]
    return data



data_coma = np.genfromtxt('Coma_large_smriti.csv',delimiter=',')

#file_RADEC_array = np.array([data_coma[:,3],data_coma[:,4]]).T
data_coma = clip_data(data_coma)
print data_coma.shape



file_RADEC_array = np.array([data_coma[:,3],data_coma[:,4]]).T
save_data_array = np.array([data_coma[:,0],data_coma[:,1],data_coma[:,2]]).T


#print min(file_RADEC_array[:,0]),max(file_RADEC_array[:,0])
file_data_pos = file_RADEC_array


R = []
Num_clus = []

fil_file = open('FILAMENTS'+'.dat', 'r')
lines = fil_file.readlines()
total_filaments = int(lines[0])

#FINDING CLUSTERS AND GROUPS USING DBSCAN
import hdbscan

from sklearn.metrics import silhouette_samples, silhouette_score

clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=10)


# Create a subplot with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)

ax1.set_xlim([-0.1, 1])
cluster_labels = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=10).fit_predict(file_RADEC_array)

n_clusters=cluster_labels.max()
silhouette_avg = silhouette_score(file_RADEC_array, cluster_labels)

print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)
sample_silhouette_values = silhouette_samples(file_RADEC_array, cluster_labels)

y_lower = 10
for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]


                print "Average cluster: ,value :",i,",",np.average(ith_cluster_silhouette_values)

                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = plt.cm.jet(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                          0, ith_cluster_silhouette_values,
                                          facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

colors = plt.cm.jet(cluster_labels.astype(float) / n_clusters)
ax2.scatter(file_RADEC_array[:, 0],file_RADEC_array[:, 1], marker='.', s=30, lw=0, alpha=0.7,c=colors)
ax2.set_xlim(max(file_RADEC_array[:,0]),min(file_RADEC_array[:,0]))

plt.savefig('Silhoutte_plot_minclus20_minsam10.png')
plt.show()


'''

for min_clus  in range(20,21):

    for mins in range(10,11):

        

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_clus, min_samples=mins)


        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-0.1, 1])
        cluster_labels = hdbscan.HDBSCAN(min_cluster_size=min_clus, min_samples=mins).fit_predict(file_RADEC_array)

        n_clusters=cluster_labels.max()
        silhouette_avg = silhouette_score(file_RADEC_array, cluster_labels)

        #if(silhouette_avg>0.2):

        print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)
        sample_silhouette_values = silhouette_samples(file_RADEC_array, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]


                print "Average cluster: ,value :",i,",",np.average(ith_cluster_silhouette_values)

                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = plt.cm.jet(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                          0, ith_cluster_silhouette_values,
                                          facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        colors = plt.cm.jet(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(file_RADEC_array[:, 0],file_RADEC_array[:, 1], marker='.', s=30, lw=0, alpha=0.7,c=colors)


        fig.savefig('Silh_'+str(min_clus)+'_'+str(mins)+'.png')

'''


