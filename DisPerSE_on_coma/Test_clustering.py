import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
#%matplotlib inline
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.75, 's' : 10, 'linewidths':0}



def clip_data(data):
    data = data[data[:, 3] > min(data[:, 3] + 5.0)]
    data = data[data[:, 3] < max(data[:, 3] - 5.0)]

    data = data[data[:, 4] > min(data[:, 4] + 5.0)]
    data = data[data[:, 4] < max(data[:, 4] - 5.0)]
    return data


def clustering(r,sam):

    dbsc = DBSCAN(eps = r, min_samples = sam).fit(file_RADEC_array)
    labels = dbsc.labels_
    core_samples = np.zeros_like(labels, dtype = bool)
    core_samples[dbsc.core_sample_indices_] = True
    labels = dbsc.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    return core_samples,labels,n_clusters_

def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)

    print labels.max()

    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

    plt.scatter(data[:,0], data[:,1], c=colors, **plot_kwds)

    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.set_xlim(max(data[:,0]),min(data[:,0]))
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)

    plt.savefig('Testing_dbscan.png')
    plt.show()

data_coma = np.genfromtxt('Coma_large_smriti.csv',delimiter=',')

#file_RADEC_array = np.array([data_coma[:,3],data_coma[:,4]]).T
data_coma = clip_data(data_coma)
print data_coma.shape



file_RADEC_array = np.array([data_coma[:,3],data_coma[:,4]]).T

#################################################################################################################
def exemplars(cluster_id, condensed_tree):
    raw_tree = condensed_tree._raw_tree
    # Just the cluster elements of the tree, excluding singleton points
    cluster_tree = raw_tree[raw_tree['child_size'] > 1]
    # Get the leaf cluster nodes under the cluster we are considering
    leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster_id)
    # Now collect up the last remaining points of each leaf cluster (the heart of the leaf)
    result = np.array([])
    for leaf in leaves:
        max_lambda = raw_tree['lambda_val'][raw_tree['parent'] == leaf].max()
        points = raw_tree['child'][(raw_tree['parent'] == leaf) &
                                   (raw_tree['lambda_val'] == max_lambda)]
        result = np.hstack((result, points))
    return result.astype(np.int)




import hdbscan
from scipy.spatial.distance import euclidean
import DBCV


clusterer = hdbscan.HDBSCAN(min_cluster_size=20).fit(file_RADEC_array)

hdbscan_labels = hdbscan.HDBSCAN(min_cluster_size=20,min_samples=10).fit_predict(file_RADEC_array)


hdbscan_score = DBCV.DBCV(file_RADEC_array, hdbscan_labels, dist_function=euclidean)
print(hdbscan_score)


'''
fig1,ax1=plt.subplots(1,1,figsize=(8,6))
pal = sns.color_palette('deep', np.unique(clusterer.labels_).max() + 1)
colors = [sns.desaturate(pal[col], sat) for col, sat in zip(clusterer.labels_,
                                                            clusterer.probabilities_)]
ax1.scatter(file_RADEC_array[:,0],file_RADEC_array[:,1], c=colors, **plot_kwds);
ax1.set_xlim(max(file_RADEC_array[:,0]),min(file_RADEC_array[:,0]))



###################################################################################################################
fig2,ax2=plt.subplots(1,1,figsize=(8,6))

tree = clusterer.condensed_tree_
ax2.scatter(file_RADEC_array[:,0], file_RADEC_array[:,1], c='grey', **plot_kwds)
for i, c in enumerate(tree._select_clusters()):
    c_exemplars = exemplars(c, tree)
    ax2.scatter(file_RADEC_array[:,0][c_exemplars], file_RADEC_array[:,1][c_exemplars], c=pal[i], **plot_kwds)

ax2.set_xlim(max(file_RADEC_array[:, 0]), min(file_RADEC_array[:, 0]))



#DISTANCE BASED MEMBERSHIP
####################################################################################################################

from scipy.spatial.distance import cdist

def min_dist_to_exemplar(point, cluster_exemplars, data):
    dists = cdist(np.array([data[point]]), data[cluster_exemplars.astype(np.int32)])
    return dists.min()

def dist_vector(point, exemplar_dict, data):
    result = {}
    for cluster in exemplar_dict:
        result[cluster] = min_dist_to_exemplar(point, exemplar_dict[cluster], data)
    return np.array(list(result.values()))

def dist_membership_vector(point, exemplar_dict, data, softmax=False):
    if softmax:
        result = np.exp(1./dist_vector(point, exemplar_dict, data))
        result[~np.isfinite(result)] = np.finfo(np.double).max
    else:
        result = 1./dist_vector(point, exemplar_dict, data)
        result[~np.isfinite(result)] = np.finfo(np.double).max
    result /= result.sum()
    return result


fig3,ax3 = plt.subplots(1,1,figsize=(8,6))

exemplar_dict = {c:exemplars(c,tree) for c in tree._select_clusters()}
colors = np.empty((file_RADEC_array.shape[0], 3))
for x in range(file_RADEC_array.shape[0]):
    membership_vector = dist_membership_vector(x, exemplar_dict, file_RADEC_array)
    color = np.argmax(membership_vector)
    saturation = membership_vector[color]
    colors[x] = sns.desaturate(pal[color], saturation)
ax3.scatter(file_RADEC_array[:,0], file_RADEC_array[:,1], c=colors, **plot_kwds);
ax3.set_xlim(max(file_RADEC_array[:, 0]), min(file_RADEC_array[:, 0]))





#OUTLIER WAY
##################################################################################################################


def max_lambda_val(cluster, tree):
    cluster_tree = tree[tree['child_size'] > 1]
    leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster)
    max_lambda = 0.0
    for leaf in leaves:
        max_lambda = max(max_lambda,
                         tree['lambda_val'][tree['parent'] == leaf].max())
    return max_lambda

def points_in_cluster(cluster, tree):
    leaves = hdbscan.plots._recurse_leaf_dfs(tree, cluster)
    return leaves



def merge_height(point, cluster, tree, point_dict):
    cluster_row = tree[tree['child'] == cluster]
    cluster_height = cluster_row['lambda_val'][0]
    if point in point_dict[cluster]:
        merge_row = tree[tree['child'] == float(point)][0]
        return merge_row['lambda_val']
    else:
        while point not in point_dict[cluster]:
            parent_row = tree[tree['child'] == cluster]
            cluster = parent_row['parent'].astype(np.float64)[0]
        for row in tree[tree['parent'] == cluster]:
            child_cluster = float(row['child'])
            if child_cluster == point:
                return row['lambda_val']
            if child_cluster in point_dict and point in point_dict[child_cluster]:
                return row['lambda_val']


def per_cluster_scores(point, cluster_ids, tree, max_lambda_dict, point_dict):
    result = {}
    point_row = tree[tree['child'] == point]
    point_cluster = float(point_row[0]['parent'])
    max_lambda = max_lambda_dict[point_cluster] + 1e-8 # avoid zero lambda vals in odd cases

    for c in cluster_ids:
        height = merge_height(point, c, tree, point_dict)
        result[c] = (max_lambda / (max_lambda - height))
    return result


def outlier_membership_vector(point, cluster_ids, tree,
                              max_lambda_dict, point_dict, softmax=True):
    if softmax:
        result = np.exp(np.array(list(per_cluster_scores(point,
                                                         cluster_ids,
                                                         tree,
                                                         max_lambda_dict,
                                                         point_dict
                                                        ).values())))
        result[~np.isfinite(result)] = np.finfo(np.double).max
    else:
        result = np.array(list(per_cluster_scores(point,
                                                  cluster_ids,
                                                  tree,
                                                  max_lambda_dict,
                                                  point_dict
                                                 ).values()))
    result /= result.sum()
    return result



cluster_ids = tree._select_clusters()
raw_tree = tree._raw_tree
all_possible_clusters = np.arange(file_RADEC_array.shape[0], raw_tree['parent'].max() + 1).astype(np.float64)
max_lambda_dict = {c:max_lambda_val(c, raw_tree) for c in all_possible_clusters}
point_dict = {c:set(points_in_cluster(c, raw_tree)) for c in all_possible_clusters}
colors = np.empty((file_RADEC_array.shape[0], 3))
for x in range(file_RADEC_array.shape[0]):
    membership_vector = outlier_membership_vector(x, cluster_ids, raw_tree,
                                                  max_lambda_dict, point_dict, False)
    color = np.argmax(membership_vector)
    saturation = membership_vector[color]
    colors[x] = sns.desaturate(pal[color], saturation)

fig4,ax4 = plt.subplots(1,1,figsize=(8,6))

ax4.scatter(file_RADEC_array[:,0],file_RADEC_array[:,1], c=colors, **plot_kwds);
ax4.set_xlim(max(file_RADEC_array[:, 0]), min(file_RADEC_array[:, 0]))



#THE MIDDLE WAY
########################################################################################################################

def combined_membership_vector(point, data, tree, exemplar_dict, cluster_ids,
                               max_lambda_dict, point_dict, softmax=False):
    raw_tree = tree._raw_tree
    dist_vec = dist_membership_vector(point, exemplar_dict, data, softmax)
    outl_vec = outlier_membership_vector(point, cluster_ids, raw_tree,
                                         max_lambda_dict, point_dict, softmax)
    result = dist_vec * outl_vec
    result /= result.sum()
    return result

colors = np.empty((file_RADEC_array.shape[0], 3))
for x in range(file_RADEC_array.shape[0]):
    membership_vector = combined_membership_vector(x, file_RADEC_array, tree, exemplar_dict, cluster_ids,max_lambda_dict, point_dict, False)
    color = np.argmax(membership_vector)
    saturation = membership_vector[color]
    colors[x] = sns.desaturate(pal[color], saturation)

fig5,ax5 = plt.subplots(1,1,figsize=(8,6))

ax5.scatter(file_RADEC_array[:,0], file_RADEC_array[:,1], c=colors, **plot_kwds);
ax5.set_xlim(max(file_RADEC_array[:, 0]), min(file_RADEC_array[:, 0]))


########################################################################################################################


def prob_in_some_cluster(point, tree, cluster_ids, point_dict, max_lambda_dict):
    heights = []
    for cluster in cluster_ids:
        heights.append(merge_height(point, cluster, tree._raw_tree, point_dict))
    height = max(heights)
    nearest_cluster = cluster_ids[np.argmax(heights)]
    max_lambda = max_lambda_dict[nearest_cluster]
    return height / max_lambda


colors = np.empty((file_RADEC_array.shape[0], 3))
for x in range(file_RADEC_array.shape[0]):
    membership_vector = combined_membership_vector(x, file_RADEC_array, tree, exemplar_dict, cluster_ids,
                                                   max_lambda_dict, point_dict, False)
    membership_vector *= prob_in_some_cluster(x, tree, cluster_ids, point_dict, max_lambda_dict)
    color = np.argmax(membership_vector)
    saturation = membership_vector[color]
    colors[x] = sns.desaturate(pal[color], saturation)

fig6,ax6 = plt.subplots(1,1,figsize=(8,6))

ax6.scatter(file_RADEC_array[:,0], file_RADEC_array[:,1], c=colors, **plot_kwds);
ax6.set_xlim(max(file_RADEC_array[:, 0]), min(file_RADEC_array[:, 0]))



plt.show()
'''