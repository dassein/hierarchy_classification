import numpy as np
from statistics import NormalDist, mean, harmonic_mean
from numpy import apply_along_axis, std
from sklearn.cluster import AffinityPropagation
from numpy import load
ovl = lambda x, y: NormalDist(*x).overlap(NormalDist(*y))


def get_dict_feature(path_features):
    data_features = load(path_features)
    data_features.files.sort(key=lambda t: int(t))
    return dict((c, data_features[c]) for c in data_features.files)

# def estimate_distrib(feature, label):
#     """
#     Note: 
#         number of samples is N 
#         number of classes is C 
#         number of features is F
    
#     Args:
#         feature: shape (N, F)
#         label: shape (N), range [0, C-1]
    
#     Returns:
#         distrib: pair of (mu, std) for estimated Normal distribution
#             corresponding to each class and each feature
#             shape (C, F, 2), distrib[i][j] = [mu, std] are estimated mean and std
#             for i-th class and j-th feature
#     """
#     c, num_feature = max(label) + 1, feature.shape[-1]
#     list_feature = [[] * c]
#     for x, y in list(zip(feature, label)):
#         list_feature[y].append(x)
#     for y in range(c):
#         list_feature[y] = np.vstack(list_feature[y]) # shape [N_y, F]
#     distrib = np.zeros(c, num_feature, 2)
#     for y in range(c):
#         distrib[y][:][0] = apply_along_axis(mean,  0, list_feature[y])
#         distrib[y][:][1] = apply_along_axis(stdev, 0, list_feature[y])
#     return distrib

def estimate_distrib(dict_features: dict):
    c, _key = len(dict_features.keys()), next(iter(dict_features.keys()))
    num_feature = dict_features[_key].shape[-1]
    distrib = np.zeros((c, num_feature, 2))
    for y, key in enumerate(sorted(dict_features.keys(), key=lambda x: int(x))):
        distrib[y,:,0] = apply_along_axis(mean,  0, dict_features[key])
        distrib[y,:,1] = apply_along_axis(std, 0, dict_features[key])
    return distrib


def compute_similar(distrib):
    """
    Note: 
        number of classes is C 
        number of features is F
    
    Args:
        distrib: pair of (mu, std) for estimated Normal distribution
            corresponding to each class and each feature
            shape (C, F, 2), distrib[i, j] = [mu, std] are estimated mean and std
            for i-th class and j-th feature
    Return:
        similar: rescaled mean of the overlapping coeffient 
            between the distribution of features for 2 classes
    """
    c, num_feature, _ = distrib.shape
    similar = np.ones((c, c))
    pairs = list((i, j) for i in range(c) for j in range(i+1, c))
    for i, j in pairs:
        similar[i, j] = \
            mean([ovl(distrib[i, ind], distrib[j, ind]) for ind in range(num_feature)])
        similar[j, i] = similar[i, j]
    min_s = similar.min()
    similar -= min_s
    similar *= 1 / (1 - min_s) # similar.max() == 1
    return similar

def cluster_category(similar):
    # 'random_state' has been introduced in 0.23. in sklearn/cluster/_affinity_propagation.py
    # It will be set to None starting from 1.0 (renaming of 0.25) 
    # which means that results will differ at every function call.
    # Set 'random_state' to None to silence this warning, 
    # or to 0 to keep the behavior of versions <0.23.
    f = AffinityPropagation(affinity="precomputed", convergence_iter=15, random_state=0)
    clusters = f.fit(similar).labels_
    return clusters

def save_cluster(clusters, path_categories, path_clusters):
    with open(path_categories, 'r') as fp_in:
        for ind, row in enumerate(fp_in):
            c_tmp = int(row.split(" ")[-1])
            if ind == 0:
                c_min = c_tmp
            elif c_min > c_tmp:
                c_min = c_tmp
    with open(path_categories, 'r') as fp_in, open(path_clusters, 'w') as fp_out:
        for row in fp_in:
            ind = int(row.split(" ")[-1]) - c_min
            fp_out.write(row.rstrip() + " " + str(clusters[ind]) + "\n")

def get_num_cluster(path_clusters):
    with open(path_clusters, 'r') as fp:
        for ind, row in enumerate(fp):
            c_tmp = int(row.split(" ")[-1])
            if ind == 0:
                c_max = c_tmp
            elif c_max < c_tmp:
                c_max = c_tmp
    return c_max + 1


def get_clusters(path_clusters):
    list_t = []
    with open(path_clusters, 'r') as fp:
        for row in fp:
            s = row.split(" ")
            list_t.append((int(s[1]), int(s[2])))
    list_t.sort(key=lambda t: t[0])
    return [x for i, x in list_t]

def get_str_categories(path_categories):
    list_t = []
    with open(path_categories, 'r') as fp:
        for row in fp:
            s = row.split(" ")
            list_t.append((int(s[1]), s[0]))
    list_t.sort(key=lambda t: t[0])
    return [x for i, x in list_t]