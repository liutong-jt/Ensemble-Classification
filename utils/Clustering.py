# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:12:27 2019

@author: LiuTong
"""

import scipy.spatial.distance as distance
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import math, copy, itertools, operator


def _get_dc(distance_array, dc_precent=0.02):
    '''
        得到距离阈值
    '''
    length = len(distance_array)
    arr = np.sort(distance_array)
    arr = arr.tolist()
    index = int(length * dc_precent)
    return arr[index]

def _get_rho_vector(distance_matrix, dc):
    rho_vector = np.sum(np.exp(-(distance_matrix/dc)**2),axis=1)
    return np.array(rho_vector)-1
#    rho_vector = np.sum(np.array(distance_matrix) < dc,axis=1)
#    n = distance_matrix.shape[0]
#    rho_vector = np.zeros([n,1])
#    for i in range(distance_matrix.shape[0]):
#        idx = distance_matrix[i] <= dc
#        rho_vector[i] = np.sum(np.exp(-(distance_matrix[i, idx]/dc)**2))
#    return np.array(rho_vector-1).reshape(n,)

def _get_delta_vector(distance_matrix, rho_vector):
    '''
        #delta_vector 存储当前与 密度比A点的所有点中(B,C,D) 与当前点最近的距离A最近的距离dist(A,B)
        Parameters:
            distance_matrix: n*n，其中[i,j]代表第i个点与等j个点的距离
            rho_vector: n*1 数据点的密度序列
    '''
    delta_vector = np.zeros(rho_vector.shape)
    rho_index = np.argsort(rho_vector).tolist()
    max_rho_index = rho_index[-1]
    while rho_index:
        temp_distance = []
        i = rho_index[0]        # i 是当前密度最小的点
        rho_index.remove(i)
        for j in rho_index:
            temp_distance.append(distance_matrix[i,j])
        if temp_distance:       # 如果 temp_distance不为空
            delta_vector[i] = min(temp_distance)    
        else:
            delta_vector[i] = 0
    delta_vector[max_rho_index] = np.max(distance_matrix[max_rho_index,:])
    return delta_vector

def _get_cluster_vector(distance_matrix, rho_vector, delta_vector, dc, gaussian_boundary):
    '''
        1.通过密度降序来遍历每一个点
        2.判断dc邻域内是否含有带标签点
            (1) 含有一个，给该点打上该标签，该有多个该点是边界点
            (2) 不含有， 该点是一个簇心
        3. 处理边界点，如果该点的是{1，2，3}的边界点，
                那么判断该点的密度是否大于
                {1，2，3}簇的密度最大值所在的gaussian，的左边界，的最小值
                如果大于则可以合并，否则不能
            #  1  ---1-----2-----3-----4-----5---*---6---7--- = 5
            #  2  ---1-----2-----3--*--4-----5-------6---7--- = 3
            #  3  ---1-----2--*--3-----4-----5---*---6---7--- = 2
            #  if boundary_rho > 2 可以合并
    '''
    cluster_center = []
    cluster_vector = np.zeros(rho_vector.shape)
    boundary_array = []
    
    # 先按照密度排序，若密度相同按照delta排序
    temp = np.zeros([rho_vector.shape[0], 3])
    temp[:,0] = rho_vector
    temp[:,1] = delta_vector
    temp[:,2] = range(len(rho_vector))
    temp = sorted(temp, key=operator.itemgetter(0,1), reverse=True)
    temp = np.array(temp)
    index = temp[:,2].astype(np.uint16)
#    index = np.argsort(-rho_vector)
    
    # The biggest rho is spllited into the first cluster
    cluster_vector[index[0]] = 1
    cluster_center.append(index[0])
    for i in index[1:]:
        Flag, belong, boundary = _can_link(i, distance_matrix, cluster_vector, dc)
        cluster_vector[i] = belong
        if Flag == False:
            cluster_center.append(i)

#         存在边界点，合并
        if len(boundary) != 0:
            boundary["point_rho"] = rho_vector[boundary["point_idx"]]
            boundary_array.append(boundary)
            if _can_merge3(boundary, gaussian_boundary, cluster_vector, rho_vector):
                belong = min(boundary["point_belong"])
                for ii in boundary["point_belong"]:
                    cluster_vector[cluster_vector==ii] = belong

    labels_set = set(cluster_vector)
    if 0 in labels_set:
        labels_set.remove(0)

    for i,j in enumerate(labels_set):
        cluster_vector[cluster_vector==j]=i+1

    return np.array(cluster_center), np.array(cluster_vector), boundary_array

def _remove_noise(cluster_vector, rho_vector, noise_boundary):
    '''
        先把小于噪声阈值的点置为噪声
        如果噪声中有可以和簇相连的点，就把该噪声点分到该簇中
    '''
    max_cv = int(max(cluster_vector))
    for i in range(1,max_cv+1):
        idx = np.where(cluster_vector==i)[0]
#        if len(idx)==0:
#            continue
        max_rho = np.max(rho_vector[idx])
        if len(idx) and  max_rho < noise_boundary:
            cluster_vector[idx] = 0

    labels_set = set(cluster_vector)
    if 0 in labels_set:
        labels_set.remove(0)
    for i,j in enumerate(labels_set):
        cluster_vector[cluster_vector==j]=i+1

    """
        簇心密度最大的簇标签置一，依次
    """
    #todo: 倘若两个数据的密度相同，delta不同时，排序不是二层排序。
    #todo: 需修改
    cluster_center = []
    rho_temp = copy.deepcopy(rho_vector)
    for i in range(1,int(max(cluster_vector))+1):
        rho_temp[np.where(cluster_vector!=i)]=-1
        ind = np.argmax(rho_temp)
        cluster_center.append(ind)
        rho_temp = copy.deepcopy(rho_vector)

    return cluster_center,cluster_vector

def get_cluster_center_By_vector(cluster_vector, rho_vector):
    cluster_center = []
    rho_temp = copy.deepcopy(rho_vector)
    for i in range(1,int(max(cluster_vector))+1):
        rho_temp[np.where(cluster_vector!=i)]=-1
        ind = np.argmax(rho_temp)
        cluster_center.append(ind)
        rho_temp = copy.deepcopy(rho_vector)
    return cluster_center
        
def _argmin_distance(point_idx, cluster_list, cluster_vector, distance_matrix):
    #distance_list 存储点与cluster_list=[1,3,5,6]的距离。distance_list=[1.distance(i,cluster_vector==1), 2.distance(i,cluster_vector==3)]
    distance_list = np.zeros(len(cluster_list))
    for i,j in enumerate(cluster_list):
        # Find the cluster closest to the point
        distance_list[i] = min(distance_matrix[point_idx, cluster_vector==j])

    return cluster_list[np.argmin(distance_list)]

def _can_link(point_idx, distance_matrix, cluster_vector, dc):
    '''
        Flag(bool): Flase stands for that the point is a cluster_center
        belong(int): The label of the cluster which the point belonged
    '''
    Flag = True
    belong_num = 0
    cluster_num = np.zeros(int(max(cluster_vector)))
    boundary = {}
    for i in range(int(max(cluster_vector))):
        if any(distance_matrix[point_idx,cluster_vector==i+1] < dc):
            cluster_num[i] = 1

    belong = np.where(cluster_num == 1)[0] + 1
    #形式 belong = array([1,2,4])
    belong_num = belong.shape[0]

    if belong_num == 1:
        belong = belong[0]

    elif belong_num > 1:
        min_temp = np.zeros(belong_num)
        for i,j in enumerate(belong):
                # Find the cluster closest to the point
            min_temp[i] = min(distance_matrix[point_idx, cluster_vector==j])
        belong_ans = belong[np.argmin(min_temp)]
        boundary = {}
        boundary["point_idx"] = point_idx
        boundary["point_belong"] = set(belong)
        #boundary_array = {point_idx, point_belong}
        #boundary_array.append(boundary)
        belong = belong_ans

    if belong_num == 0:
        Flag = False
        belong = max(cluster_vector) + 1

    return Flag, belong, boundary

def _get_gaussian_boundary(rho_vector):
    X = rho_vector.reshape(-1,1)
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 15)   #------------???
    for n_components in n_components_range:
    #    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='spherical').fit(X)
        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='spherical', random_state=1).fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
    means = best_gmm.means_.flatten()
    covars = best_gmm.covariances_.flatten()
    weights = best_gmm.weights_.flatten()
    x = []
    if len(means)==1:
        x.append(0)
    #    x.append(means[0]-3*covars[0]**0.5)
#        _showfigure(weights,means,covars**0.5, rho_vector,sorted(x))
        return x

    index = np.argsort(means)
    for i in range(len(index)-1):
        x.append(_split_gaussain(weights[index[i]], weights[index[i+1]], means[index[i]],means[index[i+1]],covars[index[i]]**0.5,covars[index[i+1]]**0.5))
    _showfigure(weights,means,covars**0.5, rho_vector,sorted(x))
    return sorted(x)

def _split_gaussain(Weight1, Weight2, Miu1, Miu2, delta1, delta2):
    """
        通过高斯密度函数计算两个高斯之间的交点
    """
    if not (delta1 and delta2):
        return (Miu1 + Miu2) / 2
    A = 1/(delta1**2) - 1/(delta2**2)
    B = -2*(Miu1/delta1**2 - Miu2/delta2**2)
    C = Miu1**2/delta1**2 - Miu2**2/delta2**2 - 2*math.log((Weight1*delta2)/(Weight2*delta1))
    x1 = (-B + (B**2 - 4*A*C)**0.5)/(2*A)
    x2 = (-B - (B**2 - 4*A*C)**0.5)/(2*A)
    if x1 < max(Miu1,Miu2) and x1 >min(Miu1,Miu2):
        return x1
    elif x2 < max(Miu1,Miu2) and x2 >min(Miu1,Miu2):
        return x2
    else :
        print('两正态分布之间无交点')
    return 0

def _can_merge3(boundary_dict, gaussian_boundary, cluster_vector, rho_vector):
    '''
        boundary_dict: is a dict type
            boundary_dict{"point_idx": int,
                          "point_belong": set,
                          "point_rho", float}
    '''
    L = len(gaussian_boundary)
    if L==1 and gaussian_boundary[0]==0:
        return False

    #---1-----2-----3-----4-----5---*---6---7--- = 5
    #---1-----2-----3--*--4-----5-------6---7--- = 3
    max_rho_min_boundary = []
    for cv in boundary_dict["point_belong"]:
        max_rho = max(rho_vector[cluster_vector==cv])
        for i,j in enumerate(reversed(gaussian_boundary)):
            if max_rho >= j:
                max_rho_min_boundary.append(j)
                break
            if i == L-1: # 如果小于最后一个高斯边界，则把阈值设置为0
                max_rho_min_boundary.append(0)
    if boundary_dict["point_rho"] > min(max_rho_min_boundary):
        return True
    else:
        return False
   

def _draw_gaussian_rho_in_scatter(data, rho_vector, gaussian_boundary):
    color = ['r','b','m','y','c','g','gold','silver','darkgrey']
    plt.scatter(data[:,0], data[:,1], s=8, c='k')
    for i in range(len(gaussian_boundary)):
        plt.scatter(data[:,0], data[:,1], s=8, c='k')
        if i==0:
            ind = rho_vector < gaussian_boundary[0]
        elif i!=len(gaussian_boundary)-1:
            ind = (rho_vector>gaussian_boundary[i-1]) * (rho_vector<gaussian_boundary[i])
        else:
            ind = rho_vector>gaussian_boundary[-1]
        plt.scatter(data[ind, 0], data[ind, 1],c=color[i])
        plt.show()

def _showfigure(weights,means,covars, rho_vector,gaussian_boundary):
    x_max = max(rho_vector)
    x_min = min(rho_vector)
    plt.plot(rho_vector, np.zeros(rho_vector.shape),'.k')
    for w,m,c in zip(weights, means, covars):
        x = np.arange(x_min,x_max,0.01)
        y=[]
        for i in x:
            y.append(w*_gaussian(i,m,c))
        plt.plot(x,y,'--k')
    plt.hist(rho_vector,bins=50,color='k',alpha=0.4)
    for x in gaussian_boundary:
        plt.plot([x,x],[0,0.01],'k')
    plt.show()

def _gaussian(x,mean,covar):
    left=1/(math.sqrt(2*math.pi)*covar)
    right=math.exp(-math.pow(x-mean,2)/(2*math.pow(covar,2)))
    return left*right

def _science_get_cluster_center(rho_vector, delta_vector):
    #rho > x, delta > y
    # cluster_center is a index_vector.[1xM],M is the number of cluster_center
    plt.figure()
    plt.plot(rho_vector, delta_vector,'.k')
    # plt.show()
    #x,y = 56.397,0.21915
    pos = plt.ginput(1)
    print(pos)
    x = pos[0][0]
    y = pos[0][1]
    # plt.close()

    center = []
    rho = rho_vector > x
    delta = delta_vector > y
    for i in range(len(rho_vector)):
        if rho[i] and delta[i]:
            center.append(i)
    return center

def _science_get_cluster_vector(distace_matrix, rho_vector, cluster_center):
    cluster_num = len(cluster_center)
    cluster_vector = np.zeros(rho_vector.shape)
    # 先给分配簇心
    for i,index in enumerate(cluster_center):
        cluster_vector[index] = i+1
    # 取从大到小排序的索引(不排序)
    sorted_rho_index = np.argsort(-rho_vector)
    for i in sorted_rho_index:
        if cluster_vector[i] == 0:
            temp_ind = np.zeros(cluster_num)
            for j in range(cluster_num):
                # cl is [T,T,T,F,F,F,F],cluser_vector == j 的位置为True
                cl = (cluster_vector==j+1)
                # cl 为True的索引,即在原数据中的位置
                cl_ind = np.where(cl==True)
                # 在j簇中,离sorted_rho_index[i],最近的点的原始数据的索引
                min_ind = np.argmin(distace_matrix[i][cl_ind])
                #不使用temp_dist[0]位置
                temp_ind[j] = cl_ind[0][min_ind]
            # 去掉第一个temp_dist[0],并且将其由'float64',转换为整型
            temp_ind = temp_ind.astype(int)
            temp_min_ind = np.argmin(distace_matrix[i][temp_ind])
            cluster_vector[i] = temp_min_ind+1
    return cluster_vector

def _science_get_hola(distance_matrix, rho_vector, cluster_vector,dc):
    m = len(rho_vector)
    max_cv = int(max(cluster_vector))
    hola_rho = np.zeros([max_cv+1,1])
    for i in range(m):
        for j in range(m):
            if cluster_vector[i] != cluster_vector[j] and distance_matrix[i,j]<dc:
                if rho_vector[i] > hola_rho[int(cluster_vector[i])]:
                    hola_rho[cluster_vector[i]] = rho_vector[i]
                if rho_vector[j]>hola_rho[int(cluster_vector[j])]:
                    hola_rho[cluster_vector[j]] = rho_vector[j]

    for i in range(1, max_cv+1):
        idx = np.where(cluster_vector==i)[0]
        for j in idx:
            if rho_vector[j]<hola_rho[i]:
                cluster_vector[j] = 0
    return cluster_vector

def _assign_noise(distance_matrix, cluster_vector):
    noise_idx = np.where(cluster_vector==0)[0]
#    cv_tmp = np.zeros(noise_idx.shape)
    
    for i, idx in enumerate(noise_idx):
        t = [0 for i in range(int(max(cluster_vector))+1)]
        for k in range(1, int(max(cluster_vector))+1):
            cv = np.where(cluster_vector==k)[0]
            t[k] = min(distance_matrix[idx][cv])
        cluster_vector[idx] = np.argmin(t[1:])+1
        
#    for i, idx in enumerate(noise_idx):
#        cluster_vector[idx] = cv_tmp[i]
    return cluster_vector
    
    
#############################
#import matlab.engine
#
#def data2matlab(data):
#    datamat = data.tolist()
#    datamat = matlab.double(datamat)
#    return datamat
#
#def get_rho_akde(data):
#    eng = matlab.engine.start_matlab()
#    
#    datamat = data2matlab(data)
#    pdf,X1,X2,bandwidth = eng.akde(datamat,nargout=4);
#    rho = eng.get_rho_after_akde(datamat,pdf,X1,X2,nargout=1)
#    rho_vector = np.array(rho)
#    eng.quit()
#    
#    rho_vector = np.nan_to_num(np.array(rho_vector))
#    # kde_x, kde_y = FFTKDE(bw='silverman').fit(rho_vector)()
#    # kde_y = FFTKDE(bw='ISJ').fit(rho_vector)(kde_x)
#    # plt.plot(kde_x, kde_y, label="FFTKDE with Improved Sheather Jones (ISJ)")
#    return rho_vector.flatten(),bandwidth
#    
#def resort_cluster_vector(cluster_vector, rho_vector):
#    labels_set = set(cluster_vector)
#    if 0 in labels_set:
#        labels_set.remove(0)
#    
#    cluster_center = get_cluster_center_By_vector(cluster_vector, rho_vector)
#    # 密度从大到小
#    cluster_center = [cluster_center[i] for i in np.argsort(-rho_vector[cluster_center])]
#    cluster_lable = cluster_vector[cluster_center]
#    
#    temp_vector = np.zeros(cluster_vector.shape)
#    for i,j in enumerate(labels_set):
#        cluster_vector[cluster_vector==j]=i+1
#    pass

        

class MyClustering(object):
    def __init__(self, dc_precent=0.02, dc=0, distance_matrix=None, distance_array=None, noise=1):
        self.dc_precent = dc_precent
        self.dc = dc
        self.distance_matrix = distance_matrix
        self.distance_array = distance_array
        self.noise = noise

    def fit(self,data):
        self.data = data
        if self.distance_matrix is None and self.distance_array is None:
            self.distance_array = distance.pdist(self.data, 'euclidean')
            self.distance_matrix = distance.squareform(self.distance_array)

        if self.dc== 0:
            self.dc = _get_dc(self.distance_array, self.dc_precent)
        self.rho_vector = _get_rho_vector(self.distance_matrix, self.dc)
        self.delta_vector = _get_delta_vector(self.distance_matrix, self.rho_vector)
        self.gaussian_boundary = _get_gaussian_boundary(self.rho_vector)
        self.cluster_center, self.cluster_vector, self.boundary_array = _get_cluster_vector(self.distance_matrix, self.rho_vector, self.delta_vector, self.dc, self.gaussian_boundary)
        if self.noise == 1:
            Noise_rho = min(self.gaussian_boundary)
            print(Noise_rho)
        else:
            Noise_rho = self.noise
        print('Noise_rho', Noise_rho)
        self.cluster_center, self.cluster_vector= _remove_noise(self.cluster_vector, self.rho_vector, Noise_rho)
        self.cluster_vector = np.array(self.cluster_vector, dtype=int)
        self.cluster_vector[self.rho_vector<Noise_rho]=0
        # self.cluster_vector = _assign_noise(self.distance_matrix, self.cluster_vector)        
        return self

    def draw_decision(self):
        plt.figure()
        color_iter = itertools.cycle(['r','b','m','y','c','g','gold','lightskyblue','darkgrey','peru','orange','coral','darkred','yellowgreen','cyan','burlywood'])
        plt.scatter(self.rho_vector, self.delta_vector, s=8, c='k')
        for i,color in zip(range(int(max(self.cluster_vector))), color_iter):
            plt.scatter(self.rho_vector[self.cluster_vector==i+1], self.delta_vector[self.cluster_vector== i+1], s=16, c=color)
            plt.scatter(self.rho_vector[self.cluster_center[i]], self.delta_vector[self.cluster_center[i]], s=16, c='k')
        plt.show()

    def draw_cluster(self):
        plt.figure()
        color_iter = itertools.cycle(['r','b','m','y','c','g','gold','lightskyblue','darkgrey','peru','orange','coral','darkred','yellowgreen','cyan','burlywood'])
        plt.scatter(self.data[:,0], self.data[:,1], s=8, c='k')
        for i,color in zip(range(int(max(self.cluster_vector))),color_iter):
            plt.scatter(self.data[self.cluster_vector== i+1,0], self.data[self.cluster_vector== i+1,1], s=8, c=color)
            plt.scatter(self.data[self.cluster_center[i],0], self.data[self.cluster_center[i],1], s=64, c='k', marker='*')
            # plt.text(data[cluster_center[i],0], data[cluster_center[i],1], str(i+1), size = 16)
        # pos = plt.ginput(30)
        # print(pos)
        plt.show()

    def draw_boundary(self):
        # 由于[{},{},{}]不可以再使用set()函数去重复数据，
        # 所以先将其转换成字符串，然后再去重复，
        # 最后再转换成集合
        single_boundary = {}
        point_belong = [str(belong['point_belong']) for belong in self.boundary_array]
        set_point_belong = set(point_belong)
        for s in set_point_belong:
            single_boundary[s] = 0
        for key, value in single_boundary.items():
            for belong in self.boundary_array:
                if str(belong['point_belong']) == key:
                    if belong['point_rho'] > value:
                        single_boundary[key] = belong

        plt.figure()
        color_iter = itertools.cycle(['r','b','m','y','c','g','gold','lightskyblue','darkgrey','peru','orange','coral','darkred','yellowgreen','cyan','burlywood'])
        plt.scatter(self.data[:,0], self.data[:,1], s=8, c='k')
        for i,color in zip(range(int(max(self.cluster_vector))),color_iter):
            plt.scatter(self.data[self.cluster_vector== i+1,0], self.data[self.cluster_vector== i+1,1], s=8, c=color)
            plt.scatter(self.data[self.cluster_center[i],0], self.data[self.cluster_center[i],1], s=64, c='k', marker='*')
        for boundary in single_boundary.values():
            idx = boundary["point_idx"]
            plt.plot(self.data[idx,0], self.data[idx,1],'.k')
        plt.show()

class DensityPeakClustering(object):
    def __init__(self, dc_precent=0.02, distance_matrix= None, distance_array=None, is_remove_noises=False):
        self.dc_precent = dc_precent
        self.is_remove_noises = is_remove_noises
        self.distance_matrix = distance_matrix
        self.distance_array = distance_array

    def fit(self,data):
        self.data = data
        self.distance_array = distance.pdist(self.data, 'euclidean')
        self.distance_matrix = distance.squareform(self.distance_array)
    #    distance_array = np.load('m_array.npy')
    #    self.distance_matrix = np.load('m.npy')
        self.dc = _get_dc(self.distance_array, self.dc_precent)
        self.rho_vector = _get_rho_vector(self.distance_matrix, self.dc)
        self.delta_vector = _get_delta_vector(self.distance_matrix, self.rho_vector)
        self.cluster_center = _science_get_cluster_center(self.rho_vector, self.delta_vector)
        self.cluster_vector = _science_get_cluster_vector(self.distance_matrix, self.rho_vector, self.cluster_center)
        self.cluster_vector = np.array(self.cluster_vector, dtype=int)
#        if self.is_remove_noises:
        self.label_no_noises=copy.deepcopy(self.cluster_vector)
        self.label_no_noises = _science_get_hola(self.distance_matrix, self.rho_vector, self.label_no_noises, self.dc)
        return self

    def draw_cluster(self,draw_noise=True):
        plt.figure()
        color_iter = itertools.cycle(['r','b','m','y','c','g','gold','lightskyblue','darkgrey','peru','orange','coral','darkred','yellowgreen','cyan','burlywood'])
        plt.scatter(self.data[:,0], self.data[:,1], s=8, c='k')
        if draw_noise:
            for i,color in zip(range(int(max(self.cluster_vector))),color_iter):
                plt.scatter(self.data[self.label_no_noises== i+1,0], self.data[self.label_no_noises== i+1,1], s=8, c=color)
                plt.scatter(self.data[self.cluster_center[i],0], self.data[self.cluster_center[i],1], s=64, c='k', marker='*')
                # plt.text(data[cluster_center[i],0], data[cluster_center[i],1], str(i+1), size = 16)
        else:
            for i,color in zip(range(int(max(self.cluster_vector))),color_iter):
                plt.scatter(self.data[self.cluster_vector== i+1,0], self.data[self.cluster_vector== i+1,1], s=8, c=color)
                plt.scatter(self.data[self.cluster_center[i],0], self.data[self.cluster_center[i],1], s=64, c='k', marker='*')
                # plt.text(data[cluster_center[i],0], data[cluster_center[i],1], str(i+1), size = 16)
        plt.show()
    def draw_decision(self):
        plt.figure()
        color_iter = itertools.cycle(['r','b','m','y','c','g','gold','lightskyblue','darkgrey','peru','orange','coral','darkred','yellowgreen','cyan','burlywood'])
        plt.scatter(self.rho_vector, self.delta_vector,s=16, c='k')
        for cc,color in zip(self.cluster_center,color_iter):
            plt.scatter(self.rho_vector[cc], self.delta_vector[cc],s=32, c=color)
        plt.show()
        
#class DensityPeakByHD(object):
#    def __init__(self, is_remove_noises=True):
#        self.is_remove_noises = is_remove_noises
#
#    def fit(self,data):
#        self.data = data
#        self.distance_matrix = distance.squareform(distance.pdist(self.data, 'euclidean'))
#        self.rho_vector, self.bandwidth = get_rho_akde(self.data)
#        # np.savetxt('rho.txt',self.rho_vector)
#        self.delta_vector = _get_delta_vector(self.distance_matrix, self.rho_vector)
#        self.cluster_center = _science_get_cluster_center(self.rho_vector, self.delta_vector)
#        self.cluster_vector = _science_get_cluster_vector(self.distance_matrix, self.rho_vector, self.cluster_center)
#        self.cluster_vector = np.array(self.cluster_vector, dtype=int)
#        self.label_no_noises = _science_get_hola(self.distance_matrix, self.rho_vector, self.cluster_vector, np.sqrt(self.bandwidth)/3.3)
#        return self
#    
#    def draw_cluster(self):
#        plt.figure()
#        color_iter = itertools.cycle(['r','b','m','y','c','g','gold','lightskyblue','darkgrey','peru','orange','coral','darkred','yellowgreen','cyan','burlywood'])
#        plt.scatter(self.data[:,0], self.data[:,1], s=8, c='k')
#        if True:
#            for i,color in zip(range(int(max(self.cluster_vector))),color_iter):
#                plt.scatter(self.data[self.label_no_noises== i+1,0], self.data[self.label_no_noises== i+1,1], s=8, c=color)
#                plt.scatter(self.data[self.cluster_center[i],0], self.data[self.cluster_center[i],1], s=64, c='k', marker='*')
#                # plt.text(data[cluster_center[i],0], data[cluster_center[i],1], str(i+1), size = 16)
#        else:
#            for i,color in zip(range(int(max(self.cluster_vector))),color_iter):
#                plt.scatter(self.data[self.cluster_vector== i+1,0], self.data[self.cluster_vector== i+1,1], s=8, c=color)
#                plt.scatter(self.data[self.cluster_center[i],0], self.data[self.cluster_center[i],1], s=64, c='k', marker='*')
#                # plt.text(data[cluster_center[i],0], data[cluster_center[i],1], str(i+1), size = 16)
#        plt.show()
#        
#    def draw_decision(self):
#        plt.figure()
#        color_iter = itertools.cycle(['r','b','m','y','c','g','gold','lightskyblue','darkgrey','peru','orange','coral','darkred','yellowgreen','cyan','burlywood'])
#        plt.scatter(self.rho_vector, self.delta_vector,s=16, c='k')
#        for cc,color in zip(self.cluster_center,color_iter):
#            plt.scatter(self.rho_vector[cc], self.delta_vector[cc],s=32, c=color)
#        plt.show()
#
#
#def fuzzy_get_cluster_center(rho_vector, delta_vector):
#    #rho > x, delta > y
#    # cluster_center is a index_vector.[1xM],M is the number of cluster_center
#
#    mean_rho = np.mean(rho_vector)
#    std_delta = np.std(delta_vector)
#    x = mean_rho
#    y = 2*std_delta
#
#    center = []
#    rho = rho_vector > x
#    delta = delta_vector > y
#    for i in range(len(rho_vector)):
#        if rho[i] and delta[i]:
#            center.append(i)
#    return center
#
#def can_merge(distance_matrix, cluster_vector, rho_vector,dc, c1=1, c2=2):
#    flag = False
#    c1idx = np.where(cluster_vector==c1)[0]
#    c2idx = np.where(cluster_vector==c2)[0]
#    temp = distance_matrix[c1idx,:][:,c2idx]
#    ans_x, ans_y = np.where(temp<dc)
##    plt.plot(data[:,0],data[:,1],'.k')
##    plt.plot(data[c1idx[ans_x],0],data[c1idx[ans_x],1],'.r')
##    plt.plot(data[c2idx[ans_y],0],data[c2idx[ans_y],1],'.g')
#    boundary = list(set(c1idx[ans_x])) + list(set(c2idx[ans_y]))
#    
#    boundary_rho = np.mean(rho_vector[boundary])
#    rhoc1 = np.mean(rho_vector[c1idx])
#    rhoc2 = np.mean(rho_vector[c2idx])
#    if boundary_rho > (rhoc1/2+rhoc2/2):
#        flag = True
#    return flag
#def f(a):
#    temp = []
#    for i in a:
#        temp.append(set(i))
#    for i in range(len(temp)):
#        for j in range(i+1,len(temp)):
#            if temp[i]&temp[j]:
#                temp[i] = temp[i] | temp[j]
#                
#    for i in range(len(temp)):
#        if temp[i]==0:
#            continue
#        for j in range(i+1,len(temp)):
#            if temp[j]!=0 and (temp[i]&temp[j]):
#                temp[j]=0
#    ans = [list(i) for i in temp if i!=0]
#    return ans
#
#
#def merge_cluster(cluster_vector, ans_recode):
#    for recode in ans_recode:
#        for i in range(1,len(recode)):
#            cluster_vector[cluster_vector==recode[i]] = recode[0]
#    return cluster_vector
#
#
#def merge(distance_matrix, cluster_vector, rho_vector, dc):
#    label_cluster = list(set(cluster_vector))
#    merge_recode = []
#    for i in range(len(label_cluster)):
#        for j in range(i+1, len(label_cluster)):
#            if can_merge(distance_matrix, cluster_vector, rho_vector,dc, c1=i, c2=j):
#                merge_recode.append((i,j))
#    ans_recode = f(np.array(merge_recode))
#    
#    cluster_vector = merge_cluster(cluster_vector, ans_recode)
#    
#    labels_set = set(cluster_vector)
#    if 0 in labels_set:
#        labels_set.remove(0)
#
#    for i,j in enumerate(labels_set):
#        cluster_vector[cluster_vector==j]=i+1
#
#    cluster_center = get_cluster_center_By_vector(cluster_vector, rho_vector)    
#    
#    return cluster_vector, cluster_center
#
#
#class AdaptiveFuzzyCluster(object):
#    def __init__(self, dc_precent=0.02,dc=0):
#        self.dc_precent = dc_precent
#        self.dc = dc
#
#    def fit(self,data):
#        self.data = data
#        self.distance_array = distance.pdist(self.data, 'euclidean')
#        self.distance_matrix = distance.squareform(self.distance_array)
#        self.dc = _get_dc(self.distance_array, self.dc_precent)
#        self.rho_vector = _get_rho_vector(self.distance_matrix, self.dc)
#        self.delta_vector = _get_delta_vector(self.distance_matrix, self.rho_vector)
#        self.cluster_center = fuzzy_get_cluster_center(self.rho_vector, self.delta_vector)
#        self.cluster_vector = _science_get_cluster_vector(self.distance_matrix, self.rho_vector, self.cluster_center)
#        self.cluster_vector, self.cluster_center = merge(self.distance_matrix, self.cluster_vector, self.rho_vector, self.dc)
#        self.cluster_vector = np.array(self.cluster_vector, dtype=int)
#        return self
#    
#    def draw_cluster(self):
#        plt.figure()
#        color_iter = itertools.cycle(['r','b','m','y','c','g','gold','lightskyblue','darkgrey','peru','orange','coral','darkred','yellowgreen','cyan','burlywood'])
#        plt.scatter(self.data[:,0], self.data[:,1], s=8, c='k')
#
#        for i,color in zip(range(int(max(self.cluster_vector))),color_iter):
#            plt.scatter(self.data[self.cluster_vector== i+1,0], self.data[self.cluster_vector== i+1,1], s=8, c=color)
#            plt.scatter(self.data[self.cluster_center[i],0], self.data[self.cluster_center[i],1], s=64, c='k', marker='*')
#            # plt.text(data[cluster_center[i],0], data[cluster_center[i],1], str(i+1), size = 16)
#        plt.show()
#        
#    def draw_decision(self):
#        plt.figure()
#        color_iter = itertools.cycle(['r','b','m','y','c','g','gold','lightskyblue','darkgrey','peru','orange','coral','darkred','yellowgreen','cyan','burlywood'])
#        plt.scatter(self.rho_vector, self.delta_vector,s=16, c='k')
#        for cc,color in zip(self.cluster_center,color_iter):
#            plt.scatter(self.rho_vector[cc], self.delta_vector[cc],s=32, c=color)
#        x = np.mean(self.rho_vector)
#        y = 2*np.std(self.delta_vector)
#        plt.plot([x,x],[0,max(self.delta_vector)],'k')
#        plt.plot([0,max(self.rho_vector)],[y,y],'k')
#        plt.show()