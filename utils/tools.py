import numpy as np
import matplotlib.pyplot as plt
import os, time, math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from Classify import Classifier
from scipy import cluster


def load_data(data_path):
    if os.path.splitext(data_path)[-1] == '.npy':
        data = np.load(data_path)
        label = data[:,-1]
        label = np.where(label==1, 1, 0)
        data = data[:,:-1]

    elif os.path.splitext(data_path)[-1] == '.csv':
        data = pd.read_csv(data_path)
        label = np.where(data['class'] == 'p', 1, 0)
        data = data.iloc[:,:-1].values
    else:
        raise TypeError
    return data, label

def load_feature_impts_from_dir(file_dir= r'save_vimps', show=False):
    '''
        导入基因重要性特征
    '''
    file_list = os.listdir(file_dir)
    for i, file in enumerate(file_list[:]):
        file_path =  os.path.join(file_dir, file)
        if i == 0:
            data = np.load(file_path)
        else:
            tmp = np.load(file_path)
            if np.sum(tmp[:,0] != 0) == 0:
                print(file)
                pass
            else:
                data += tmp
            # data += np.load(file_path)

    vimps = data[:,0] / data[:,1]
    if show == True:
        plt.plot(np.arange(data.shape[0]), data[:,1], '.k')
        plt.xlabel('Feature Index')
        plt.ylabel('Checked feature Count')
        plt.title('Feature Count')
        plt.show()


        plt.plot(np.arange(data.shape[0]), vimps, '.k')
        sorted_index = np.argsort(-vimps)
        for i in range(0):
            plt.text(sorted_index[i], vimps[sorted_index[i]], str(sorted_index[i]))
        plt.title('Gene Importance')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Score')
        # plt.ylim([0,0.25])
        plt.show()
        # print(sorted_index[:20])
    return vimps

def load_graph_impts_from_dir(file_dir=r'save_vimps', show=True):
    '''
        导入基因重要性特征
    '''
    file_list = os.listdir(file_dir)
    for i, file in enumerate(file_list[:]):
        file_path =  os.path.join(file_dir, file)
        if i == 0:
            data = np.load(file_path)
        else:
            tmp = np.load(file_path)
            if np.sum(tmp[:,0] !=0) == 0:
                pass
            else:
                data += tmp
            # data += np.load(file_path)
        # if np.max(data[0]) > 100:
        #     print(i, file)

    # plt.hist(data[:,1])
    # plt.show()
    # 注意 除数 为 0 的情况

    index = np.where(data[1]==0)
    data[1, index[0], index[1]] = 1

    vimps = np.diag(data[0]) / np.diag(data[1])
    graph = data[0] / data[1]
    
    # t = graph.flatten()

    # mean = np.mean(data[1])
    # for i in range(data.shape[1]):
    #     data[1, i, i] = mean
    # plt.imshow(data[1])
    # plt.show()

    if show == True:
        plt.plot(np.arange(data.shape[1]), np.diag(data[1]), '.k')
        plt.xlabel('Feature Index')
        plt.ylabel('Checked feature Count')
        plt.title('Feature Count')
        plt.show()


        plt.plot(np.arange(data.shape[1]), vimps, '.k')
        sorted_index = np.argsort(-vimps)
        for i in range(30):
            plt.text(sorted_index[i], vimps[sorted_index[i]], str(sorted_index[i]))
        plt.title(file_dir[33:])
        plt.xlabel('feature idx')
        plt.ylabel('feature score')
        # plt.ylim([0,0.25])
        plt.show()
        # print(sorted_index[:20])

        plt.imshow(graph , cmap='hot')
        plt.show()

    return vimps, graph

def load_and_show_feature_std(vimps, file_dir=r'save_svimps'):
    file_list = os.listdir(file_dir)
    for i, file in enumerate(file_list):
        file_path =  os.path.join(file_dir, file)
        if i == 0:
            t = np.load(file_path)
            data = np.zeros([len(file_list), t.shape[0]])
            data[i, :] = t[:,0] / t[:,1]
        else:
            t = np.load(file_path)
            data[i, :] = t[:,0] / t[:,1]

    std = data.std(axis=0)
    plt.plot(data.std(axis=0),'.k')
    plt.title('std')
    plt.show()

    index = np.where(vimps > 0.005)[0]
    index = np.where(std > 0.01)[0]
    plt.boxplot(data[:, index], showmeans=True)

    plt.xticks(range(1, len(index) + 1), index)
    plt.show()

    # for i in np.where(std > 0.02)[0]:
    #     plt.plot(data[:, i],'.', label=str(i))
    # plt.legend()
    # plt.plot(data[:, 14],'.k')
    # plt.show()
    return std

def get_best_base_clfs(X, y):
    '''
        将输入数据X,y， 按照7：3分为训练集和测试集，使用训练集来训练所有的分类器，使用测试集找出得分最高的分类器
        input: X, y
        return best_clf
    '''
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=None)
    classifier = Classifier()
    classifier = classifier.fit(x_train, y_train)
    classifier.find_best_ensembles(x_test, y_test, accuracy=False)
    return classifier.best_clf

class Vimps:
    '''
        计算特征重要性时，用来记录每次抽取的特征重要性

        一个简单的数据结构
    '''
    def __init__(self,num_of_features, save_dir=None):
        self.vimps = np.zeros(num_of_features,)
        self.count_feature = np.zeros(num_of_features,)
        self.save_dir = save_dir
        self.__check()

    def __check(self):
        if self.save_dir != None and not os.path.isdir(self.save_dir):
            raise f'This dir {self.save_dir} is not exist.'

    def update(self, n_feature_vimps, feature_count=None):
        if isinstance(n_feature_vimps, dict) and isinstance(feature_count, dict):
            for key, value in n_feature_vimps.items():
                self.vimps[key] += value
                self.count_feature[key] += feature_count[key]

        elif isinstance(n_feature_vimps, np.ndarray) and isinstance(feature_count, np.ndarray):
            self.vimps += n_feature_vimps
            self.count_feature += feature_count


    def get_avg_vimps(self):
        not_nan_idx = np.where(self.count_feature != 0 )
        return self.vimps[not_nan_idx] / self.count_feature[not_nan_idx]

    def save_vimps(self, save_path=None):
        if self.save_dir!=None and  save_path == None:
            save_path = os.path.join(self.save_dir, f'vimps_and_count_{time.time()}.npy')
        elif self.save_dir == None and save_path == None :
            save_path = f'vimps_and_count_{time.time()}.npy'
        elif self.save_dir != None and save_path != None:
            save_path = os.path.join(self.save_dir, save_path)
        np.save(save_path, np.hstack([self.vimps.reshape(-1,1),
                                        self.count_feature.reshape(-1,1)]))

class GraphVimps:
    '''
        The difference of Vimps is that we don't add vimps which minus zeros .
    '''
    def __init__(self, num_features, save_dir) -> None:
        self.num_features = num_features
        self.save_dir = save_dir
        self.linkage_matrix = np.zeros([num_features, num_features])
        self.count = np.zeros_like(self.linkage_matrix)

    def update(self, n_feature_vimps, best_score):
        for i in n_feature_vimps.keys():
            for j in n_feature_vimps.keys():
                self.count[i,j] += 1

        # if best_score != 0:
        keys, values = np.array(list(n_feature_vimps.keys())), np.array(list(n_feature_vimps.values()))
        sum_permutation = np.sum(np.abs(values))
        # 不要小于等于0 的基因重要性
        has_value_gene = np.where(values != 0)[0]

        if np.max(self.linkage_matrix) > 100:
            print('Warning')

        for i in has_value_gene:
            for j in has_value_gene:
                if i != j:
                    self.linkage_matrix[keys[i], keys[j]] += (values[i] / sum_permutation) * best_score
                else:
                    self.linkage_matrix[keys[i], keys[j]] += values[i]


    def save_graph(self, save_path=None):
        if self.save_dir!=None and  save_path == None:
            save_path = os.path.join(self.save_dir, f'graphvimps_and_count_{time.time()}.npy')
        elif self.save_dir == None and save_path == None :
            save_path = f'graphvimps_and_count_{time.time()}.npy'
        elif self.save_dir != None and save_path != None:
            save_path = os.path.join(self.save_dir, save_path)

        np.save(save_path, np.stack([self.linkage_matrix, self.count], axis=0))

def split_gaussian(Weight1, Weight2, Miu1, Miu2, delta1, delta2):
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

def showfigure(weights,means,covars, data_vector,gaussian_boundary):
    x_max = max(data_vector)
    x_min = min(data_vector)
    for i, (w,m,c) in enumerate(zip(weights, means, covars)):
        x = np.linspace(min(data_vector), max(data_vector), 1000)
        # else:
        #     x = np.arange(0.004, 0.1, 0.0001)
        y=[]
        for j in x:
            y.append(w*_gaussian(j,m,c))

        plt.plot(x,y,'--r')
    # plt.hist(data_vector,bins=50,color='r',alpha=0.4)
    for x in gaussian_boundary:
        plt.plot([x,x],[0,5], 'b')
    plt.plot(data_vector, np.zeros(data_vector.shape),'.k', markersize=8)
    plt.ylabel('Density')
    # plt.xlabel('Feature Index')
    # plt.xlabel('Accumulated importance')
    # plt.tick_params(labelsize = 18)
    plt.show()

def _gaussian(x,mean,covar):
    left=1/(math.sqrt(2*math.pi)*covar)
    right=math.exp(-math.pow(x-mean,2)/(2*math.pow(covar,2)))
    return left*right

def get_gaussian_boundary(data, n_components=5, show=False):
    X = data.reshape(-1,1)
    lowest_bic = np.infty
    bic = []
    for n_component in range(1, n_components + 1):
    #    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='spherical').fit(X)
        gmm = GaussianMixture(n_components=n_component, covariance_type='spherical', random_state=1).fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
    
    means = best_gmm.means_.flatten()
    covars = best_gmm.covariances_.flatten()
    weights = best_gmm.weights_.flatten()
    x = []
    if len(means)==1:
        x.append(means[0] + covars[0] ** 0.5 * 2)
        return x

    index = np.argsort(means)
    for i in range(len(index)-1):
        x.append(split_gaussian(weights[index[i]], weights[index[i+1]], means[index[i]],means[index[i+1]],covars[index[i]]**0.5,covars[index[i+1]]**0.5))

    if show == True:
        showfigure(weights,means,covars**0.5, data, sorted(x))
    return sorted(x)

def draw_hierarchy_cluster(data, show=True):
    Z = cluster.hierarchy.linkage(data, method='complete',optimal_ordering=True)
    if show == True:
        cluster.hierarchy.dendrogram(Z, no_labels=False)
        plt.show()
    index = cluster.hierarchy.cut_tree(Z, n_clusters=[2])
    return index

class TransData:
    def __init__(self, A_Data, B_Data):
        self.A_Data = A_Data
        self.B_Data = B_Data

    def trans2zero(self):
        '''
            1. 减均值
            2. 线性变换2线性无关
            3. 横纵拉伸, 
        '''
        print(self.A_Data.shape)
        data = self.A_Data - np.mean(self.A_Data, axis=1).reshape(-1,1)

        cov = np.cov(data)
        A_evalue, A_evector = np.linalg.eig(cov)

        # print(evector)
        # if np.sum(A_evector > 0) >= len(A_evalue) * (len(A_evalue) - 1) // 2 :
        #     if A_evector[0, 0] < 0 :
        #         A_evector = -A_evector
        # print(evector)

        data = np.dot(A_evector.T, data)
        data = data / np.std(data, axis=1).reshape(-1,1)
        # plt.plot(data[0,:], data[1,:], '.k')
        # plt.show()
        self.A_zero = data
        # return data


        '''
            1. 求tcga数据的均值，方差，相关性
            2. 对data进行横纵拉伸到tcga数据相同(更改方差)
            3. 对data进行旋转，将其变换成tcga数据相同的线性相关
            4. 对data进行平移，平移到tcga数据集的位置（均值，最低点）
        '''
        # print(self.A_zero.shape, self.B_Data.shape)

        B_zeros = self.B_Data - np.mean(self.B_Data, axis=1).reshape(-1, 1)
        B_cov = np.cov(B_zeros)
        B_evalue, B_evector = np.linalg.eig(B_cov)
        # if np.sum(B_evector > 0) >=  len(B_evalue) * (len(B_evalue) - 1) // 2  :
        #     if B_evector[0, 0] < 0 :
        #         B_evector = -B_evector


        # flag_of_Bevalue = np.array([1 if i > 0  else -1 for i in B_evalue ]).reshape(-1,1)
           
        # self.B_like1 = self.A_zero * np.abs(B_evalue).reshape(-1, 1) * flag_of_Bevalue
        # # B_evector[1,:] = - B_evector[1, :]
        # self.B_like = np.dot(B_evector.T, self.B_like1)

        # B_evector[3,:] = - B_evector[3, :]
        # self.B_like_V2 = np.dot(B_evector.T, self.B_like1)
        # self.B_like_V2 = self.B_like_V2 + np.mean(self.B_Data, axis=1).reshape(-1, 1)

        # for i in range(len(B_evalue)):
        #     if np.mean(self.B_like[i, label==0]) > np.mean(self.B_like[i, label==1]):
        #         if np.mean(self.A_Data[i, label==0]) < np.mean(self.A_Data[i, label==1]):
        #             B_evector[i,:] = - B_evector[i, :]
        #     elif np.mean(self.B_like[i, label==0]) < np.mean(self.B_like[i, label==1]):
        #         if np.mean(self.A_Data[i, label==0]) > np.mean(self.A_Data[i, label==1]):
        #             B_evector[i,:] = - B_evector[i, :]

        self.B_like = self.A_zero * np.sqrt(np.abs(B_evalue).reshape(-1, 1) ) # * flag_of_Bevalue
        self.B_like = np.dot(B_evector, self.B_like)
        self.B_like = self.B_like + np.mean(self.B_Data, axis=1).reshape(-1, 1)

        # print(np.mean(self.B_like, axis=1))
        # self.B_like_V2 = np.dot(-B_evector, self.B_like) 
        # self.B_like_V2 = self.B_like_V2 + np.mean(self.B_Data, axis=1).reshape(-1, 1)
        # print(np.mean(self.B_like, axis=1))


    def trans_data(self, cor=False):   
        if cor == False:
            self.B_like = (self.A_Data - np.mean(self.A_Data, axis=1).reshape(-1,1)) / np.std(self.A_Data, axis=1).reshape(-1,1) * np.std(self.B_Data, axis=1).reshape(-1,1) + np.mean(self.B_Data, axis=1).reshape(-1,1)
            # self.B_like = (self.A_Data - np.min(self.A_Data, axis=1).reshape(-1,1)) / np.max(self.A_Data, axis=1).reshape(-1,1) * np.max(self.B_Data, axis=1).reshape(-1,1) + np.min(self.B_Data, axis=1).reshape(-1,1)
            return self.B_like.T
        else:
            if len(self.A_Data.shape)  > 1 and self.A_Data.shape[0] != 1:
                self.trans2zero()
                # self.trans2tcga()
                return self.B_like.T
            elif len(self.A_Data.shape) == 1 or self.A_Data.shape[0] == 1:
                self.B_like = (self.A_Data - np.mean(self.A_Data) ) / np.std(self.A_Data) * np.std(self.B_Data) + np.mean(self.B_Data) 
                return self.B_like.reshape(-1,1)

def trans2zero(data):
    '''
        1. 减均值
        2. 线性变换2线性无关
        3. 横纵拉伸, 
    '''
    data = data - np.mean(data, axis=1).reshape(-1,1)
    cov = np.cov(data)
    evalue, evector = np.linalg.eig(cov)

    # print(evector)
    # if np.sum(evector > 0) >= len(evalue) * (len(evalue) + 1) // 2 :
    #     if evector[0, 0] < 0 :
    #         evector = -evector
    print(evector)

    data = np.dot(evector.T, data)
    data = data / np.std(data, axis=1).reshape(-1,1)
    # plt.plot(data[0,:], data[1,:], '.k')
    # plt.show()
    return data

def trans2tcga(data, A):
    '''
        1. 求tcga数据的均值，方差，相关性
        2. 对data进行横纵拉伸到tcga数据相同(更改方差)
        3. 对data进行旋转，将其变换成tcga数据相同的线性相关
        4. 对data进行平移，平移到tcga数据集的位置（均值，最低点）
    '''
    A_zeros = A - np.mean(A, axis=1).reshape(-1, 1)
    A_cov = np.cov(A_zeros)
    evalue, evector = np.linalg.eig(A_cov)
    # print(evector)
    # if np.sum(evector > 0) >= len(evalue) * (len(evalue) + 1) // 2 :
    #     if evector[0, 0] < 0 :
    #         evector = -evector
    print(evector)

    data = data * np.sqrt(evalue.reshape(-1, 1) )
    data = np.dot(evector, data)
    # print(np.mean(data, axis=1))
    data = data + np.mean(A, axis=1).reshape(-1, 1)
#     data = data - np.mean(data, axis=1).reshape(2, -1) + np.min(A, axis=1).reshape(2, -1)
#     plt.plot(data[0,:], data[1,:], '.k')
#     plt.show()
    return data

def trans_data(A, B):
    '''
        A.shape === (n_feature, n_sample_A)
        B.shape === (n_feature, n_sample_B)
        return :
            B_like.shape === (n_sample, n_feature)
    '''
    A_zero = trans2zero(A)
    B_like = trans2tcga(A_zero, B).T
    return B_like


def get_TCGA_Gene_Info(Ens_Table_Path='gProfiler_hsapiens_2021-12-20 下午1-40-11.csv'):
    ensemble_table = pd.read_csv(Ens_Table_Path)
    # # ensemble_table.set_index('transcript_id',  inplace=True)
    # # ensemble.dropna(axis=0, how='any', inplace=True)
    # # ensemble.drop_duplicates(subset=['transcript_id','transcript_name'] ,keep='first', inplace=True)
    # # ensemble.to_csv(r'D:\BBD_DATA\LUADData\ensemble_2.csv')
    # Gene_Index =[]
    # with open(Gene_TXT, 'r') as f:
    #     for line in f.readlines():
    #         ensemble = re.split('\s+', line)[1]
    #         Gene_Index.append(re.split('\s+', line)[0])
    #         Gene_list.append(ensemble.split('.')[0])

    Gene_dict = {}
    for i in ensemble_table['name']:
        print(f'"{i}"', end=', ')

    print('\n'* 3)

    for ens, index in zip(ensemble_table['name'], ensemble_table['converted_alias']):
        if ens == 'nan':pass
        Gene_dict[ens] = index
        print(f'"{ens}":{index}', end=', ')

    return Gene_dict #, list(Gene_dict.keys())

    # if __name__ == '__main__':
    #     gene_ens_table = r'D:\BBD_DATA\TCGA_DATA\DEGs\DEGs_STAD\gProfiler_hsapiens_2021-12-20 下午1-40-11.csv'
    #     Gene_dict, Gene_list = get_TCGA_Gene_Info(gene_ens_table)
    #     print(Gene_list, Gene_dict)

if __name__ == '__main__':
    data_path = r'Data\TCGA-LIHC\exp_processed.csv'
    data = pd.read_csv(data_path)
    print(data.shape)
    not_zero_index = data.iloc[:, 2:].std(axis=1) != 0
    t = data.loc[not_zero_index,:]

    # label = np.zeros(611, dtype=np.int8)
    # for i, col in enumerate(t.columns[2:]):
    #     if int(col[13:15]) >= 10:
    #         label[i] = 1

    # np.save('Data\TCGA-KIRC\label.npy', label)
    
    t = t.iloc[:, 1:]
    t.to_csv(r'Data\TCGA-KIRC\exp_processed.csv')
