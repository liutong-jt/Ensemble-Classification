import random
from Classify import Classifier, get_shuffled_data
import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import concurrent.futures as cf
from sklearn import preprocessing
from utils.tools import load_data, Vimps, GraphVimps
from tqdm import tqdm 
import warnings 
import sys
warnings.filterwarnings('ignore')
'''
    > 1. 计算基因的重要性，没有交叉验证
'''

def train_and_trainscore(X, y):
    # ans 将迭代num_iter次的结果都存储起来，到结束后一起保存，节省计算机读写文件时间。
    # ans = np.zeros([num_iter, X.shape[1]+1])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    classifier = Classifier()
    classifier = classifier.fit(x_train, y_train)
    _, _, best_score = classifier.find_best_ensembles(x_test, y_test)

    # # 初始化每个特征的重要性， vimps[:-1] 记录每个特征重要性，vimps[-1]记录分类器的种类
    vimps = np.zeros((x_test.shape[1]))
    if best_score < 0.75:
        return 0, vimps


    # # 遍历所有特征(这里是108维)，计算
    for n_dim in range(x_test.shape[1]):
        # 打乱一维特征，再使用最好的分类器来预测一下结果
        shuffle_one_feature_data = get_shuffled_data(x_test, n_dim)
        score_shuffled = classifier.get_best_score(shuffle_one_feature_data, y_test, vote=False, accuracy=False)
        # 定义：该维特征的重要性 = 未打乱这个维度之前的准确性 - 打乱这个维度之后的准确性
        one_feature_imptns = best_score - score_shuffled
        # if one_feature_imptns < 0:
        #     one_feature_imptns = 0 
        #     print(n_dim)
        # 保存
        vimps[n_dim] = one_feature_imptns
    # # # vimps[-1] = idx
    return best_score, vimps

def extract_feature_for_acc(X, y, n_features, times, save_dir):
    # get n features from X randomly.
    sample = range(X.shape[1])
    vimps_ans = Vimps(X.shape[1], save_dir)
    # graph_vimps = GraphVimps(X.shape[1], save_dir)
    feature_dict = {}
    feature_count = {}

    for _ in range(times):
        extract_features = random.sample(sample, n_features)
        # print(extract_features)
        # extract_features = np.array(extract_features)
        # # get those n features's score. NOT SHUFFLE
        best_score, vimps = train_and_trainscore(X[:,extract_features], y)
        # vimps = train_and_trainscore(X, y)
        # feature score. SHUFFLE.
        # for i, vimps in zip(extract_features, vimps):
        #     if i in feature_dict:
        #         feature_dict[i] += vimps
        #     else:
        #         feature_dict[i] = vimps
        # feature_dict = {i:vimp for i, vimp in zip(extract_features, vimps)}

        for i, vimp in zip(extract_features, vimps):
            if i in feature_dict:
                feature_dict[i] += vimp
                feature_count[i] += 1
            else:
                feature_dict[i] = vimp
                feature_count[i] = 1

    vimps_ans.update(feature_dict, feature_count)
    # if best_score != 0:
    # graph_vimps.update(feature_dict, best_score)
    vimps_ans.save_vimps()
    # graph_vimps.save_graph()
    print('End:', time.ctime())


if __name__ == '__main__':

    # name = sys.argv[1]
    # WORK_PATH = r'D:\BBD_DATA\TCGA_DATA'
    # Data_path = os.path.join(WORK_PATH, f'TCGA-{name}/exp.csv')
    # Label_path = os.path.join(WORK_PATH, f'TCGA-{name}/label.npy')
    # graph_save_dir = os.path.join(WORK_PATH, f'DEGs/DEGs_{name}/vimps_for_all_RNA_5feature')

    name = 'KIRC-lncmmiRNA'
    WORK_PATH = r'.'
    Data_path = os.path.join(WORK_PATH, f'Data/TCGA-{name}/exp.csv')
    Label_path = os.path.join(WORK_PATH, f'Data/TCGA-{name}/label.npy')
    graph_save_dir = os.path.join(WORK_PATH, f'Data/DEGs/DEGs_{name}/vimps_for_all_RNA_5feature')

    print('Start!')
    print(time.ctime())

    # # Load data
    # # data, label = load_data(r'C:\Users\BBD\Documents\Importance\Data\TCGA-LIHC\mi_RNAdata.npy')
    # # print(data.shape)

    data = pd.read_csv(Data_path).iloc[:, 1:].T.values
    label = np.load(Label_path)
    # data = np.load(r'Data\Semulate\semulate_data.npy')
    # label = np.load(r'Data\Semulate\semulate_label.npy')
    # graph_save_dir = os.path.join(r'Data\Semulate\vimps_for_all_RNA_2feature')


    # import matplotlib.pyplot as plt
    # for i in range(35, 39):
    #     for j in range(i+1, 40):
    #         plt.plot(data[label==0, i], data[label==0, j],'.')
    #         plt.plot(data[label==1, i], data[label==1, j],'.')
    #         plt.title(f'{i} - {j}')
    #         plt.show()


    # not_zero_index = df.iloc[:, 2:].std(axis=1) != 0
    # t = df.loc[not_zero_index,:]
    # t.to_csv(r'D:\BBD_DATA\TCGA_DATA\TCGA-LIHC\exp_processed1.csv')

    # x_train, x_test, y_train, y_test = train_test_split(data, label, stratify=label, test_size=0.5, random_state=1)
    
    # data = x_train
    # label = y_train

    print(data.shape)
    # std = np.std(data, axis=0)
    # idx = np.where(std != 0)[0]
    # print(idx)
    # data = data[:,idx]

    # data = scipy.stats.zscore(data)
    # Training set and Testing Set

    # tims 1000 , n_job 16 : 39s
    # tims 1000 , n_job 16 : 4min24s

    if not os.path.exists(graph_save_dir):
        os.makedirs(graph_save_dir)

    works = 100
    # times_for_a_work = int( 10000 /works )
    n_feature = 5
    times_for_a_work = int(data.shape[1] * (100 / n_feature ) / works)
    # vimps = Vimps(num_of_features=data.shape[1])
    # result = extract_feature_for_acc(data, label, n_feature)
    # vimps.update(result)
    # vimps.save_vimps()
    # plt.plot(vimps.avg_vimps())
    # plt.show()
###########################################################################################
    # if os.cpu_count() == 16:
    #     n_jobs = 10
    # elif os.cpu_count() == 12:
    #     n_jobs = 10
    # else: 
    #     n_jobs= int(os.cpu_count() * 0.8 // 2 * 2)
    
    jobs = []

    # extract_feature_for_acc(x_train, y_train, n_feature, times_for_a_work, graph_save_dir)

    print('*' * 20)
    with cf.ProcessPoolExecutor(max_workers=12) as pool:
        for i in range(works):
            jobs.append(pool.submit(extract_feature_for_acc, data, label, n_feature, times_for_a_work, graph_save_dir))


    print('Compute the importances of feature completed!')
    print(time.ctime())




        # vimps.save_vimps(f'vimps_for_extract/m_lnc_miRNA_data_{ifold}_fold_vims_count.npy')
        # vimps.save_vimps(f'vimps_for_extract/tmp.vimps.npy')
    # os.system('python send_a_mail.py')
    # plt.plot(vimps.avg_vimps())
    # plt.show()
