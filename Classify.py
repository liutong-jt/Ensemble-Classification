# 计算基因重要性
import os, datetime, random, sys,copy, time
import numpy as np
import scipy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
# from clfbase import MLP_, Logist_, LDA, _SVC, KNN, DTC, GNB
from sklearn.utils import check_random_state, shuffle
from sklearn.model_selection import train_test_split
from multiprocessing import Process,Pool, cpu_count
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError

import concurrent.futures as cf
from time import sleep
import datetime

class ClassifierMixin:
    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class Classifier(ClassifierMixin):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.best_clf = None
        self.ensembles = self._build_ensemble()

    def _build_ensemble(self):
        '''
            Create a ensemble classify.
            return: List[clf]
        '''
        LDA = LinearDiscriminantAnalysis()
        KNN = KNeighborsClassifier()
        DTC = DecisionTreeClassifier()
        svc = SVC(probability=True)
        LR = LogisticRegression()
        MLP = MLPClassifier(max_iter=1000)
        MNB = GaussianNB()
        self.ensembles = [LDA, KNN, DTC, svc, MLP, LR, MNB]
        return self.ensembles

    def fit(self, x_train, y_train):
        '''
            fit every clf in ensembles
            return : List[clf(fitted)]
        '''
        self.ensembles = [clf.fit(x_train,y_train) for clf in self.ensembles]
        return self

    def find_best_ensembles(self, x_test, y_test, accuracy=False):
        '''
            训练所有的分类器:ensemble_fitted
            使用测试集合得出每个分类器的准确性
            param: 
                accuracy: True/False: 根据什么来判断是否是最好的
                    True: 准确率， False: ErrorOOB
                # self.ensembles: 集成分类器（训练好的）
                 x_test, y_test: 训练集和测试集
            reutrn:
                idx, best_clf, best_score
                准确性最高的分类器的索引，训练好的分类器，最好分类器的准确性
        '''
        if accuracy  == True:
            scores = [clf.score(x_test, y_test) for clf in self.ensembles]
        else:
            scores = [0 for _ in self.ensembles]
            for i, clf in enumerate(self.ensembles):
                try:
                    y_pred = clf.predict(x_test)
                    scores[i] = 1 - self.ErrOOB_score(y_true=y_test, y_pred=y_pred)
                except NotFittedError as e:
                    scores[i] = 0
            # scores = [1 - self.ErrOOB_score(y_true=y_test, y_pred=clf.predict(x_test)) for clf in self.ensembles]

        idx = np.argsort(scores)[-1]
        self.best_clf =self.ensembles[idx]
        self.best_score = scores[idx]
        return idx, self.best_clf, self.best_score

    def _predict_by_voted(self, x_test):
        '''
            仅适用二分类情况(0,1)
            投票过半数计为1, 否则计为0
        '''
        y_preds = [clf.predict(x_test) for clf in self.ensembles]
        y_preds = np.array(y_preds).reshape(len(y_preds), -1)
        y_pred = np.mean(y_preds, axis=0)> 0.5
        return y_pred

    def _predict_by_best_clf(self, x_test):
        '''
        直接调用分类器自带的predict方法
        '''
        y_pred = self.best_clf.predict(x_test)
        return y_pred

    def predict(self, x_test, vote=False):
        '''
        两种预测方法
        一: 使用投票策略，使用所有的分类器进行预测，将所有的lable进行投票，投票结果为最终结果
        二:使用最好的分类器，进行预测，预测结果为最终结果
        '''
        if vote == True:
            y_pred =  self._predict_by_voted(x_test)
        else:
            if self.best_clf == None:
                raise ValueError('The best_clf is not defined!\n U Should execute the function named "find_best_ensembles()"!')
            y_pred = self._predict_by_best_clf(x_test)
        return  y_pred

    def predict_proba(self, x_test):
        '''
            使用单个最好的分类器，来预测数据属于某一类别的概率
            其结果可以用来画ROC曲线
        '''
        if self.best_clf == None:
            raise AttributeError('No best_clf')
        y_pred_proba = self.best_clf.predict_proba(x_test)
        return y_pred_proba


    def get_best_score(self, x_test, y_test, vote=False, accuracy=True):
        '''
            得到最好的score，也是有两种方式
            一:使用投票/最好分类器 来得到预测结果
            二:使用准确性(accuracy)还是文章中的ErrOOB来作为最后的结果
        '''
        y_pred = self.predict(x_test, vote=vote)
        if accuracy == True:
            self.best_score = accuracy_score(y_test, y_pred)
        else:
            self.best_score = 1 - self.ErrOOB_score(y_test, y_pred)
        return self.best_score

    def ErrOOB_score(self, y_true, y_pred):
        '''
        对于二分类
        TP:真阳--原本是正例，预测为正例
        TF:真阴--原本是反例，预测为反例
        FP:假阳--原本是反例，预测为正例
        FN:假阴--原本是正例，预测为反例
        ERROR = [FN/(TP + FN) + FP/(TN + FP)] / 2
        '''
        TP, TN, FP, FN = self.get_mixture_matrix(y_true, y_pred)
        return (FN/(TP+FN) + FP/(TN+FP))/2

    def get_mixture_matrix(self, y_true, y_pred):
        '''
        对于二分类
        TP:真阳--原本是正例，预测为正例
        TF:真阴--原本是反例，预测为反例
        FP:假阳--原本是反例，预测为正例
        FN:假阴--原本是正例，预测为反例
        '''
        T = y_pred==1
        F = y_pred!=1
        P = y_true==1
        N = y_true!=1
        TP, TN, FP, FN = np.sum(T&P), np.sum(F&N),np.sum(T&N), np.sum(F&P)
        return TP, TN, FP, FN

def Error_score(y_true, y_pred):
    T = y_pred==1
    F = y_pred!=1
    P = y_true==1
    N = y_true!=1

    # T = y_pred == 0
    # F = y_pred != 0
    # P = y_true == 0
    # N = y_true != 0
    TP, TN, FP, FN = np.sum(T&P), np.sum(F&N),np.sum(T&N), np.sum(F&P)
    print('TP, TN, FP, FN:', TP, TN, FP, FN)

    TP_rate = TP / (TP + FN)
    FP_rate = FP / (FP + TN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_measure = 2 * Precision * Recall / (Precision + Recall)

    print(f'TP_rate: {round(TP_rate, 3)}')
    print(f'FP_rate: {round(FP_rate, 3)}')
    print(f'Precision: {round(Precision, 3)}')
    print(f'Recall: {round(Recall, 3)}')
    print(f'F1-measure: {round(F1_measure, 3)}')

    return 1 - (FN/(TP+FN) + FP/(TN+FP))/2


def get_shuffled_data(data, dim, random_state=None):
    '''
    返回 打乱后的data
    '''
    myrandom = check_random_state(random_state)
    index = np.arange(data.shape[0])
    myrandom.shuffle(index)
    t_data = data.copy()
    t_data[:,dim] = data[:, dim][index]
    return t_data

def cal_imptns(ensemble_clf, x_test, y_test):
    '''
        计算每个特征的重要性
        param: 
            best_clf: 集成分类器中，最好的那个分类器
            x_test, y_test: 测试集
            best_score: 集成分类器中，最准确的分类器对应的准确率
        return:
            vimps: 特征重要性
    '''
    # 计算最好的分数
    idx, best_clf, best_score= ensemble_clf.find_best_ensembles(x_test, y_test, accuracy=False)

    #TODO: 这个可以使用多进程不，会快么
    # 初始化每个特征的重要性， vimps[:108] 记录每个特征重要性，vimps[-1]记录分类器的种类
    vimps = np.zeros((x_test.shape[1]+1,))
    # 遍历所有特征(这里是108维)，计算
    for n_dim in range(x_test.shape[1]):
        # 打乱一维特征，再使用最好的分类器来预测一下结果
        shuffle_one_feature_data = get_shuffled_data(x_test, n_dim)
        score_shuffled = ensemble_clf.get_best_score(shuffle_one_feature_data, y_test, vote=False, accuracy=False)
        # 定义:该维特征的重要性 = 未打乱这个维度之前的准确性 - 打乱这个维度之后的准确性
        one_feature_imptns = best_score - score_shuffled
        # 保存
        vimps[n_dim] = one_feature_imptns
    vimps[-1] = idx
    return vimps

def main(X, y, start, end):
    num_iter = end - start
    # ans 将迭代num_iter次的结果都存储起来，到结束后一起保存，节省计算机读写文件时间。
    ans = np.zeros([num_iter, X.shape[1]+1])
    print(str(start) + '__iter__' + str(end) + '\t', datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
    
    for i in range(start, end):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)
        classifier = Classifier()
        classifier = classifier.fit(x_train, y_train)
        vimps = cal_imptns(classifier, x_test, y_test)

        ans[i%num_iter] = vimps

    ans_save_name = 'score_' + str(start) + 'to'+ str(end)
    ans_save_name = os.path.join('save_vimps', ans_save_name)
    np.save(ans_save_name, ans)
    print(str(start) + '__iter__' + str(end) + '\tCompleted!\t', datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))


if __name__ == '__main__':
    data = np.load(r'C:\Users\BBD\Desktop\特征提取\植物五肽\origional.removeFirstM.npy')
    label = data[:,-1]
    label = np.where(label==1, 1, 0)
    data = data[:,:-1]
    data = scipy.stats.zscore(data)
    # Training set and Testing Set
    x_train, independed_x, y_train, independed_y = train_test_split(data,label,test_size=0.5, random_state=1)

    save_dir = 'save_vimps'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    # main(x_train, y_train, 0, 100)
    # clfs = build_clfs(x_train, y_train, 10000)

    # print(len(clfs))

    # x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_train, y_train, test_size=0.5)
    # classify = Classifier()
    # classify = classify.fit(x_train_train, y_train_train)
    # vimps = classify.cal_imptns(x_train_test, y_train_test)
    # print(vimps)

    n_jobs, start, end = 10, 0, 10000
    # n_jobs = sys.argv[1]
    # start = eval(sys.argv[2])
    # end = eval(sys.argv[3])

    if n_jobs == '-1' or n_jobs==None or n_jobs=='':
        n_jobs = cpu_count()
    elif int(n_jobs) >= cpu_count():
        n_jobs = cpu_count
    else:
        n_jobs = int(n_jobs)

    avg = (end - start) // n_jobs
    n_iter = list(range(start, end+avg, avg))
    # print(n_iter)
    
    for i in range(n_jobs):
        print(n_iter[i], n_iter[i] + avg)


    jobs = []
    with cf.ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for i in range(n_jobs):
            jobs.append(pool.submit(main, x_train, y_train, n_iter[i], n_iter[i] + avg))

    while len(jobs) != 0:
        for job in jobs:
            if job.done():
                # vimps.update(job.result())
                jobs.remove(job)
        sleep(10)

    # pool = Pool(n_jobs)
    # n_jobs = []
    # for n_iter_start in n_iter[:-1]:
    #     pool.apply_async(main,args=(x_train, y_train, n_iter_start, n_iter_start+avg))
    # pool.close()
    # pool.join()

    # for job in n_jobs:
    #     job.close()

    # for job in n_jobs:
    #     job.join()