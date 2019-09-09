import time, threading

# 假定这是你的银行存款:
balance = 0
lock = threading.Lock()
def change_it(n):
    # 先存后取，结果应该为0:
    global balance
    balance = balance + n
    balance = balance - n

def run_thread(n):
    for i in range(100000):
        lock.acquire()
        change_it(n)
        lock.release()

t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)

# from multiprocessing import Process, Queue
# import os, time, random

# # 写数据进程执行的代码:
# def write(q):
#     print('Process to write: %s' % os.getpid())
#     for value in ['A', 'B', 'C']:
#         print('Put %s to queue...' % value)
#         q.put(value)
#         time.sleep(random.random())

# # 读数据进程执行的代码:
# def read(q):
#     print('Process to read: %s' % os.getpid())
#     while True:
#         value = q.get(True)
#         print('Get %s from queue.' % value)

# if __name__=='__main__':
#     # 父进程创建Queue，并传给各个子进程：
#     q = Queue()
#     pw = Process(target=write, args=(q,))
#     pr = Process(target=read, args=(q,))
#     # 启动子进程pw，写入:
#     pw.start()
#     # 启动子进程pr，读取:
#     pr.start()
#     # 等待pw结束:
#     pw.join()
#     # pr进程里是死循环，无法等待其结束，只能强行终止:
#     pr.terminate()

# from multiprocessing import Pool
# import os, time, random

# def long_time_task(name):
#     print('Run task %s (%s)...' % (name, os.getpid()))
#     start = time.time()
#     time.sleep(random.random() * 3)
#     end = time.time()
#     print('Task %s runs %0.2f seconds.' % (name, (end - start)))

# if __name__=='__main__':
#     print('Parent process %s.' % os.getpid())
#     p = Pool(4)
#     for i in range(5):
#         p.apply_async(long_time_task, args=(i,))
#     print('Waiting for all subprocesses done...')
#     p.close()
#     p.join()
#     print('All subprocesses done.')

C.J.L:
from multiprocessing import Process
import os, time
def run_proc(name):
    print('Run child process', name , '(', os.getpid, ')')
    time.sleep(2)

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    pro = []
    for i in range(2):
        ptemp = Process(target=run_proc, args=('test',))
        pro.append(ptemp)
    print('Child process will start.')
    for p in pro:
        p.start()
    for p in pro:
        p.join()
    print('Child process end.')


C.J.L:
'''
数据预处理
row is sample
column is feature
'''

import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
import numpy as np

csv_data = """A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,"""

if __name__=='__main__':
    df = pd.read_csv(StringIO(csv_data))
    # 缺失值统计
    print(df.isnull().sum())
    # 删除所有特征的为空的样本
    temp1 = df.dropna(how= 'all')
    # 删除少于thresh的空值样本
    temp2 = df.dropna(thresh = 4)
    # 删除第subset特征为空值的样本
    temp3 = df.dropna(subset = ['C']) 
    # 补充缺失值
    im = Imputer(missing_values= np.nan, strategy='mean', axis=0)
    im = im.fit(df)
    temp4 = im.transform(df.values)

    df = pd.DataFrame([['green', 'M', 10.1, 'class1'], ['red', 'L', 13.5, 'class2'], ['blue', 'XL', 15.3, 'class1']])

    df.columns = ['color', 'size', 'price', 'classlabel']
    class_le = LabelEncoder()
    y = class_le.fit_transform(df['classlabel'].values)
    class_le.inverse_transform(y)
    size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

    df['size'] = df['size'].map(size_mapping)
    temp = df[['price', 'color', 'size']]
    temp=pd.get_dummies(df, columns=['color'])
    ohe = OneHotEncoder(categorical_features=['color'])
    res = ohe.fit_transform(temp).toarray()
    print(df.shape)

from sklearn.preprocessing import Binarizer,OneHotEncoder,MinMaxScaler,MaxAbsScaler,StandardScaler,Normalizer
from sklearn.feature_selection import VarianceThreshold,SelectKBest,f_classif,RFE,RFECV,SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
import numpy as np
from sklearn.decomposition import DictionaryLearning

C.J.L:
#Binary
X=[   [1,2,3,4,5],
      [5,4,3,2,1],
      [3,3,3,3,3],
      [1,1,1,1,1]]
 
print("before transform:",X)
binarizer=Binarizer(threshold=2.5)
print("after transform:",binarizer.transform(X))
 
#OneHotEncoder
X=[   [1,2,3,4,5],
      [5,4,3,2,1],
      [3,3,3,3,3],
      [1,1,1,1,1]]
print("before transform:",X)
encoder=OneHotEncoder(sparse=False)
encoder.fit(X)
print("active_feature_:",encoder.active_features_)
print("feature_indices_:",encoder.feature_indices_)
print("n_values:",encoder.n_values_)
print("after transform:",encoder.transform([[1,2,3,4,5]]))

#standardization

#MinMaxScaler
X=[ [1,5,1,2,10],
     [2,6,3,2,7],
     [3,7,5,6,4],
     [4,8,7,8,1]
 ]

print("before transform:",X)
scaler=MinMaxScaler(feature_range=(0,2))
scaler.fit(X)
print("min_is:",scaler.min_)
print("scale_is:",scaler.scale_)
print("data_max_ is:",scaler.data_max_)
print("data_min_ is:",scaler.data_min_)
print("data_range_ is:",scaler.data_range_)
print("after transform:",scaler.transform(X))
#MaxAbsScaler
X=[      [1,5,1,2,10],
      [2,6,3,2,7],
      [3,7,5,6,4],
      [4,8,7,8,1]
]

print("before transform:",X)
scaler=MaxAbsScaler()
scaler.fit(X)
print("scale_is:",scaler.scale_)
print("max_abs_ is:",scaler.max_abs_)
print("after transform:",scaler.transform(X))

C.J.L:
#StandardScaler:z-score
X=[      [1,5,1,2,10],
      [2,6,3,2,7],
      [3,7,5,6,4],
      [4,8,7,8,1]
]
print("before transfrom:",X)
scaler=StandardScaler()
scaler.fit(X)
print("scale_ is:",scaler.scale_)
print("mean_ is:",scaler.mean_)
print("var_ is:",scaler.var_)
print("after transfrom:",scaler.transform(X))

#Normalizer
X=[      [1,2,3,4,5],
      [5,4,3,2,1],
      [1,3,5,2,4],
      [2,4,1,3,5]
]
print("before transform:",X)
normalizer=Normalizer(norm='l2')
print("after transform:",normalizer.transform(X))
 
#VarianceThreshold
X=[     [100,1,2,3],
     [100,4,5,6],
     [100,7,8,9],
     [101,11,12,13]
]
selector=VarianceThreshold(1)
selector.fit(X)
print("Variances is %s"%selector.variances_)
print("After transform is %s"%selector.transform(X))
print("The surport is %s"%selector.get_support(True))
print("After reverse transform is %s"%selector.inverse_transform(selector.transform(X)))

C.J.L:
#SelectKBest
X=[   [1,2,3,4,5],
      [5,4,3,2,1],
      [3,3,3,3,3],
      [1,1,1,1,1]]
Y=[0,1,0,1]
print("before transform:",X)
selector=SelectKBest(score_func=f_classif,k=3)
selector.fit(X,Y)
print("scores_:",selector.scores_)
print("pvalues_:",selector.pvalues_)
print("selected index:",selector.get_support(True))
print("after transform:",selector.transform(X))

#RFE
iris=load_iris()
X=iris.data
Y=iris.target
estimator=LinearSVC()
selector=RFE(estimator=estimator,n_features_to_select=2)
print("Before transform,X=",X)
selector.fit(X,Y)
selector.transform(X)
print("After transform,X=",X)
print("Ranking %s"%selector.ranking_)
 
#RFECV
iris=load_iris()
X=iris.data
Y=iris.target
estimator=LinearSVC()
selector=RFECV(estimator=estimator,cv=3)
selector.fit(X,Y)
print("Grid Scores %s"%selector.grid_scores_)

C.J.L:
#SelectFromModel
iris=load_iris()
X=iris.data
Y=iris.target
estimator=LinearSVC(penalty='l1',dual=False)
selector=SelectFromModel(estimator=estimator,threshold='mean')
selector.fit(X,Y)
selector.transform(X)
print("Threshold %s"%selector.threshold_)
print("Support is %s"%selector.get_support(indices=True))
 
#DictionaryLearning
X=[      [1,2,3,4,5],
      [6,7,8,9,10],
      [10,9,8,7,6],
      [5,4,3,2,1]
]
print("before transform:",X)
dct=DictionaryLearning(n_components=3)
dct.fit(X)
print("components is :",dct.components_)
print("after transform:",dct.transform(X))

from flask import Flask, Markup
app = Flask(__name__)
@app.route('/')
def index():
    return Markup('<div> hello %s </div>') % '<em>Flask</em>'

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=50, debug=True)
