'''
标准缩放：1. 归一化  2. 标准化
类别型：one-hot编码
时间类型：时间的切分
'''
'''
特征选择
'''
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

def PCADemo(data, n_componet=0.95):
    '''
    目的：数据维数压缩，尽可能降低原数据维数
    作用：可以削减回归分析或者聚类分析中特征数量
    应用场景：特征数量上百，特征相关性高
    n_componet:小数表示缩小的百分比
    '''
    temp = PCA(n_components=n_componet)
    res = temp.fit_transform(data)
    return res



def variancethreshold(data):
    vt = VarianceThreshold(threshold=0.0)
    res = vt.fit_transform(data)
    return res


def ImputerApi(data):
    '''
    缺失值处理
    '''
    im = Imputer(missing_values="NaN", strategy="mean", axis=0)
    res = im.fit_transform(data)
    return res

def StandardScalerapi(data):
    '''
    数据标准化
    '''
    standscalar = StandardScaler()
    ret = standscalar.fit_transform(data)
    print(ret)
    return  ret

def minmaxscale(data, feature_range = [0, 1]):
    '''
    对数据进行归一化
    '''
    mimx = MinMaxScaler(feature_range=feature_range)
    ret = mimx.fit_transform(data)
    print(ret)
    return ret

if __name__ =='__main__':
    data = [90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]
    res = minmaxscale(data)
    print(res)
    res = StandardScalerapi(data)
    print(res)
    data = [[0,2,0,3],[0,1,4,3], [0,1,1,3]]
    res = variancethreshold(data)
    print(res)
    data = [[2,8,4,5],[6,3,0,8], [5,4,9,1]]
    res = PCADemo(data)
    print(res)