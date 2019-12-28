# 特征抽取
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidfvec(data):
    '''
    tfidf
    '''
    tfidf = TfidfVectorizer()
    data =tfidf.fit_transform(data)
    return tfidf.get_feature_names(), data.toarray() 

def dictvec(data,flag = True):
    '''
    字典数据抽取
    '''
    dictdata = DictVectorizer(sparse=flag)
    res = dictdata.fit_transform(data)
    print(dictdata.get_feature_names())
    return res

def ChineseVect(data):
    '''
    中文特征化
    '''
    # jieba分词
    ret = jieba.cut(data)
    return ret

if __name__ == '__main__':
    # 实例化CounterVectorizer
    vector = CountVectorizer()
    res = vector.fit_transform(["life is short, i like python", "life is too long, i dislike python"])
    # 打印结果
    print(vector.get_feature_names())
    print(res.toarray())

    # 字典数据
    data = [
        {'name':'ryan', 'age':20},
        {'name':'alex', 'age':20},
        {'name':'Jane', 'age':18}
    ]
    temp = dictvec(data, False)
    print(temp)

