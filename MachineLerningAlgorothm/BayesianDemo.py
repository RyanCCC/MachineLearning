from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def Naviebayes():
    # 获取数据集
    news = datasets.fetch_20newsgroups(subset='all')
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.traget, test_size= 0.25)
    tf = TfidfVectorizer()
    # 以训练集中的词列表进行每篇文章重要性统计
    x_train = tf.fit_transform(x_train)
    print(tf.get_feature_names())
    x_test = tf.fit_transform(x_test)
    # 朴素贝叶斯分类
    mlt = MultinomialNB(alpha=1.0)
    print(x_train.toarray())
    mlt.fit(x_train, y_train)
    y_predict = mlt.predict(x_test)
    scores = mlt.score(x_test, y_test)
    print(y_predict)
    print(scores)

if __name__ == '__main__':
    
    Naviebayes()