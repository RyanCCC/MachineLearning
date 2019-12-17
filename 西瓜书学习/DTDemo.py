from sklearn import datasets
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import numpy as np

if __name__ == '__main__':
    #载入数据集
    iris = datasets.load_iris()
    iris.data=Normalizer().fit_transform(iris.data)
    iris_data = iris["data"]
    iris_label = iris["target"]
    iris_target_name = iris['target_names']
    X = np.array(iris_data)
    Y = np.array(iris_label)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=1)
    # 加载数据集
    ds_iris = datasets.load_iris()
    # 决策树
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, Y_train)
    y_predict = clf.predict(X_test)
    sum = 0
    for i in range(len(Y_test)):
        if y_predict[i]==Y_test[i]:
            sum +=1
    print("sum:%s,len(Y_test):%s"%(sum,len(Y_test)))
    print("预测率：%s"%(sum/len(Y_test)))
    print("类别是",iris_target_name[clf.predict([[7,1,1,1]])[0]])
