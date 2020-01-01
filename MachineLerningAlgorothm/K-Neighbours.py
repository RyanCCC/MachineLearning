from sklearn.neighbors import KNeighborsClassifier
import sklearn.datasets as ds


def KnnClassifier(data):
    # 读取数据
    print("")
    # 处理数据

    # 特征工程（标准化）


if __name__ == '__main__':
    boston = ds.load_boston()
    print(boston.data)
    print(boston.target)
    print(boston)
