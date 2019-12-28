from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

if __name__ == "__main__":
    dataset = load_iris()
    print(dataset.data)
    print(dataset.target)
    # 返回值：训练集 train x_train  y_train  测试集 test  x_test   y_test
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.25)
    print(f'训练集特征值：{x_train}, 目标值：{y_train}')
    print(f'测试集特征值：{x_test}, 目标值：{y_test}')
