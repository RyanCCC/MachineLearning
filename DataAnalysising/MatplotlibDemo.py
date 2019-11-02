from matplotlib import pyplot as plt

if __name__ == '__main__':
    print('hello world')
    x = range(2, 26, 2)
    y = [15, 13, 14, 5, 17, 20, 25, 26, 26, 24, 22, 18]
    plt.plot(x, y)
    plt.show()