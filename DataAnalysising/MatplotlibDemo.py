from matplotlib import pyplot as plt
import matplotlib
from matplotlib import font_manager

# font= {
#     'family':'MicroSoft YaHei',
#     'weight':'bold',
#     'size':'larger'
# }

# matplotlib.rc("font", **font)

# my_font = font_manager.FontProperties()

if __name__ == '__main__':
    # print('hello world')
    fig = plt.figure(figsize=(20, 8), dpi = 80)
    x = range(2, 26, 2)
    y = [15, 13, 14, 5, 17, 20, 25, 26, 26, 24, 22, 18]
    plt.plot(x, y)
    # 设置x刻度
    _x = x
    _xticks_labels = [f"10:0{i+1}" if i<9 else f"10:{i+1}" for i in range(0,13)]
    plt.xticks(x, _xticks_labels,rotation=90)
    plt.yticks(range(min(y),max(y)+1))
    plt.xlabel("Time")
    plt.ylabel("Tempearature")
    plt.title('Temperature')
    plt.savefig("./sig_size.png")
    plt.show()