# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
if __name__ =='__main__':
    labels='商城批发','超市','专营店','网络'

    sizes=55,8,12,25

    colors='lightgreen','gold','lightskyblue','lightcoral'

    explode=0,0,0,0

    plt.pie(sizes,explode=explode,labels=labels,
    colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)
    plt.axis('equal')
    plt.show()