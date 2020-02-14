# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

# 递归统计一个1-9999中出现的数字
def count_digit(number):
    return len(str(number))

def countThree(digit):
    if not isinstance(digit,int):
        raise TypeError('number is not int')
    # digit = len(str(number))
    if(digit <=0):
        return 0
    if(digit ==1):
        return 1
    return 10*countThree(digit-1) + 10 **(digit-1)
if __name__ =='__main__':
    # print(10**2)
    print(countThree(count_digit(9999)))
    labels='商城批发','超市','专营店','网络'

    sizes=55,8,12,25

    colors='lightgreen','gold','lightskyblue','lightcoral'

    explode=0,0,0,0

    plt.pie(sizes,explode=explode,labels=labels,
    colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)
    plt.axis('equal')
    plt.show()