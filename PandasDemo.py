import pandas as pd
import numpy as np


if __name__ == '__main__':
    df = pd.DataFrame(
        {
            '类别':['水果','水果','水果','蔬菜','蔬菜','肉类', '肉类'],
            '产地':['美国','中国','中国','中国','新西兰', '新西兰', '美国'],
            '名称':['苹果','梨子', '草莓', '番茄', '马铃薯', '牛肉', '羊肉'],
            '价格':[5, 5, 9, 3, 2, 10, 8],
            '数量':[5, 5, 10, 3, 3, 13, 20]
        }
    )
    print(df)
    print(df.pivot_table(index=['产地'], columns=['类别']))
    # 缺失值为0
    print(df.pivot_table(index=['产地'], columns=['类别'], fill_value=0))
