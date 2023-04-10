
import pandas as pd
import numpy as np

def remove_simi(tag:str, df:pd.DataFrame, sh:float = 1.,  bakend:str = 'numpy'):
    """
    给定一组数，去除一些(最小数目)数，使任意两数差的绝对值大于或等于阈值\n
    算法模仿自Needleman-Wushsch序列对比算法\n
    e.g.:\n
    df = pd.DataFrame({'d':[1, 2, 3, 3, 5, 6, 8, 13]})\n
    print(remove_simi('d', df, 2.1, 'numpy'))\n
    """
    ndf = df.sort_values(by = tag, ascending=True)
    to_remove_idx = []
    if bakend  == 'numpy':
        arr = np.array(ndf[tag]).reshape([1, len(ndf[tag])])
        mat = arr.repeat(arr.shape[1], axis = 0) - arr.transpose(1, 0).repeat(arr.shape[1], axis = 1)
        i, j, k = 0, 0, mat.shape[0]
        while i < k and j < k:
            if i == j:
                j += 1
            elif mat[i][j] < sh:
                to_remove_idx.append(j)
                mat[i][j] = mat[i][j-1]#skip for next element in this row
                mat[j] = arr - mat[i][j]#skip for row j
                j += 1
            elif mat[i][j] >= sh:
                i += 1
    ndf.drop(labels = to_remove_idx, inplace=True)
    return ndf
    