'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-04-04 16:45:23
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-11-15 17:09:29
Description: 
'''
from itertools import combinations
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scikit_posthocs as sp
import scipy
import statsmodels.api as sm
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import MultiComparison

import mbapy.stats.df as msd


def get_interval(mean = None, se = None, data:Optional[Union[np.ndarray, List[int], pd.Series]] = None, confidence:float = 0.95):
    """
    置信区间\n
    ± 1.96 * SE or other depends on confidence
    """
    assert se is not None or data is not None, 'se is None and data is None'
    kwargs = {
        'scale':se if se is not None else scipy.stats.sem(data)       
    }
    if mean is not None:
        kwargs.update({'loc':mean})
    if data is not None:
        kwargs.update({'df':len(data) - 1, 'loc':np.mean(data).item()})
    return scipy.stats.norm.interval(confidence = confidence, loc = kwargs['loc'], scale = kwargs['scale']), kwargs

def _get_x1_x2(x1 = None, x2 = None,
               factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None):
    """以同一列factors提取同一列tag的x1和x2，其余factors仅作筛选作用"""
    if factors is not None and tag is not None and df is not None:
        sub_df = msd.get_df_data(factors, [tag], df)
        fac_name = list(factors.keys())[0]
        x1 = sub_df.loc[sub_df[fac_name] == factors[fac_name][0], [tag]].values.reshape(-1)
        if len(factors[fac_name]) == 2:
            x2 = sub_df.loc[sub_df[fac_name] == factors[fac_name][1], [tag]].values.reshape(-1)
        elif len(factors[fac_name]) > 2:
            raise ValueError('Only support 1 or 2 value of factors')
    return x1, x2

def _get_x1_x2_R(x1 = None, x2 = None,
               factors:Dict[str, List[str]] = None, tags:List[str] = None, df:pd.DataFrame = None):
    """
    提取多列tag的x1和x2，factors仅作筛选作用
    tags为x1和x2...的tag
    """
    ret = [x1, x2]
    if factors is not None and tags is not None and df is not None:
        ret = []
        sub_df = msd.get_df_data(factors, tags, df)
        ret = [sub_df.loc[:, [tag]].values.reshape(-1) for tag in tags]
    return ret

def ttest_1samp(x1 = None, x2:float = None,
                 factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None, **kwargs):
    """单样本T检验"""
    x1, x2 = _get_x1_x2(x1, x2, factors, tag, df)
    return scipy.stats.ttest_1samp(x1, x2, **kwargs)

def ttest_ind(x1 = None, x2 = None,
                 factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None, **kwargs):
    """
    独立样本t检验(双样本T检验):检验两组独立样本均值是否相等\n
    要求正态分布，正态检验结果由scipy.stats.levene计算并返回\n
    levene 检验P值 > 0.05，接受原假设，认为两组方差相等\n
    如不相等， scipy.stats.ttest_ind() 函数中的参数 equal_var 会设置成 False
    """
    x1, x2 = _get_x1_x2(x1, x2, factors, tag, df)
    levene = scipy.stats.levene(x1, x2)
    return levene.pvalue, scipy.stats.ttest_ind(x1, x2, equal_var=levene.pvalue > 0.05)

def ttest_rel(x1 = None, x2 = None,
                 factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None, **kwargs):
    """配对样本T检验:比较从同一总体下分出的两组样本在不同场景下，均值是否存在差异(两个样本的样本量要相同)"""
    x1, x2 = _get_x1_x2(x1, x2, factors, tag, df)
    return scipy.stats.ttest_rel(x1, x2, **kwargs)

def mannwhitneyu(x1 = None, x2 = None,
                 factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None, **kwargs):
    """
    Mann-Whitney U检验:数据不具有正态分布时使用。\n
    评估了两个抽样群体是否可能来自同一群体，这两个群体在数据方面是否具有相同的形状？\n
    证明这两个群体是否来自于具有不同水平的相关变量的人群。\n
    此包装函数同时支持直接输入和mbapy-style数据输入
    """
    x1, x2 = _get_x1_x2(x1, x2, factors, tag, df)
    return scipy.stats.mannwhitneyu(x1, x2, **kwargs)

def wilcoxon(x1 = None, x2 = None,
             factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None, **kwargs):
    """
    Wilcoxon signed-rank test, 样本不符合正态分布但配对时使用。
    """
    x1, x2 = _get_x1_x2(x1, x2, factors, tag, df)
    return scipy.stats.ranksums(x1, x2, **kwargs)

def shapiro(x1 = None,
            factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None, **kwargs):
    """
    夏皮洛-威尔克检验（Shapiro—Wilk test），一般又称W检验。W检验是一种类似于利用秩进行相关性检验的方法。同样需要注意的是，W检验与K-S检验一样，原假设是“样本数据来自的分布与正态分布无显著差异”，因此一般来说，检验结果P>0.05才是我们的目标。
    p > 0.05 为正态分布
    """
    x1, _ = _get_x1_x2(x1, None, factors, tag, df)
    return scipy.stats.shapiro(x1, **kwargs)

def pearsonr(x1 = None, x2 = None,
             factors:Dict[str, List[str]] = None, tags:List[str] = None, df:pd.DataFrame = None, **kwargs):
    """
    pearsonr相关系数:检验两个样本是否有线性关系\n
    该皮尔逊相关系数取值在-1和+1之间，为0时表示没有相关性。 -1或+1的相关性表示存在明确的线性关系。 正相关表示，随着x的增加，y也随之增加。 负相关性表示随着x增加，y减小。
    """
    x1, x2 = _get_x1_x2_R(x1, x2, factors, tags, df)
    return scipy.stats.pearsonr(x1, x2, **kwargs)

def auto_ind_test(x1 = None, x2 = None,
                  factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None,
                  float_round:int = 5, **kwargs):
    """
    Params:
        - x1, x2: samples to compare
        - factors, tag, df: mbapy-style data input
        - float_round: round the result to float_round decimal places
        - **kwargs: other parameters for the test function
        
    Returns:
        - result: the result of the test function
        
    Notes:
    自动选择最合适的检验方法\n
    - 符合正态分布
        - 配对：scipy.stats.ttest_rel
        - 不配对：
            - 等方差：scipy.stats.ttest_ind(equal_var=True)
            - 不确定等方差：scipy.stats.ttest_ind(equal_var=False)
    - 不符合正态分布
        - 配对：scipy.stats.wilcoxon
        - 不配对：scipy.stats.mannwhitneyu
    
    Items:
        - skew: 偏度，0代表正态，大于0代表右偏，小于0代表左偏
        - kurtosis: 峰度，峰度包括正态分布（峰度值=3），厚尾（峰度值>3），瘦尾（峰度值<3）
        - normality: 正态性检验结果
        - equal_var: 方差是否相等，直接检验X1和X2
    """
    x1, x2 = _get_x1_x2(x1, x2, factors, tag, df)
    _fmt_result = lambda result: f'{result[0]:.{float_round}f}, p={result[1]:.{float_round}f}, {result[1] < 0.05}' # < 0.05 为显著差异
    _fmt_result_gt = lambda result: f'{result[0]:.{float_round}f}, p={result[1]:.{float_round}f}, {result[1] > 0.05}' # < 0.05 为显著差异
    print('-'*22, 'mbapy.stats.auto_ind_test', '-'*23)
    if factors is not None and tag is not None and df is not None:
        fac_name = factors[list(factors.keys())[0]]
        if len(fac_name) == 0:
            print('no sub factor found in the passed factors, try to find it in the df')
            fac_name = factors[list(factors.keys())[0]]
            if len(fac_name) != 2:
                print(f'only accept 2 sub factors, but got {fac_name}, try to use the first two sub factors')
        elif len(fac_name) != 2:
            print(f'only accept 2 sub factors, but got {fac_name}, try to use the first two sub factors')
        print(f'Factor: {list(factors.keys())[0]}')
        print(f'X1: {fac_name[0]}, X2: {fac_name[1]}')
    # show x1 and x2 statistics info (each)
    shapiro_results = []
    print('-'*19, 'X1 and X2 Statistics Info (each)', '-'*19)
    for i, x in enumerate([x1, x2]):
        print(f'X{i+1}: N={len(x)}, mean={np.mean(x):.{float_round}f}, std={np.std(x):.{float_round}f}, SE={scipy.stats.sem(x):.{float_round}f}')
        print(f'skew={scipy.stats.skew(x):.{float_round}f}, kurtosis={scipy.stats.kurtosis(x):.{float_round}f}')
        shapiro_results.append(shapiro(x))
        print(f'normality(shapiro)={_fmt_result_gt(shapiro_results[-1])}')
        print('-'*70)
    # show x1 and x2 statistics info (together)
    print('-'*17, 'X1 and X2 Statistics Info (together)', '-'*17)
    equal_vars = scipy.stats.levene(x1, x2)
    print(f'equal_var(levene)={_fmt_result_gt(equal_vars)}')
    print('-'*70)
    # perform test
    if shapiro_results[0][1] > 0.05 and shapiro_results[1][1] > 0.05:
        if len(x1) != len(x2):
            is_equal_vals = equal_vars[1] > 0.05
            result = scipy.stats.ttest_ind(x1, x2, equal_var=is_equal_vals, **kwargs)
            print(f'perform ttest_ind(equal_var={is_equal_vals}): {_fmt_result(result)}')
        else:
            result = scipy.stats.ttest_rel(x1, x2, **kwargs)
            print(f'perform ttest_rel: {_fmt_result(result)}')
    else:
        if len(x1) == len(x2):
            result = scipy.stats.wilcoxon(x1, x2, **kwargs)
            print(f'perform wilcoxon: {_fmt_result(result)}')
        else:
            result = scipy.stats.mannwhitneyu(x1, x2, **kwargs)
            print(f'perform mannwhitneyu: {_fmt_result(result)}')
    print('-'*70)
    return result


def _get_observe(observed = None,
                 factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None):
    if observed is None and factors is not None and tag is not None and df is not None:
        @msd.pro_bar_data_R(list(factors.keys()), [tag], df, [''])
        def get_sum(values):
            return [values.sum()]
        ndf = get_sum()
        col = list(factors.keys())[0]
        rol = list(factors.keys())[1]
        mat = pd.DataFrame(np.zeros(shape = (len(factors[rol]), len(factors[col]))),
                           index=factors[rol], columns=factors[col])
        for c in factors[col]:
            for r in factors[rol]:
                mat[c][r] = sum(ndf.loc[(ndf[col] == c) & (ndf[rol] == r), [tag]].values)
        observed = mat
    return observed

def chi2_contingency(observed = None,
                      factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None, **kwargs):
    """
    卡方检验 Chi-Squared Test:\n
    p > 0.05 为独立(不相关)。\n
    若存在某一个格子的理论频数T<5或p值与规定的显著性水平(如0.05)相近时，改用Fisher's exact test\n
    1. 样本来自简单随机抽样
    2. 各个格子是相互独立的;
    3. 样本量应尽可能大。总观察数应不小于40，且每个格子的频数应大于等于5\n
    支持直接输入和mbapy-style数据输入\n
    mbapy-style: factors={'a':['a1', 'a2', ...], 'b':['b1', 'b2', ...]}, tag is value of 0/1 or number
    """
    observed = _get_observe(observed, factors, tag, df)
    return scipy.stats.chi2_contingency(observed, **kwargs), observed

def fisher_exact(observed = None,
                      factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None, **kwargs):
    """
    Fisher确切概率法 Fisher's exact test:\n
    2x2 contingency table, p > 0.05 为独立(不相关)\n
    支持直接输入和mbapy-style数据输入\n
    mbapy-style: factors={'a':['a1', 'a2'], 'b':['b1', 'b2']}, tag is value of 0/1 or number
    """
    observed = _get_observe(observed, factors, tag, df)
    return scipy.stats.fisher_exact(observed, **kwargs), observed

def f_oneway(Xs:list = None,
             factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None):
    """
    方差分析检验(ANOVA) Analysis of Variance Test (ANOVA):p < 0.05 为显著差异\n
    检验两个或多个独立样本的均值是否有显著差异\n
    1. 每个样本中的观测值都是独立和相同分布的(iid)。
    2. 每个样本中的观测值都是正态分布。
    3. 每个样本中的观测值具有相同的方差。\n
    支持直接输入(Xs)和mbapy-style数据输入
    """
    if Xs is None and factors is not None and tag is not None and df is not None:
        sub_df = msd.get_df_data(factors, [tag], df)
        fac_name = list(factors.keys())[0]
        sub_facs = factors[fac_name]
        Xs = [sub_df.loc[sub_df[fac_name] == f, [tag]].values for f in sub_facs]
    return scipy.stats.f_oneway(*Xs)

def p_value_to_stars(p_value):
    """
    Determine the number of stars for a given p-value.

    Parameters:
    - p_value (float): The p-value to convert to stars.

    Returns:
    - str: The string representation of the number of stars. If p >= 0.05, return ''
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

def multicomp_turkeyHSD(factors:Dict[str, List[str]], tag:str, df:pd.DataFrame, alpha:float = 0.05):
    """
    using statsmodels.stats.multicomp.pairwise_tukeyhsd, Tukey's HSD 检验是一种多重比较方法，用于比较多个处理组之间的差异。\n
    Tukey的HSD法要求各样本的样本相等或者接近, 在样本量相差很大的情况下还是建议使用其他方法\n
    Tukey的HSD检验比Bonferroni法更加的保守\n
    Parameters:
    -----------
    factors: dict[str, list[str]], the first key-values is the data-choosed factor, the other are the factors to help to choose
    tag: str, data column name
    df: pd.DataFrame
    alpha: float
    """
    sub_df = msd.get_df_data(factors, [tag], df)
    return sm.stats.multicomp.pairwise_tukeyhsd(sub_df[tag], sub_df[list(factors.keys())[0]], alpha)

def turkey_to_table(turkey_result):
    """
    Generate a table summarizing the results of the Tukey's HSD test.

    Parameters:
    - turkey_result (sm.stats.multicomp.TukeyHSDResult): The result object from the Tukey's HSD test.

    Returns:
    - table (pd.DataFrame): A DataFrame containing the following columns:
        - 'Group 1': The first group being compared.
        - 'Group 2': The second group being compared.
        - 'Mean Difference': The mean difference between the two groups.
        - 'p-value': The p-value associated with the mean difference.
        - 'Stars': The number of stars indicating the significance level of the mean difference.
        - 'Reject': A boolean value indicating whether the null hypothesis is rejected for the mean difference.
    
    Example:
    >>>     Group 1 Group 2  Mean Difference   p-value Stars  Reject
    >>> 0       A       B             -1.0  0.386244     -   False
    >>> 1       A       C             -2.0  0.063756     -   False
    >>> 2       B       C             -1.0  0.386244     -   False
    """
    groups = turkey_result.groupsunique
    data = turkey_result.meandiffs
    p_values = turkey_result.pvalues
    reject = turkey_result.reject
    
    table_data = []
    for i, (group1, group2) in enumerate(combinations(groups, 2)):
        mean_diff = data[i]
        p_value = p_values[i]
        is_rejected = reject[i]
        stars = p_value_to_stars(p_value)
        
        table_data.append([group1, group2, mean_diff, p_value, stars, is_rejected])
    
    table = pd.DataFrame(table_data, columns=['Group 1', 'Group 2', 'Mean Difference', 'p-value', 'Stars', 'Reject'])
    return table

def multicomp_dunnett(factor:str, exp:List[str], control:str, df:pd.DataFrame, **kwargs):
    """
    using SciPy 1.11 scipy.stats.dunnett, 用于比较一个处理组与其他多个处理组之间的差异\n
    Parameters:
    -----------
    factors: str, means colum name for expiremental factor and control factor
    exp: list[str], sub factors stands for experiment group
    control: str, sub factors stands for control group
    df: pd.DataFrame
    """
    exps = [np.array(df[factor][df[factor]==e]) for e in exp]
    return scipy.stats.dunnett(*exps, np.array(df[factor][df[factor]==control]), **kwargs)

def multicomp_bonferroni(factors:Dict[str, List[str]], tag:str, df:pd.DataFrame, alpha:float = 0.05):
    """
    using scikit_posthocs.posthoc_dunn, Dunn 检验是一种非参数的多重比较方法，用于比较多个处理组之间的差异。\n
    Bonferroni method\n
    Parameters:
    -----------
    factors: dict[str, list[str]], the first key-values is the data-choosed factor, the other are the factors to help to choose
    tag: str, data column name
    df: pd.DataFrame
    alpha: float
    """
    sub_df = msd.get_df_data(factors, [tag], df)
    return sp.posthoc_dunn(sub_df, val_col=tag, group_col=list(factors.keys())[0],
                           p_adjust='bonferroni')

if __name__ == '__main__':
    df = pd.read_excel('./data/plot.xlsx', sheet_name='ym')
    model = auto_ind_test(None, None, {'solution':['K', 'Mg'], 'time':['after']}, 'whole', df)
    