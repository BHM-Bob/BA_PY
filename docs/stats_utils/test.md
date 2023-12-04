# mbapy.stats.test

This module provides statistical analysis functions for data analysis.  

## Functions

### get_interval(mean = None, se = None, data:Optional[Union[np.ndarray, List[int], pd.Series]] = None, confidence:float = 0.95) -> Tuple[Tuple[float, float], Dict[str, Any]]

Calculate the confidence interval.  

#### Parameters  
- mean (float, optional): The mean value. Defaults to None.  
- se (float, optional): The standard error. Defaults to None.  
- data (np.ndarray or List[int] or pd.Series, optional): The data. Defaults to None.  
- confidence (float, optional): The confidence level. Defaults to 0.95.  

#### Returns
- Tuple[Tuple[float, float], Dict[str, Any]]: The confidence interval and additional information.  

#### Notes
- The confidence interval is calculated as ± 1.96 * SE or other depends on the confidence level.  

#### Example
```python
get_interval(mean=10, se=2, confidence=0.95)
```

### ttest_1samp(x1 = None, x2:float = None, factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None, **kwargs) -> Tuple[float, float]

Perform a one-sample t-test.  

#### Parameters  
- x1 (float or None, optional): The first sample. Defaults to None.  
- x2 (float, optional): The second sample. Defaults to None.  
- factors (Dict[str, List[str]], optional): The factors for data filtering. Defaults to None.  
- tag (str, optional): The column name of the data. Defaults to None.  
- df (pd.DataFrame, optional): The data frame. Defaults to None.  
- kwargs: Additional arguments for the t-test.  

#### Returns
- Tuple[float, float]: The p-value and the t-statistic.  

#### Example
```python
ttest_1samp(x1=10, tag='score', df=data)
```

### ttest_ind(x1 = None, x2 = None, factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None, **kwargs) -> Tuple[float, float]

Perform an independent samples t-test.  

#### Parameters  
- x1 (float or None, optional): The first sample. Defaults to None.  
- x2 (float or None, optional): The second sample. Defaults to None.  
- factors (Dict[str, List[str]], optional): The factors for data filtering. Defaults to None.  
- tag (str, optional): The column name of the data. Defaults to None.  
- df (pd.DataFrame, optional): The data frame. Defaults to None.  
- kwargs: Additional arguments for the t-test.  

#### Returns
- Tuple[float, float]: The p-value from the Levene's test and the p-value from the t-test.  

#### Example
```python
ttest_ind(factors={'group': ['A', 'B']}, tag='score', df=data)
```

### ttest_rel(x1 = None, x2 = None, factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None, **kwargs) -> Tuple[float, float]

Perform a paired samples t-test.  

#### Parameters  
- x1 (float or None, optional): The first sample. Defaults to None.  
- x2 (float or None, optional): The second sample. Defaults to None.  
- factors (Dict[str, List[str]], optional): The factors for data filtering. Defaults to None.  
- tag (str, optional): The column name of the data. Defaults to None.  
- df (pd.DataFrame, optional): The data frame. Defaults to None.  
- kwargs: Additional arguments for the t-test.  

#### Returns
- Tuple[float, float]: The p-value and the t-statistic.  

#### Example
```python
ttest_rel(factors={'group': ['A', 'B']}, tag='score', df=data)
```

### mannwhitneyu(x1 = None, x2 = None, factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None, **kwargs) -> Tuple[float, float]

Perform a Mann-Whitney U test.  

#### Parameters  
- x1 (float or None, optional): The first sample. Defaults to None.  
- x2 (float or None, optional): The second sample. Defaults to None.  
- factors (Dict[str, List[str]], optional): The factors for data filtering. Defaults to None.  
- tag (str, optional): The column name of the data. Defaults to None.  
- df (pd.DataFrame, optional): The data frame. Defaults to None.  
- kwargs: Additional arguments for the Mann-Whitney U test.  

#### Returns
- Tuple[float, float]: The p-value and the U statistic.  

#### Example
```python
mannwhitneyu(factors={'group': ['A', 'B']}, tag='score', df=data)
```

### shapiro(x1 = None, factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None, **kwargs) -> Tuple[float, float]

Perform a Shapiro-Wilk test for normality.  

#### Parameters  
- x1 (float or None, optional): The sample. Defaults to None.  
- factors (Dict[str, List[str]], optional): The factors for data filtering. Defaults to None.  
- tag (str, optional): The column name of the data. Defaults to None.  
- df (pd.DataFrame, optional): The data frame. Defaults to None.  
- kwargs: Additional arguments for the Shapiro-Wilk test.  

#### Returns
- Tuple[float, float]: The test statistic and the p-value.  

#### Example
```python
shapiro(factors={'group': ['A', 'B']}, tag='score', df=data)
```

### pearsonr(x1 = None, x2 = None, factors:Dict[str, List[str]] = None, tags:List[str] = None, df:pd.DataFrame = None, **kwargs) -> Tuple[float, float]

Calculate the Pearson correlation coefficient.  

#### Parameters  
- x1 (float or None, optional): The first sample. Defaults to None.  
- x2 (float or None, optional): The second sample. Defaults to None.  
- factors (Dict[str, List[str]], optional): The factors for data filtering. Defaults to None.  
- tags (List[str], optional): The column names of the data. Defaults to None.  
- df (pd.DataFrame, optional): The data frame. Defaults to None.  
- kwargs: Additional arguments for the Pearson correlation coefficient.  

#### Returns
- Tuple[float, float]: The correlation coefficient and the p-value.  

#### Example
```python
pearsonr(factors={'group': ['A', 'B']}, tags=['x', 'y'], df=data)
```

### chi2_contingency(observed = None, factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None, **kwargs) -> Tuple[Tuple[float, float, int, np.ndarray], pd.DataFrame]

Perform a chi-squared test for independence.  

#### Parameters  
- observed (np.ndarray or None, optional): The observed frequencies. Defaults to None.  
- factors (Dict[str, List[str]], optional): The factors for data filtering. Defaults to None.  
- tag (str, optional): The column name of the data. Defaults to None.  
- df (pd.DataFrame, optional): The data frame. Defaults to None.  
- kwargs: Additional arguments for the chi-squared test.  

#### Returns
- Tuple[Tuple[float, float, int, np.ndarray], pd.DataFrame]: The test statistic, the p-value, the degrees of freedom, the expected frequencies, and the observed frequencies.  

#### Example
```python
chi2_contingency(factors={'group': ['A', 'B']}, tag='outcome', df=data)
```

### fisher_exact(observed = None, factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None, **kwargs) -> Tuple[Tuple[float, float], pd.DataFrame]

Perform a Fisher's exact test.  

#### Parameters  
- observed (np.ndarray or None, optional): The observed frequencies. Defaults to None.  
- factors (Dict[str, List[str]], optional): The factors for data filtering. Defaults to None.  
- tag (str, optional): The column name of the data. Defaults to None.  
- df (pd.DataFrame, optional): The data frame. Defaults to None.  
- kwargs: Additional arguments for the Fisher's exact test.  

#### Returns
- Tuple[Tuple[float, float], pd.DataFrame]: The odds ratio, the p-value, the expected frequencies, and the observed frequencies.  

#### Example
```python
fisher_exact(factors={'group': ['A', 'B']}, tag='outcome', df=data)
```

### f_oneway(Xs:list = None, factors:Dict[str, List[str]] = None, tag:str = None, df:pd.DataFrame = None) -> Tuple[float, float]

Perform a one-way ANOVA.  

#### Parameters  
- Xs (list or None, optional): The samples. Defaults to None.  
- factors (Dict[str, List[str]], optional): The factors for data filtering. Defaults to None.  
- tag (str, optional): The column name of the data. Defaults to None.  
- df (pd.DataFrame, optional): The data frame. Defaults to None.  

#### Returns
- Tuple[float, float]: The F-statistic and the p-value.  

#### Example
```python
f_oneway(Xs=[x1, x2, x3], factors={'group': ['A', 'B', 'C']}, tag='score', df=data)
```

### multicomp_turkeyHSD(factors:Dict[str, List[str]], tag:str, df:pd.DataFrame, alpha:float = 0.05) -> sm.stats.multicomp.pairwise_tukeyhsd

Perform a Tukey's HSD test for multiple comparisons.  

#### Parameters  
- factors (Dict[str, List[str]]): The factors for data filtering.  
- tag (str): The column name of the data.  
- df (pd.DataFrame): The data frame.  
- alpha (float, optional): The significance level. Defaults to 0.05.  

#### Returns
- sm.stats.multicomp.pairwise_tukeyhsd: The result object from the Tukey's HSD test.  

#### Example
```python
multicomp_turkeyHSD(factors={'group': ['A', 'B', 'C']}, tag='score', df=data)
```

### turkey_to_table(turkey_result:sm.stats.multicomp.pairwise_tukeyhsd) -> pd.DataFrame

Generate a table summarizing the results of the Tukey's HSD test.  

#### Parameters  
- turkey_result (sm.stats.multicomp.pairwise_tukeyhsd): The result object from the Tukey's HSD test.  

#### Returns
- pd.DataFrame: A DataFrame containing the results of the Tukey's HSD test.  

#### Example
```python
turkey_result = multicomp_turkeyHSD(factors={'group': ['A', 'B', 'C']}, tag='score', df=data)
turkey_to_table(turkey_result)
```

### multicomp_dunnett(factor:str, exp:List[str], control:str, df:pd.DataFrame, **kwargs) -> scipy.stats.dunnett

Perform a Dunnett's test for multiple comparisons.  

#### Parameters  
- factor (str): The column name of the factor.  
- exp (List[str]): The names of the experimental groups.  
- control (str): The name of the control group.  
- df (pd.DataFrame): The data frame.  
- kwargs: Additional arguments for the Dunnett's test.  

#### Returns
- scipy.stats.dunnett: The result object from the Dunnett's test.  

#### Example
```python
multicomp_dunnett(factor='group', exp=['A', 'B', 'C'], control='D', df=data)
```

### multicomp_bonferroni(factors:Dict[str, List[str]], tag:str, df:pd.DataFrame, alpha:float = 0.05) -> pd.DataFrame

Perform a Bonferroni correction for multiple comparisons.  

#### Parameters  
- factors (Dict[str, List[str]]): The factors for data filtering.  
- tag (str): The column name of the data.  
- df (pd.DataFrame): The data frame.  
- alpha (float, optional): The significance level. Defaults to 0.05.  

#### Returns
- pd.DataFrame: A DataFrame containing the p-values after Bonferroni correction.  

#### Example
```python
multicomp_bonferroni(factors={'group': ['A', 'B', 'C']}, tag='score', df=data)
```

## Constants

### p_value_to_stars(p_value:float) -> str

Convert a p-value to stars indicating the significance level.  

#### Parameters  
- p_value (float): The p-value to convert to stars.  

#### Returns
- str: The string representation of the number of stars. If p >= 0.05, return ''.  

#### Example
```python
p_value_to_stars(0.01)
```