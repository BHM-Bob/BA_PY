'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-04-06 20:44:44
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-07-04 22:54:27
Description: 
'''
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def linea_reg_OLS(x: Union[str, List[str]], y:str, df:pd.DataFrame):
    """
    Perform ordinary least squares regression on the given DataFrame.

    Parameters:
        - x (str | List[str]): The column name for the independent variable, can be a list of column names for multiple independent variables.
        - y (str): The column name for the dependent variable.
        - df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        - dict: A dictionary containing the regression model, coefficients, intercept, and R-squared value.
            - 'regressor' (OLS): The fitted regression model.
            - 'coef' (np.ndarray): The coefficients of the regression model.
            - 'intercept' (float): The intercept of the regression model.
            - 'r2' (float): The R-squared value of the regression.
            - 'p' (float): The p-value of the regression.
    """
    y = df[y]
    if isinstance(x, str):
        x = [x]
    X = sm.add_constant(df[x])
    results = sm.OLS(y, X).fit()
    # 从结果中提取斜率（系数）和截距
    a = results.params[x] if isinstance(x, str) else results.params  # 斜率
    b = results.params['const']  # 截距，'const'是添加的截距项的默认列名
    r2 = results.rsquared
    p = results.pvalues[0]
    return {
       'results':results,
        'a': a,
        'b': b,
        'r2': r2,
        'p': p,
    }


def linear_reg(x:str, y:str, df:pd.DataFrame):
    """
    Perform linear regression on the given DataFrame.

    Parameters:
        - x (str): The column name for the independent variable.
        - y (str): The column name for the dependent variable.
        - df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        - dict: A dictionary containing the regression model, slope, intercept, and R-squared value.
            - 'regressor' (LinearRegression): The fitted regression model.
            - 'a' (float): The slope of the regression line.
            - 'b' (float): The intercept of the regression line.
            - 'r2' (float): The R-squared value of the regression.
    """
    x = np.array(df[x]).reshape(-1, 1)
    y = np.array(df[y]).reshape(-1, 1)
    regressor = LinearRegression()
    regressor = regressor.fit(x, y)
    a, b = regressor.coef_.item(), regressor.intercept_.item()
    r2 = regressor.score(x, y)
    return {
        'regressor':regressor,
        'a':a,
        'b':b,
        'r2':r2,
        'equation': f"y = {a:.2f}x {'+' if b>=0 else '-'} {abs(b):.2f}",
        'r2_equation': f'R^2 = {r2:.2f}'
    }
    
def quadratic_reg(x_str: str, y_str: str, df: pd.DataFrame):
    """
    Perform quadratic regression on the given DataFrame.

    Parameters:
        x_str (str): The column name for the independent variable.
        y_str (str): The column name for the dependent variable.
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        - dict: A dictionary containing the regression model, coefficients, intercept, and R-squared value.
            - 'regressor' (Pipeline): The fitted regression pipeline that includes PolynomialFeatures and LinearRegression.
            - 'polynomial_features' (PolynomialFeatures): The PolynomialFeatures object used to transform the input data.
            - 'a' (float): The coefficient of the quadratic term.
            - 'b' (float): The coefficient of the linear term.
            - 'c' (float): The intercept of the regression curve.
            - 'r2' (float): The R-squared value of the regression.
    """
    # Convert columns to numpy arrays
    x = np.array(df[x_str]).reshape(-1, 1)
    
    # Prepare polynomial features with degree 2 (for quadratic term)
    poly_features = PolynomialFeatures(degree=2)
    x_poly = poly_features.fit_transform(x)

    # Fit the quadratic regression model
    regressor = LinearRegression()
    regressor = regressor.fit(x_poly, np.array(df[y_str]).reshape(-1, 1))

    # Extract coefficients and intercept
    coefficients = regressor.coef_
    intercept = regressor.intercept_

    # Unpack coefficients into a, b, c format for a quadratic equation: y = ax^2 + bx + c
    a, b, c = coefficients[0][-1], coefficients[0][1], intercept[0]

    # Calculate R-squared value
    r2 = regressor.score(x_poly, np.array(df[y_str]).reshape(-1, 1))

    return {
        'regressor': regressor,
        'polynomial_features': poly_features,
        'a': a,
        'b': b,
        'c': c,
        'r2': r2,
        'equation': f"y = {a:.2f}x^2 {'+' if b>=0 else '-'} {abs(b):.2f}x {'+' if c>=0 else '-'} {abs(c):.2f}",
        'r2_equation': f'R^2 = {r2:.2f}'
    }


if __name__ == '__main__':
    # dev code
    data = pd.read_excel('data/plot.xlsx', sheet_name='MWM')
    print(linea_reg_OLS('Duration', 'First Entry Speed', data))
    print(linear_reg('Duration', 'First Entry Speed', data))
