# mbapy.sats.reg

This function performs linear regression on a given DataFrame.

## Function

### linear_reg(x:str, y:str, df:pd.DataFrame) -> dict

Perform linear regression on the given DataFrame.

#### Parameters
- x (str): The column name for the independent variable.
- y (str): The column name for the dependent variable.
- df (pd.DataFrame): The DataFrame containing the data.

#### Returns
- dict: A dictionary containing the regression model, slope, intercept, and R-squared value.
    - 'regressor' (LinearRegression): The fitted regression model.
    - 'a' (float): The slope of the regression line.
    - 'b' (float): The intercept of the regression line.
    - 'r2' (float): The R-squared value of the regression.

#### Example
```python
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]})
result = linear_reg('x', 'y', df)
print(result)
```

Output:
```
{
    'regressor': LinearRegression(),
    'a': 2.0,
    'b': 0.0,
    'r2': 1.0
}
```

Notes:
- The function uses the `LinearRegression` class from the `sklearn.linear_model` module to perform the linear regression.
- The independent variable `x` and dependent variable `y` should be column names in the DataFrame `df`.
- The function reshapes the input arrays `x` and `y` to have a single feature.
- The function returns a dictionary containing the fitted regression model, slope, intercept, and R-squared value.