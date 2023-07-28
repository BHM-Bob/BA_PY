# mbapy.stats

This module provides some stats functions, and directly contains a function for performing Principal Component Analysis (PCA) on a pandas DataFrame.  

## Functions

### pca(df: pd.DataFrame, out_dim: int) -> pd.DataFrame

Perform Principal Component Analysis (PCA) on a pandas DataFrame.  

Parameters:  
- df (pd.DataFrame): The input DataFrame.  
- out_dim (int): The number of dimensions to reduce the DataFrame to.  

Returns:  
- pd.DataFrame: The transformed DataFrame after PCA.  

Example:  
```python
import pandas as pd
from mbapy.pca import pca

data = {'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]}
df = pd.DataFrame(data)

transformed_df = pca(df, 1)
print(transformed_df)
```

Output:  
```
          0
0 -2.828427
1 -1.414214
2  0.000000
3  1.414214
4  2.828427
```

## Constants

There are no constants defined in this module.  

## Classes

There are no classes defined in this module.