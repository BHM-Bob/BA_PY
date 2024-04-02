'''
Date: 2024-04-02 09:36:21
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-04-02 10:18:13
Description: 
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mbapy.stats.reg import quadratic_reg

def test_quadratic_reg():
    # 指定生成的模拟数据具有二次关系
    n_samples = 100
    n_features = 1
    noise = 0.1
    coef = np.array([0.4, 2, 3])  # 这里设置系数，对应于 y = 1*x^2 + 2*x + 3 + noise

    # 创建包含二次关系的数据
    X = np.random.uniform(-5, 5, size=(n_samples, n_features))
    X_poly = np.hstack((X ** 2, X))  # 添加二次项和一次项
    y_true = np.dot(X_poly, coef[:2]) + coef[2] + noise * np.random.normal(size=n_samples)

    df = pd.DataFrame({"x": X[:, 0], "y": y_true})

    # 使用quadratic_reg函数拟合数据
    result = quadratic_reg("x", "y", df)

    # 提取模型参数
    regressor, polynomial_features = result['regressor'], result['polynomial_features']
    a, b, c, r2 = result['a'], result['b'], result['c'], result['r2']

    # 使用模型预测整个x范围内的y值
    x_grid = np.linspace(df["x"].min(), df["x"].max(), 1000).reshape(-1, 1)
    y_grid = regressor.predict(polynomial_features.transform(x_grid))

    # 绘制原始数据点和拟合曲线
    plt.scatter(df["x"], df["y"], label="Data points")
    plt.plot(x_grid, y_grid, color="red", label="Quadratic Regression")

    # 添加回归线方程和R²信息
    plt.title(f"Quadratic Regression: y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}, R² = {r2:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    
    
if __name__ == '__main__':
    test_quadratic_reg()