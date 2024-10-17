from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colormaps
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerLine2D, HandlerPathCollection
from matplotlib.lines import Line2D

if __name__ == '__main__':
    # dev mode
    import mbapy_lite.base as mb
    import mbapy_lite.stats.df as msd
    import mbapy_lite.stats.reg as msr
    import mbapy_lite.stats.test as mst
else:
    # release mode
    from .. import base as mb
    from ..stats import df as msd
    from ..stats import reg as msr
    from ..stats import test as mst


@mb.parameter_checker(method = lambda arg: arg in ['line', 'quad', 'sns'])
def plot_reg(X: str, Y: str, df: pd.DataFrame,
             ax: plt.Axes = None, figsize: Tuple[float, float] = (11, 6),
             method: str = 'line', ci: int = 95, order: int = 1,
             color: str = 'black', linewidth: float = 2,):
    """
    Parameters:
        - X(str), Y(str), df(pd.DataFrame): X-axis variable in df, Y-axis variable in df, and dataFrame.
        - ax: plt.Axes, axis to plot the scatter plot.
        - figsize: Tuple[float, float], size of the figure.
        - method: str, method to perform regression, only support 'line', 'quad', and 'sns'.
        - ci: int, confidence interval for the regression line, only works for 'sns' method.
        - order: int, order of the polynomial for the quadratic regression, only works for 'sns' method.
        - color: str, color of the regression line.
        - linewidth: float, width of the regression line.
    
    Returns:
        - returns: dict, containing the following keys:
            - ax: plt.Axes, axis to plot the scatter plot.
            - result: dict, regression result, only works for 'line' and 'quad' method, or order is 1 or 2 for 'sns' method.
            - line: Line2D, regression line, only works for 'line' and 'quad' method, or order is 1 or 2 for 'sns' method.
    """
    returns = {}
    # set figure size
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    returns['ax'] = ax
    # perform regression
    x_grid = np.linspace(df[X].min(), df[X].max(), 1000).reshape(-1, 1)
    if method == 'line' or (method =='sns' and order == 1):
        reg_result = msr.linear_reg(X, Y, df)
        y_grid = reg_result['regressor'].predict(x_grid)
    elif method == 'quad' or (method =='sns' and order == 2):
        reg_result = msr.quadratic_reg(X, Y, df)
        polynomial_features = reg_result['polynomial_features']
        y_grid = reg_result['regressor'].predict(
            polynomial_features.transform(x_grid))
    elif method == 'sns' and order > 2:
        reg_result = None
    # plot
    if method in ['line', 'quad']:
        reg_line = ax.plot(x_grid, y_grid, color=color, linewidth=linewidth)[0]
    else:
        ax = sns.regplot(x = X, y = Y, data = df, scatter = False,
                         color = color, truncate = False,
                         ci = ci, order = order, ax = ax)
        reg_line = ax.lines[-1]
    # return
    returns['result'] = reg_result
    returns['line'] = reg_line
    return returns


def plot_scatter(X: str, Y: str, df: pd.DataFrame, label: str = None,
                 size: Union[str, np.ndarray] = None,
                 color: Union[str, np.ndarray] = 'black',
                 cmap = None, cmap_label: str = None,
                 cmap_lim: Tuple[float, float] = (None, None), cmap_tick_fn = lambda x: f'{x:.1f}',
                 cmap_anchor: Tuple[float, float] = (0.0, 0.5),
                 alpha = 1, marker: str = 'o',
                 ax: plt.Axes = None, figsize: Tuple[float, float] = (10, 6),
                 ticks_fontsize: int = 14, label_fontsize: int = 15, axis_fontsize: int = 20,
                 reg: bool = False, reg_kwargs: dict = None, ):
    """
    Parameters:
        - X(str), Y(str), df(pd.DataFrame): X-axis variable in df, Y-axis variable in df, and dataFrame.
        - label: str, label for the scatter plot
        - size: Union[str, np.ndarray], size of the scatter plot, can be a column name in df or a numpy array
        - color: Union[str, np.ndarray], color of the scatter plot, can be a column name in df or a numpy array
        - cmap: plt.Colormap, colormap for the scatter plot.
        - cmap_label: str, label for the colormap.
        - cmap_lim: Tuple[float, float], limits for the colormap.
        - cmap_tick_fn: Callable, function to format the ticks of the colormap.
        - cmap_anchor: Tuple[float, float], anchor point for the colorbar.
        - alpha: float, transparency of the scatter plot
        - marker: str, marker style of the scatter plot, such as 'o', 'v', 'D', 's', etc.
        - ax: plt.Axes, axes for the plot. If None, a new figure and axes will be created as default.
        - figsize: Tuple[float, float], size of the plot, only used when ax is None.
        - reg: bool, whether to perform linear regression.
        - reg_kwargs: dict, keyword arguments for linear regression.
            - 'method': str, method for linear regression, default is 'line', and supoort 'quad', 'sns'.
            - 'col': str, color of the regression line, default is 'black'.
            - 'linewidth': float, line width of the regression line, default is 2.
    
    Returns:
        - dict of the following items:
            - 'ax': plt.Axes, axes for the plot.
            - 'fig': plt.Figure, figure for the plot. If ax is not None, fig will not exist.
            - 'scatter': plt.PathCollection, scatter plot.
            - 'reg_line': plt.Line2D, regression line. If reg is False, reg_line will not exist.
            - 'reg_result': dict, regression result. If reg is False, reg_result will not exist.
            - 'cbar': plt.Colorbar, colorbar for the scatter plot. If cmap is None, cbar will not exist.
    """
    returns = {}
    # set figure size
    if ax is None:
        returns['fig'], ax = plt.subplots(figsize=figsize)
    returns['ax'] = ax
    # set color
    if isinstance(color, str) and color in df.columns:
        color = df[color]
    # set size
    if isinstance(size, str) and size in df.columns:
        size = df[size]
    # plot scatter
    sc = ax.scatter(df[X], df[Y],
                    c = color, cmap = cmap, vmin = cmap_lim[0], vmax = cmap_lim[1],
                    marker = marker, alpha = alpha, s = size, label = label)
    returns['scatter'] = sc
    # set default reg_kwargs
    if reg and reg_kwargs is None:
        reg_kwargs = {'method': 'line', 'color': 'black', 'linewidth': 2}
    # perform regression
    if reg:
        reg_result = plot_reg(X, Y, df, ax = ax, **reg_kwargs)
        returns['reg_line'] = reg_result['line']
        returns['reg_result'] = reg_result['result']
    # add cmap label
    if cmap is not None:
        cbar = plt.colorbar(sc, ax=ax, drawedges=False, anchor = cmap_anchor)
        if isinstance(cmap_label, str):
            cbar.set_label(cmap_label, loc='top', fontsize=label_fontsize)
        ticks = cbar.get_ticks()[1:-1]
        cbar.set_ticks(ticks, labels = [cmap_tick_fn(i) for i in ticks], fontsize = ticks_fontsize)
        cbar.minorticks_on()
        returns['cbar'] = cbar
    # set style
    ax.tick_params(axis='both', labelsize=ticks_fontsize)
    plt.xlabel(X, fontsize = axis_fontsize)
    plt.ylabel(Y, fontsize = axis_fontsize)
    # return
    return returns


def add_scatter_legend(scatters: Union[PathCollection, List[PathCollection]],
                       reg_lines: Union[Line2D, List[Line2D]], reg_results: Union[dict, List[dict]],
                       ax: plt.Axes,
                       main_h_kwgs: dict = None, reg_h_kwgs: dict = None,
                       main_kwgs: dict = None, reg_kwgs: dict = None):
    """
    Parameters:
        - scatters: Union[PathCollection, List[PathCollection]], scatter plot.
        - reg_lines: Union[Line2D, List[Line2D]], regression line.
        - reg_results: Union[dict, List[dict]], regression result.
        - ax: plt.Axes, axes for the plot.
        - main_h_kwgs: dict, kwgs for main legend handles, default is {'_A':[0], numpoints=1, yoffsets=[0], sizes=[100], marker_pad=-0.5}.
        - reg_h_kwgs: dict, kwgs for regression legend handles, default is {'numpoints': 1,'marker_pad': 0.5}.
        - main_kwgs: dict, kwgs for main legend, default is {'frameon': True, 'framealpha': 0.45, 'borderpad': 1.2, 'borderaxespad': 1, 'labelspacing': 1}.
        - reg_kwgs: dict, kwgs for regression legend, defaut is {'title': "Linear Regression", 'labelspacing': 1, 'framealpha': 0.1, 'frameon': False, 'borderpad': 1.1, 'borderaxespad': 1}.
    """
    # process input
    scatters = [scatters] if isinstance(scatters, PathCollection) else scatters
    reg_lines = [reg_lines] if isinstance(reg_lines, Line2D) else reg_lines
    reg_results = [reg_results] if isinstance(reg_results, dict) else reg_results
    # add main legend
    ## set amin legend default kwgs
    d_main_h_kwgs = dict(_A=[0], numpoints=1, yoffsets=[0], sizes=[100], marker_pad=0)
    d_main_h_kwgs.update(main_h_kwgs or {})
    d_main_kwgs = dict(frameon=True, framealpha = 0.45, borderpad=1.2, borderaxespad=1, labelspacing=1)
    d_main_kwgs.update(main_kwgs or {})
    ## update func for main legend
    main_legend_h_A = d_main_h_kwgs['_A']
    del d_main_h_kwgs['_A']
    def update_func(legend_handle, orig_handle):
        legend_handle.update_from(orig_handle)
        legend_handle._A = np.array(main_legend_h_A) # 设置图例中三个scatters的cmap value
    m_handle_map = {sc:HandlerPathCollection(**d_main_h_kwgs,
                                             update_func=update_func) for sc in scatters}
    ## create main legend
    main_legend = plt.legend(handler_map=m_handle_map, **d_main_kwgs)
    ax.add_artist(main_legend)
    ## delete main label, avoid duplicate label in reg legend
    [sc.set_label(None) for sc in scatters]
    
    # add reg legend
    ## set reg legend default kwgs
    d_reg_h_kwgs = dict(numpoints=1, marker_pad=0.5)
    d_reg_h_kwgs.update(reg_h_kwgs or {})
    d_reg_kwgs = dict(title="Linear Regression", labelspacing=1, framealpha = 0.1, frameon=False, borderpad=1.1, borderaxespad=1)
    d_reg_kwgs.update(reg_kwgs or {})
    ## make reg label
    reg_labels = [f'${r["equation"]}, {r["r2_equation"]}$' for r in reg_results]
    [rl.set_label(label) for rl, label in zip(reg_lines, reg_labels)]
    r_handle_map = {rl:HandlerLine2D(**d_reg_h_kwgs) for rl in reg_lines}
    reg_legend = plt.legend(handler_map=r_handle_map, **d_reg_kwgs)
    ax.add_artist(reg_legend)
    # return ax, main_legend, reg_legend
    return ax, main_legend, reg_legend


__all__ = [
    'plot_reg',
    'plot_scatter',
    'add_scatter_legend']
    
    
if __name__ == '__main__':
    # dev code
    df = pd.read_excel('./data/plot.xlsx', sheet_name='weight')
    result = plot_scatter('day', 'weight', df, label = 'weight-day',
                          reg=True, reg_kwargs=dict(method = 'sns', order = 1, ci = 95, color = 'blue', linewidth = 2))
    add_scatter_legend(result['scatter'], result['reg_line'], result['reg_result'], result['ax'],
                       main_kwgs = dict(loc='lower right'),)
    plt.show()