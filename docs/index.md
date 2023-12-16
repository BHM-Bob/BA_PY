<!-- mkdocs serve -->

Welcome to the BA_PY docs!

# BA_PY
[![Downloads](https://static.pepy.tech/badge/mbapy)](https://pepy.tech/project/mbapy) ![PyPI - Downloads](https://img.shields.io/pypi/dm/mbapy) ![GitHub all releases](https://img.shields.io/github/downloads/BHM-Bob/BA_PY/total?label=GitHub%20all%20releases%20downloads)

![GitHub repo size](https://img.shields.io/github/repo-size/BHM-Bob/BA_PY) ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/BHM-Bob/BA_PY) [![GitHub Commit Activity](https://img.shields.io/github/commit-activity/m/BHM-Bob/BA_PY)](https://github.com/BHM-Bob/BA_PY/pulse)

![PyPI - Status](https://img.shields.io/pypi/status/mbapy?label=PyPI%20Status) ![PyPI](https://img.shields.io/pypi/v/mbapy) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mbapy)

![Read the Docs](https://img.shields.io/readthedocs/ba-py) ![GitHub](https://img.shields.io/github/license/BHM-Bob/BA_PY) [![built with Codeium](https://codeium.com/badges/main)](https://codeium.com)

mbapy is a Python package that includes a collection of useful Python scripts as sub-modules, and it's goal is *Basic for All in Python*.  
mbapy primarily focus on areas such as sci-plot, stats, web-crawler, sci-paper utilities, and deep learning with pytorch.  

## get start

### install 
Now, mbapy only support pypi install:  
```
pip install mbapy
```

If you find the latest release version has some problems, you can try install the up-to-date version on github:
```
pip install git+https://github.com/BHM-Bob/BA_PY.git
```

# Contains
## [base](base.md)
Some global utils in this package, most of them are also for users.  
## web
Some web-crawlers utils, mainly contains [request](web_utils/request.md) for requesting html, [parse](web_utils/parse.md) for parsing html and [task](web_utils/task.md) for managing task.  
## [file](file.md)
Smoe file tools, mainly contains json, excel, [video](file_utils/video.md) and [image](file_utils/image.md) utils.  
## [plot](plot.md)
Some plot tools, mainly contains pandas.dataFrame tools for plot and some plot utils.  
## [stats](stats.md)
Some stats functions, most of them are just import form scipy and warps for scipy to make data transformation same as mabpy style.  
Incude [df](stats_utils/df.md) for data frame utils, [reg](stats_utils/reg.md) for regression utils and [test](stats_utils/test.md) for stats test utils.  
## dl_torch
Some pytorch utils and some models.  
1. [utils](dl_torch/utils.md) for deep learning training pepline construction utils.  
2. [basic blocks](dl_torch/basic_blocks.md) for some cnn blocks and transformer layers.  
3. [model](dl_torch/model.md) for some visual and language layers and models.  
4. [data](dl_torch/data.md) for some data utils.  
5. [optim](dl_torch/optim.md) for some learning rate utils.  
## [paper](paper.md)
Some scientific paper utils, including parsing RIS, downloading pdf by title or doi through sci-hub, search through baidu-xueshu, extract specific section from a sci-paper and so on.  
1. [parse](sci_utils/paper_parse.md) for parsing pdf and extract specific sections.  
2. [download](sci_utils/paper_download.md) for downloading pdf by title or doi through sci-hub.  
3. [search](sci_utils/paper_search.md) for searching through baidu-xueshu, pubmed and wos.
## [scripts](scripts.md)
Some helpful command line scripts.  
1. [cnipa](scripts.md#cnipa)  
2. [extract_paper](scripts.md#extract_paper)  
3. [scihub_selenium](scripts.md#scihub_selenium)  
4. [scihub](scripts.md#scihub)  
