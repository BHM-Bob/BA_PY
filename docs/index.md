<!--
 * @Date: 2023-07-29 09:56:37
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-08-31 08:50:33
 * @Description: 
-->
<!-- mkdocs serve -->

Welcome to the BA_PY docs!

# BA_PY: Optimize Your Workflow with Python!
[![Downloads](https://static.pepy.tech/badge/mbapy)](https://pepy.tech/project/mbapy) ![PyPI - Downloads](https://img.shields.io/pypi/dm/mbapy) ![GitHub all releases](https://img.shields.io/github/downloads/BHM-Bob/BA_PY/total?label=GitHub%20all%20releases%20downloads)

![GitHub repo size](https://img.shields.io/github/repo-size/BHM-Bob/BA_PY) ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/BHM-Bob/BA_PY) [![GitHub Commit Activity](https://img.shields.io/github/commit-activity/m/BHM-Bob/BA_PY)](https://github.com/BHM-Bob/BA_PY/pulse)

![PyPI - Status](https://img.shields.io/pypi/status/mbapy?label=PyPI%20Status) ![PyPI](https://img.shields.io/pypi/v/mbapy) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mbapy)

![Read the Docs](https://img.shields.io/readthedocs/ba-py) ![GitHub](https://img.shields.io/github/license/BHM-Bob/BA_PY) [![built with Codeium](https://codeium.com/badges/main)](https://codeium.com)

![platform - WINDOWS](https://camo.githubusercontent.com/c292429e232884db22e86c2ea2ea7695bc49dc4ae13344003a95879eeb7425d8/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f57696e646f77732d3030373844363f7374796c653d666f722d7468652d6261646765266c6f676f3d77696e646f7773266c6f676f436f6c6f723d7768697465) ![platform - LINUX](https://camo.githubusercontent.com/7eefb2ba052806d8a9ce69863c2eeb3b03cd5935ead7bd2e9245ae2e705a1adf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c696e75782d4643433632343f7374796c653d666f722d7468652d6261646765266c6f676f3d6c696e7578266c6f676f436f6c6f723d626c61636b)

mbapy is a Python package that includes a collection of useful Python scripts as sub-modules, and it's goal is *Basic for All in Python*.  
mbapy primarily focus on data works, including data-retrieval, data-management, data-visualization, data-analysis and data-computation. It is built for both python-users and command-line-users.

## get start

#### install 
Now, mbapy only support pypi install:  
```
pip install mbapy
```

mbapy is a multi-funtional package, and it does not require every third-party packages to make every sub-module work. However, it provides some requriements option to install more specified requirements to make some sub-modules work:  
1. bio: some packages for biology(sci).  
    install as `pip install mbapy[bio]`  
2. game: some packages for game(pygame).  
    install as `pip install mbapy[game]`  
3. full: full requirements to make almost every sub-module in mbapy work(except dl_torch).  
     install as `pip install mbapy[full]`  

If you find the latest release version has some problems, you can try install the up-to-date version on github or gitee:  
```
pip install git+https://github.com/BHM-Bob/BA_PY.git
```
```
pip install git+https://gitee.com/BHM-Bob/BA_PY.git
```

# Contains
## [base](base.md)
Some global utils in this package, most of them are also for users.  
## web
Some web-crawlers utils, mainly contains:
1. [request](web_utils/request.md) for requesting html.  
2. [parse](web_utils/parse.md) for parsing html.  
3. [task](web_utils/task.md) for managing task.  
4. [spider](web_utils/spider.md) for constructing and running a simple web spider.  
## [file](file.md)
Smoe file tools, mainly contains json, excel, [video](file_utils/video.md) and [image](file_utils/image.md) utils.  
## [plot](plot.md)
Some plot tools, mainly contains pandas.dataFrame tools for plot and some plot utils.  
## [stats](stats.md)
Some stats functions, most of them are just import form scipy and warps for scipy to make data transformation same as mabpy style.  
1. [df](stats_utils/df.md) for data frame utils.  
2. [reg](stats_utils/reg.md) for regression utils.  
3. [test](stats_utils/test.md) for stats test utils.  
## bio
1. [peptide](bio/peptide.md) for peptide utils.  
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
## sci_instrument
Some scientific instrument data file utils.  
- [_base](sci_instrument/_base.md)  
- [_utils](sci_instrument/_utils.md)  

1. [HPLC]  
- [_base](sci_instrument/hplc/_base.md)  
- [_utils](sci_instrument/hplc/_utils.md)  
- [SCIEX](sci_instrument/hplc/sciex.md)  
- [waters](sci_instrument/hplc/waters.md)  
1. [Mass]  
- [_base](sci_instrument/mass/_base.md)  
- [_utils](sci_instrument/mass/_utils.md)  
- [SCIEX](sci_instrument/mass/sciex.md)  
## [scripts](scripts.md)
Some helpful command line scripts.  
1. [avif](scripts/avif.md)  
2. [cnipa](scripts/cnipa.md)  
3. [cp](scripts/cp.md)  
4. [duitang](scripts/duitang.md)  
5. [extract_paper](script/extract_paper.md)  
6. [extract-dir](scripts/extract_dir.md)  
7. [file-size](scripts/file_size.md)  
8. [hplc](scripts/hplc.md)  
9. [mass](scripts/mass.md)  
10. [mv](scripts/mv.md)  
11. [peptide](scripts/peptide.md)  
12. [reviz](scripts/reviz.md)  
13. [rm](scripts/rm.md)  
14. [scihub_selenium](scripts/scihub_selenium.md)  
15. [scihub](scripts.md#scihub)  
16. [video](scripts/video.md)  

# Release History
- [0.9.1](release_notes/0.9.1.md)
- [0.9.0](release_notes/0.9.0.md)
- [0.8.9](release_notes/0.8.9.md)
- [0.8.8](release_notes/0.8.8.md)  
- [0.8.7](release_notes/0.8.7.md)  
- [0.8.6](release_notes/0.8.6.md)  
- [0.8.5](release_notes/0.8.5.md)  
- [0.8.4](release_notes/0.8.4.md)  
- [0.8.3](release_notes/0.8.3.md)  
- [0.8.2](release_notes/0.8.2.md)  
- [0.8.1](release_notes/0.8.1.md)  
- [0.8.0](release_notes/0.8.0.md)  
- [0.7.4](release_notes/0.7.4.md)  
- [0.7.3](release_notes/0.7.3.md)  
- [0.7.2](release_notes/0.7.2.md)  
- [0.7.1](release_notes/0.7.1.md)  
- [0.7.1b1](release_notes/0.7.1b1.md)  
- [0.6.3](release_notes/0.6.3.md)  
- [0.6.2](release_notes/0.6.2.md)  
- [0.6.1](release_notes/0.6.1.md)  
- [0.6.0](release_notes/0.6.0.md)  
- [0.5.0](release_notes/0.5.0.md)  
- [0.4.0](release_notes/0.4.0.md)  
- [0.3.0](release_notes/0.3.0.md)  
- [0.2.0](release_notes/0.2.0.md)  
- [0.1.4](release_notes/0.1.4.md)  
- [0.1.3](release_notes/0.1.3.md)  
- [0.1.2](release_notes/0.1.2.md)  
- [0.1.1](release_notes/0.1.1.md)  
- [0.1.0](release_notes/0.1.0.md)  
- [0.0.14](release_notes/0.0.14.md) 
- [0.0.12](release_notes/0.0.12.md) 
- [0.0.9](release_notes/0.0.9.md) 
- [0.0.6](release_notes/0.0.6.md) 
- [0.0.4](release_notes/0.0.4.md) 
- [0.0.3](release_notes/0.0.3.md)  
- [0.0.2](release_notes/0.0.2.md)  
- [0.0.1](release_notes/0.0.1.md)  