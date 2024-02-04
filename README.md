<!--
 * @Author: BHM-Bob 2262029386@qq.com
 * @Date: 2022-10-19 22:16:22
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2023-10-05 19:26:42
 * @Description: 
-->
# BA_PY
[![Downloads](https://static.pepy.tech/badge/mbapy)](https://pepy.tech/project/mbapy) ![PyPI - Downloads](https://img.shields.io/pypi/dm/mbapy) ![GitHub all releases](https://img.shields.io/github/downloads/BHM-Bob/BA_PY/total?label=GitHub%20all%20releases%20downloads)

![GitHub repo size](https://img.shields.io/github/repo-size/BHM-Bob/BA_PY) ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/BHM-Bob/BA_PY) [![GitHub Commit Activity](https://img.shields.io/github/commit-activity/m/BHM-Bob/BA_PY)](https://github.com/BHM-Bob/BA_PY/pulse)

![PyPI - Status](https://img.shields.io/pypi/status/mbapy?label=PyPI%20Status) ![PyPI](https://img.shields.io/pypi/v/mbapy) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mbapy)

![Read the Docs](https://img.shields.io/readthedocs/ba-py) ![GitHub](https://img.shields.io/github/license/BHM-Bob/BA_PY) [![built with Codeium](https://codeium.com/badges/main)](https://codeium.com)

![platform - WINDOWS](https://camo.githubusercontent.com/c292429e232884db22e86c2ea2ea7695bc49dc4ae13344003a95879eeb7425d8/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f57696e646f77732d3030373844363f7374796c653d666f722d7468652d6261646765266c6f676f3d77696e646f7773266c6f676f436f6c6f723d7768697465) ![platform - LINUX](https://camo.githubusercontent.com/7eefb2ba052806d8a9ce69863c2eeb3b03cd5935ead7bd2e9245ae2e705a1adf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c696e75782d4643433632343f7374796c653d666f722d7468652d6261646765266c6f676f3d6c696e7578266c6f676f436f6c6f723d626c61636b)

mbapy is a Python package that includes a collection of useful Python scripts as sub-modules, and it's goal is *Basic for All in Python*.  
mbapy primarily focus on areas such as sci-plot, stats, web-crawler, sci-paper utilities, and deep learning with pytorch.  

## get start

#### install 
Now, mbapy only support pypi install:  
```
pip install mbapy
```

mbapy is a multi-funtional package, and it does not require every third-party packages to make every sub-module work. However, it provides some requriements option to install more specified requirements to make some sub-modules work:  
1. bio: some packages for biology(sci).  
    install as `pip install mbapy[bio]`  
2. full: full requirements to make almost every sub-module in mbapy work(except dl_torch).  
     install as `pip install mbapy[full]`  

If you find the latest release version has some problems, you can try install the up-to-date version on github or gitee:  
```
pip install git+https://github.com/BHM-Bob/BA_PY.git
```
```
pip install git+https://gitee.com/BHM-Bob/BA_PY.git
```

#### docs
The documentation for mbapy can be found on [read the docs](https://ba-py.readthedocs.io/en/latest/), and it is the one that I will regularly update.
The API document for mbapy is available on its [wiki](https://github.com/BHM-Bob/BA_PY/wiki). However, please note that this wiki on GitHub has not been updated since 2023-07. 

#### web sites
- open source at:  
    1. [github： https://github.com/BHM-Bob/BA_PY](https://github.com/BHM-Bob/BA_PY)  
    2. [gitee： https://gitee.com/BHM-Bob/BA_PY](https://gitee.com/BHM-Bob/BA_PY)  
- docs at: [read the docs: https://ba-py.rtfd.io](https://ba-py.readthedocs.io/en/latest/)  

# contain  
## mbapy python package  
#### \_\_version\_\_  
*some version info*  
#### base  
*some utils for easier coding*

#### file
##### image
*imgae utils*, including reading, saving and process a image into a feature tensor via pytorch.  
##### video
*video utils*, including extract frames or unique frames from a video.  

#### plot
*pandas.dataFrame utils for plot and some simple plot based on plt*  

#### web
*utils for web-crawler*  
##### request
*get a web hyml page or a selenium browser warpper for easier usage*.  
##### parse
*utils for parsing html*  
##### task
*small task manager*  

#### stats
##### cluster
*BAKmeans, KOptim, KBayesian from KMeans, and a func for many cluster*  
##### df
*pandas.dataFrame utils for stats*  
##### reg
*regression*  
##### test
*some test func(using scipy and mostly give a support for mbapy-style data input)*  

#### dl-torch
*pytorch utils for deeplearning*  
##### bb
*basic blocks : tiny network structures*  
##### data
*utils for dataset loading*  
##### loss
*some loss function*  
##### m
*model : deeplearning model constructed with basic blocks*  
##### utils
*deeplearning training utils*  
##### optim
*learning rate scheduler*

#### paper
*sci-paper utils, contains paper searching, downloading and parsing*  
##### paper_search
*search papers via pubmed, baidu xueshu, wos*  
##### paper_download
*download papers via scihub*  
##### paper_parse
*parse paper from a pdf file into a dict of each sections* 

#### scripts
*some useful scripts for command user*  
launch by `python -m mbapy.scripts.XXX` or `mbapy-cli XXX`.  

## examples
#### file
*some file utils things*  

#### plot
1. stack bar plot with hue style  

#### web/crawler
1. chaoxin ppt multi threads downloader (jpg->pdf)
2. wujin search http://www.basechem.org
3. chemSub search http://chemsub.online.fr
4. cnipa https://pss-system.cponline.cnipa.gov.cn/seniorSearch

