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

mbapy is a Python package that includes a collection of useful Python scripts as sub-modules, and it's goal is *Basic for All in Python*.  
mbapy primarily focus on areas such as sci-plot, stats, web-crawler, sci-paper utilities, and deep learning with pytorch.  

## get start

#### install 
Now, mbapy only support pypi install:  
```
pip install mbapy
```

If you find the latest release version has some problems, you can try install the up-to-date version on github:
```
pip install git+https://github.com/BHM-Bob/BA_PY.git
```

#### docs
The documentation for mbapy can be found on [read the docs](https://ba-py.readthedocs.io/en/latest/), and it is the one that I will regularly update.
The API document for mbapy is available on its [wiki](https://github.com/BHM-Bob/BA_PY/wiki). However, please note that this wiki on GitHub has not been updated since 2023-07. 
*Given my limited time, I heavily rely on chatGPT to generate the documentation*.

# contain  
## mbapy python package  
#### \_\_version\_\_  
*some version info*
#### base  

#### file

#### plot
*pandas.dataFrame utils for plot and some simple plot based on plt*

#### web
*utils for web-crawler*

#### stats

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
##### paper


## examples
#### file
*some file utils things*

#### plot
1. stack bar plot with hue style

#### torch
1. seq2seq core from bentrevett/pytorch-seq2seq

#### web/crawler
1. chaoxin ppt multi threads downloader (jpg->pdf)
2. wujin search http://www.basechem.org
3. chemSub search http://chemsub.online.fr

