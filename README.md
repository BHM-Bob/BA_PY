<!--
 * @Author: BHM-Bob 2262029386@qq.com
 * @Date: 2022-10-19 22:16:22
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-10-19 16:38:17
 * @Description: 
-->

<h1 style="text-align:center;">BA_PY: Optimize Your Workflow with Python!</h1>

<p style="text-align:center;">
<a href="https://www.pepy.tech/projects/mbapy"><img src="https://static.pepy.tech/badge/mbapy" alt="Downloads" style="display:inline-block; margin-left:auto; margin-right:auto;" /></a>
<img src="https://img.shields.io/pypi/dm/mbapy" alt="Downloads" style="display:inline-block; margin-left:auto; margin-right:auto;" />
<img src="https://img.shields.io/github/downloads/BHM-Bob/BA_PY/total?label=GitHub%20all%20releases%20downloads" alt="Downloads" style="display:inline-block; margin-left:auto; margin-right:auto;" />
</p>

<p style="text-align:center;">
<a href="https://github.com/BHM-Bob/BA_PY/"><img src="https://img.shields.io/github/repo-size/BHM-Bob/BA_PY" alt="repo size" style="display:inline-block; margin-left:auto; margin-right:auto;" /></a>
<a href="https://github.com/BHM-Bob/BA_PY/"><img src="https://img.shields.io/github/languages/code-size/BHM-Bob/BA_PY" alt="code size" style="display:inline-block; margin-left:auto; margin-right:auto;" /></a>
<a href="https://github.com/BHM-Bob/BA_PY/releases"><img src="https://img.shields.io/github/v/release/BHM-Bob/BA_PY?label=GitHub%20Release" alt="GitHub release (latest by date)" style="display:inline-block; margin-left:auto; margin-right:auto;" /></a>
<a href="https://github.com/BHM-Bob/BA_PY/releases"><img src="https://img.shields.io/github/commit-activity/m/BHM-Bob/BA_PY" alt="GitHub Commit Activity" style="display:inline-block; margin-left:auto; margin-right:auto;" /></a>
<a><img src="https://img.shields.io/github/last-commit/BHM-Bob/BA_PY?label=GitHub%20Last%20Commit" alt="GitHub last commit" style="display:inline-block; margin-left:auto; margin-right:auto;" /></a>
</p>

<p style="text-align:center;">
<a href="https://pypi.org/project/mbapy/"><img src="https://img.shields.io/pypi/status/mbapy?label=PyPI%20Status" alt="PyPI Status" style="display:inline-block; margin-left:auto; margin-right:auto;" /></a>
<a href="https://pypi.org/project/mbapy/"><img src="https://img.shields.io/pypi/v/mbapy?label=PyPI%20Release" alt="PyPI" style="display:inline-block; margin-left:auto; margin-right:auto;" /></a>
<a href="https://pypi.org/project/mbapy/"><img src="https://img.shields.io/pypi/pyversions/mbapy" alt="python versions" style="display:inline-block; margin-left:auto; margin-right:auto;" /></a>
</p>

<p style="text-align:center;">
<img alt="GitHub language count" src="https://img.shields.io/github/languages/count/BHM-Bob/BA_PY">
<a href="https://github.com/BHM-Bob/BA_PY/"><img src="https://img.shields.io/readthedocs/ba-py" alt="docs" style="display:inline-block; margin-left:auto; margin-right:auto;" /></a>
<a href="https://github.com/BHM-Bob/BA_PY/"><img src="https://img.shields.io/github/license/BHM-Bob/BA_PY" alt="license" style="display:inline-block; margin-left:auto; margin-right:auto;" /></a>
<a href="https://github.com/BHM-Bob/BA_PY/"><img src="https://codeium.com/badges/main" alt="codeium" style="display:inline-block; margin-left:auto; margin-right:auto;" /></a>
</p>

<p style="text-align:center;">
<a href="https://github.com/BHM-Bob/BA_PY/"><img src="https://camo.githubusercontent.com/c292429e232884db22e86c2ea2ea7695bc49dc4ae13344003a95879eeb7425d8/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f57696e646f77732d3030373844363f7374796c653d666f722d7468652d6261646765266c6f676f3d77696e646f7773266c6f676f436f6c6f723d7768697465" alt="windows" style="display:inline-block; margin-left:auto; margin-right:auto;" /></a>
<a href="https://github.com/BHM-Bob/BA_PY/"><img src="https://camo.githubusercontent.com/7eefb2ba052806d8a9ce69863c2eeb3b03cd5935ead7bd2e9245ae2e705a1adf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c696e75782d4643433632343f7374796c653d666f722d7468652d6261646765266c6f676f3d6c696e7578266c6f676f436f6c6f723d626c61636b" alt="linux" style="display:inline-block; margin-left:auto; margin-right:auto;" /></a>
</p>


mbapy is a Python package that includes a collection of useful Python scripts as sub-modules, and it's goal is *Basic for All in Python*.  
mbapy primarily focus on data works, including data-retrieval, data-management, data-visualization, data-analysis and data-computation. It is built for both python-users and command-line-users.

<h2 style="text-align:center;">Get start</h2>

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

#### docs
The documentation for mbapy can be found on [read the docs](https://ba-py.readthedocs.io/en/latest/) or just in the `docs` folder.

#### web sites
- open source at:  
    1. [github： https://github.com/BHM-Bob/BA_PY](https://github.com/BHM-Bob/BA_PY)  
    2. [gitee： https://gitee.com/BHM-Bob/BA_PY](https://gitee.com/BHM-Bob/BA_PY)  
    3. [SourceForge： https://sourceforge.net/projects/ba-py/](https://sourceforge.net/projects/ba-py/)
- docs at: [read the docs: https://ba-py.rtfd.io](https://ba-py.readthedocs.io/en/latest/)  
- PyPI: [https://pypi.org/project/mbapy/](https://pypi.org/project/mbapy/)  

<h2 style="text-align:center;">Contents</h2>

# mbapy python package  
### \_\_version\_\_  
*some version info*  

### base  
*some utils for easier coding*

### file
##### image
*imgae utils*, including reading, saving and process a image into a feature tensor via pytorch.  
##### video
*video utils*, including extract frames or unique frames from a video.  

### plot
*pandas.dataFrame utils for plot and some simple plot based on plt*  

### web
*utils for web-crawler*  
##### request
*get a web hyml page or a selenium browser warpper for easier usage*.  
##### parse
*utils for parsing html*  
##### task
*small task manager*  
##### spider
*a light-weight web spider architecture*  

### stats
##### cluster
*BAKmeans, KOptim, KBayesian from KMeans, and a func for many cluster*  
##### df
*pandas.dataFrame utils for stats*  
##### reg
*regression*  
##### test
*some test func(using scipy and mostly give a support for mbapy-style data input)*  

### dl-torch
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

### paper
*sci-paper utils, contains paper searching, downloading and parsing*  
##### paper_search
*search papers via pubmed, baidu xueshu, wos*  
##### paper_download
*download papers via scihub*  
##### paper_parse
*parse paper from a pdf file into a dict of each sections* 

### bio
##### peptide
*class and funcs to calcu peptide MW, mutations*
##### high_level
*some high-level utils for bio*

### sci_instrument
##### hplc
*HPLC instrument data processing and visualization*
##### mass
*mass spectrometry instrument data processing and visualization*

#### scripts
*some useful scripts for command user*  
launch by `python -m mbapy.scripts.XXX` or `mbapy-cli XXX`.  

## examples
#### web/crawler
1. chaoxin ppt multi threads downloader (jpg->pdf)
2. wujin search http://www.basechem.org
3. chemSub search http://chemsub.online.fr
4. cnipa https://pss-system.cponline.cnipa.gov.cn/seniorSearch

## Additional Info
### Requirements
1. mbapy requires python 3.8~3.11 because of the use of type hint and require matplotlib>=3.7.5, and the developer do not test it on other python version.  
2. mbapy only requires a part of third-party packages in a specific version. This is because the developer do not want to make a big change during the installation. Bellow are the specific requirements:  
- `matplotlib>=3.7.5`: HPLC and Mass data visualization need set legend `draggable`, this is only supported in 3.7+  
- `seaborn>=0.13.0`: plot_utils.bar_utils.plot_bar need set seaborn stripplot `native_scale`, this is only supported in 0.13+  
- `nicegui[highcharts]`: scripts/hplc: explore-hplc need a highcharts as interactive plot for manual peaking.  
- `torch any`: though dl_torch is important for mbapy, but the developer kowns torch is a big package, and do not has a specific function requirement.  
