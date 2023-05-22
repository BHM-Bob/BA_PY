<!--
 * @Author: BHM-Bob 2262029386@qq.com
 * @Date: 2022-10-19 22:16:22
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2023-05-22 16:19:05
 * @Description: 
-->
# BA_PY
[![Downloads](https://static.pepy.tech/badge/mbapy)](https://pepy.tech/project/mbapy) ![PyPI - Downloads](https://img.shields.io/pypi/dm/mbapy) ![GitHub all releases](https://img.shields.io/github/downloads/BHM-Bob/BA_PY/total?label=GitHub%20all%20releases%20downloads)

![GitHub](https://img.shields.io/github/license/BHM-Bob/BA_PY)

some helpful python scripts. (Basic for All in Python)
Mainly contains sci-plot, stats, web-crawler and deeplearing-torch.

# contain  
## mbapy python package  
#### \_\_version\_\_  
*some version info*
#### base  
1. TimeCosts: a Wrapper to test cost time
2. rand_choose_times: a func
3. put_err: a func to print err info
4. MyArgs: a class to process **kwargs
5. get_wanted_args: a func to create MyArgs from defalut_args and kwargs
6. autoparse: a Wrapper to parse args for class __init__

#### file
1. detect_byte_coding: decode bytes depending it's encoding
2. save_json: func to save obj as json
3. read_json: func to read a json file
4. save_excel: func to save list\[list\[str]] as xlsx
5. read_excel: func to read xlsx
6. update_excel : update a excel(xlsx) file with multi sheets

#### plot
*pandas.dataFrame utils for plot and some simple plot based on plt*
1. pro_bar_data: func to calcu bar data as mean value and SE value
2. pro_bar_data_R: wrapper to calcu bar data by a usr-define func to process value
3. sort_df_factors: func
4. plot_bar: func to create a stack bar plot with hue style
5. get_df_data: func to make extracting data from dataFrame more easily based on df.loc
6. plot_positional_hue: wrapper to create a pos-y plot with hue style
7. qqplot: func to plot a qq-plot using statsmodels
8. save_show: func for save and show plt fig
9. get_palette: func, get a seq of hex colors

#### web
*utils for web-crawler*
1. get_url_page: func for get a html-page
2. get_url_page_s: skip error with get_url_page
3. get_url_page: func for parsing a html-page with BeautifulSoup
4. get_url_page: func for parsing a html-page with BeautifulSoup while getting the page through selenium
5. get_between: func for getting a sub-str
6. get_between_re: func for getting a sub-str through re
7. send_browser_key: func for sending a key to selenium browser
8. click_browser: func for making a click in selenium browser
9. ThreadsPool: a tiny threads pool for web-crawler task

#### stats
1. pca : func, wrap of sklearn.decomposition.PCA
##### df
*pandas.dataFrame utils for stats*
1. remove_simi: func for remove similar data in a pandas.Series
2. interp: func to make two pandas.Series the same length using numpy.interp

##### reg
*regression*
1. linear_reg: do linear regression using sklearn.linear_model.LinearRegression

##### test
*some test func(using scipy and mostly give a support for mbapy-style data input)*
1. get_interval: func to get interval
2. _get_x1_x2: inner tool func: get x1 and x2 from a mbapy-style data input
3. _get_x1_x2_R: inner tool func: get x1 and ... from a mbapy-style data input
4. ttest_1samp: scipy.stats.ttest_1samp
5. ttest_ind: func to make scipy.stats.ttest_ind with scipy.stats.levene
6. ttest_rel: scipy.stats.ttest_rel
7. mannwhitneyu: scipy.stats.mannwhitneyu
8. shapiro: scipy.stats.shapiro
9. pearsonr: scipy.stats.pearsonr
10. _get_observe: inner tool func: get observe table from a mbapy-style data input
11. chi2_contingency: scipy.stats.chi2_contingency
12. fisher_exact: scipy.stats.fisher_exact
13. f_oneway: scipy.stats.f_oneway
14. multicomp_turkeyHSD: do multicomp(turkeyHSD) using statsmodels(pairwise_tukeyhsd)
15. multicomp_dunnett: do multicomp(dunnett) using scipy.stats.dunnett
16. multicomp_bonferroni: do multicomp(bonferroni) using scikit_posthocs

#### dl-torch
*pytorch utils for deeplearning*
##### bb
*basic blocks : tiny network structures*
##### data
*utils for dataset loading*
1. denarmalize_img: func,  denarmalize a tensor type img.
2. DataSetRAM: a class to load dataset to memory with some options.
##### loss
*some loss function*
1. AsymmetricLossOptimized, just from Alibaba-MIIL/ASL
##### m
*model : deeplearning model constructed with basic blocks*
##### utils
*deeplearning training utils*
1. Mprint: logging tools
2. GlobalSettings: global setting tools for deeplearning pipeline
3. init_model_parameter: func, initialize model weigths
4. adjust_learning_rate: from MoCo
5. format_secs: func, format secs from a sum secs to h,m,s
6. AverageMeter: from MoCo
7. ProgressMeter: from MoCo
8. TimeLast: calcu time last
9. save_checkpoint: func, save checkpoint to a pytorch style file
10. resume: func, load checkpoint from a pytorch style file
11. VizLine: func, draw a line by visdom
12. re_viz_from_json_record: load visualization data and re-draw them using visdom
##### paper
1. NonLocalBlock: Non-local Neural Networks (CVPR 2018)
2. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, just import flash_attn package
3. HydraAttention: Efficient Attention with Many Heads, arXiv:2209.07484v1


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
3. chemSub search http://chemsub.online.fr/

