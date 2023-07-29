<!-- mkdocs serve -->

Welcome to the BA_PY docs!

# Install
This is a PyPI package, you can install it by
```
pip install mbapy
```
By the way, this package may have a hot-fix for the latest release, so you may try this often
``` 
pip install --upgrade mbapy
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
## paper
Some scientific paper utils, including parsing RIS, downloading pdf by title or doi through sci-hub, search through baidu-xueshu, extract specific section from a sci-paper and so on.  
