Welcome to the BA_PY wiki!

Install
=======
This is a PyPI package, you can install it by:

.. code-block:: bash

    pip install mbapy

By the way, this package may have a hot-fix for the latest release, so you may try this often:

.. code-block:: bash

    pip install --upgrade mbapy

Contains
========
base
----
Some global utils in this package, most of them are also for users.

web
---
Some web-crawlers utils, small warps for BeautifulSoup and selenium.

file
----
Some file tools, mainly contains json and excel.

plot
----
Some plot tools, mainly contains pandas.dataFrame tools for plot and some plot utils.

stats
-----
Some stats functions, most of them are just import form scipy and warps for scipy to make data transformation same as mabpy style.

dl_torch
--------
Some pytorch utils and some models.

paper
-----
Some scientific paper utils, including parsing RIS, downloading pdf by title or doi through sci-hub, search through baidu-xueshu, extract specific section from a sci-paper and so on.