'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-11-01 18:30:01
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-05-22 16:21:54
Description: 
'''
"""
something is from https://github.com/pypa/sampleproject
thanks to https://zetcode.com/python/package/
"""

"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import pathlib

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

requires = [
    "beautifulsoup4>=4.10.1",
    "bokeh>=2.3.3",
    "chardet>=5.0.0",
    "cn2an>=0.5.17",
    "holoviews>=1.13.1",
    "imageio>=2.20.2",
    "jieba>=0.42.1",
    "Markdown>=3.4.1",
    "matplotlib>=3.5.3",
    "multiprocess>=0.70.13",
    # "numpy>=1.22.1",
    "pandas>=1.4.3",
    "pathtools>=0.1.2",
    "pdfkit>=1.0.0",
    "Pillow>=9.2.0",
    "plotly>=5.10.0",
    "requests>=2.25.1",
    "scikit-learn>=1.1.2",
    "scipy>=1.5.1",
    "seaborn>=0.11.2",
    "selenium>=4.2.0",
    "urllib3>=1.22.12",
    "openpyxl",
    "statsmodels",
    "scikit_posthocs",
]

setup(
    name = "mbapy",
    version = "0.1.0",

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        # "Programming Language :: Python :: 3.11",# wait to numpy
        # "Programming Language :: Python :: 3 :: Only",
    ],
        
    keywords = ["mbapy", "Utilities", "plot"],
    description = "MyBA in Python",
    long_description = long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.7, <3.11",
    license = "MIT Licence",

    url = "https://github.com/BHM-Bob/BA_PY",
    author = "BHM-Bob G",
    author_email = "bhmfly@foxmail.com",
    
    # packages=["mbapy"],
    packages=["mbapy", "mbapy/stats", "mbapy/dl_torch", "mbapy/dl_torch/paper"],
    
    include_package_data = True,
    platforms = "any",
    
    install_requires=requires,
)

# pip install .


# python setup.py sdist
# twine upload dist/mbapy-0.0.15.tar.gz