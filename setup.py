'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-11-01 18:30:01
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-12-04 21:22:17
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
requirements = (here / "requirements.txt").read_text(encoding="utf-8").split('\n')
version_info = (here / "mbapy/__version__.py").read_text(encoding="utf-8")
for line in version_info.split('\n'):
    if '__version__' in line:
        __version__ = line[line.find('"')+1:-1]
    if '__author_email__' in line:
        __author_email__ = line[line.find('"')+1:-1]
    if '__author__' in line:
        __author__ = line[line.find('"')+1:-1]
    if '__url__' in line:
        __url__ = line[line.find('"')+1:-1]

setup(
    name = "mbapy",
    version = __version__,

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
        "Programming Language :: Python :: 3.11",
        # "Programming Language :: Python :: 3 :: Only",
    ],
        
    keywords = ["mbapy", "Utilities", "plot"],
    description = "MyBA in Python",
    long_description = long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.7, <3.12",
    license = "MIT Licence",

    url = __url__,
    author = __author__,
    author_email = __author_email__,
    
    packages = find_packages(exclude=["test"]),
    
    include_package_data = True, # define in MANIFEST.in file
    platforms = "any",
    
    install_requires=requirements,
)

# pip install .

# python setup.py sdist
# twine upload dist/mbapy-0.5.0.tar.gz