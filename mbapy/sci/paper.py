'''
Date: 2023-07-07 20:51:46
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-07-07 22:07:15
FilePath: \BA_PY\mbapy\sci\paper.py
Description: 
'''
import os
from scihub_cn.scihub import SciHub

if __name__ == '__main__':
    # dev mode
    from mbapy.base import put_err
else:
    # release mode
    from ..base import put_err

scihub = SciHub()

def download_by_scihub(doi: str, dir: str, use_title_as_name: bool = True):
    """
    Download a paper from Sci-Hub using its DOI.

    Parameters:
        doi (str): The DOI of the paper.
        dir (str): The directory where the downloaded paper will be saved.
        use_title_as_name (bool, optional): Whether to use the paper's title as the file name. Defaults to True.

    Returns:
        dict or None: If successful, returns a dictionary containing information about the paper. 
                      If there is an error, returns an error message. If the download fails, returns None.
    """
    res, paper_info = scihub.fetch({'doi':doi})
    file_name = (paper_info.title if use_title_as_name else doi) + '.pdf'

    if type(res) == dict and 'err' in res:        
        return put_err(res['err'])
    if not res:
        return None
    scihub._save(res.content, os.path.join(dir, file_name))
    return paper_info

if __name__ == '__main__':
    # dev code
    download_by_scihub('10.3389/fpls.2018.00696', 'E:/')