'''
Date: 2023-07-07 20:51:46
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-07-17 21:18:03
FilePath: \BA_PY\mbapy\paper.py
Description: 
'''
import os
import re
from typing import Dict, List

import requests
import rispy

if __name__ == '__main__':
    # dev mode
    import mbapy.web as web
    from mbapy.base import *
    from mbapy.file import (convert_pdf_to_txt, opts_file, read_text,
                            replace_invalid_path_chr)
    from mbapy.sci_utils.paper_download import *
    from mbapy.sci_utils.paper_parse import *
    # Assembly of functions
    from mbapy.sci_utils.paper_search import *
else:
    # release mode
    from . import web
    from .base import *
    from .file import convert_pdf_to_txt, opts_file, replace_invalid_path_chr
    from .sci_utils.paper_download import *
    from .sci_utils.paper_parse import *
    # Assembly of functions
    from .sci_utils.paper_search import *


session = requests.Session()

@parameter_checker(check_parameters_path, raise_err = False)
def parse_ris(ris_path:str, fill_none_doi:str = None):
    """
    Parses a RIS file and returns the contents as a list of dictionaries.

    Parameters:
        ris_path (str): The path to the RIS file.
        fill_none_doi (str, optional): The DOI value to fill in for missing entries. Defaults to None.

    Returns:
        list: A list of dictionaries containing the parsed contents of the RIS file.
    """
    with open(ris_path, 'r', encoding='utf-8') as bibliography_file:
        ris = rispy.load(bibliography_file)
        if fill_none_doi is not None:
            for r in ris:
                if 'doi' not in r:
                    r['doi'] = fill_none_doi
        return ris
    

if __name__ == '__main__':
    # dev code
    from mbapy.base import rand_choose
    from mbapy.file import convert_pdf_to_txt, read_json

    # RIS parse
    ris = parse_ris('./data_tmp/savedrecs.ris', '')
    ris = rand_choose(ris)
    print(f'title: {ris["title"]}\ndoi: {ris["doi"]}')
    
    # search
    search_result_bd = search_by_baidu('linaclotide', 11)
    search_result_pm = search_by_pubmed('linaclotide', read_json('./data_tmp/id.json')['edu_email'], 11)
    search_result_wos = search_by_wos("linaclotide", 11, browser_driver_path=web.CHROMEDRIVERPATH)
    
    # download
    dl_result = download_by_scihub('./data_tmp/', title = search_result_bd[0]['title'])
    download_by_scihub('./data_tmp/', search_result_bd[0]['doi'], ris["title"], file_full_name = f'{ris["title"]:s}.pdf')
    
    # extract section
    pdf_path = replace_invalid_path_chr("./data_tmp/{:s}.pdf".format(ris["title"]))
    sections = get_english_part_of_bookmarks(get_section_bookmarks(pdf_path))
    paper, section = convert_pdf_to_txt(pdf_path), rand_choose(sections, 0)
    print(sections, section, get_section_from_paper(paper, section, keys=sections))
