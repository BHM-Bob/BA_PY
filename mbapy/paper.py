'''
Date: 2023-07-07 20:51:46
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-11-04 19:15:35
FilePath: \BA_PY\mbapy\paper.py
Description: 
'''
from typing import Dict, List

import requests
import rispy

if __name__ == '__main__':
    # dev mode
    from mbapy.base import check_parameters_path, parameter_checker
    # Assembly of functions
    from mbapy.sci_utils.paper_download import *
    from mbapy.sci_utils.paper_parse import *
    from mbapy.sci_utils.paper_search import *
else:
    # release mode
    from .base import check_parameters_path, parameter_checker
    # Assembly of functions
    from .sci_utils.paper_download import *
    from .sci_utils.paper_parse import *
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
    
    
__all__ = [
    'parse_ris',
    
    'session',
    '_get_available_scihub_urls',
    '_update_available_scihub_urls',
    'get_clean_doi',
    '_get_scihub_valid_download_link',
    'download_from_scihub_by_doi',
    'download_from_scihub_by_title',
    'download_by_scihub',
    
    'has_sci_bookmarks',
    'get_sci_bookmarks_from_pdf',
    'get_section_bookmarks',
    'get_english_part_of_bookmarks',
    'get_section_from_paper',
    'format_paper_from_txt',
    'parse_grobid',
    
    'search_IF',
    'search_by_baidu',
    'search_by_pubmed',
    'search_by_wos',
    'search',
    'get_reference_by_doi',
]  


if __name__ == '__main__':
    # dev code
    from mbapy.base import rand_choose
    from mbapy.file import (convert_pdf_to_txt, read_json,
                            replace_invalid_path_chr)

    # RIS parse
    ris = parse_ris('./data_tmp/savedrecs.ris', '')
    ris = rand_choose(ris)
    print(f'title: {ris["title"]}\ndoi: {ris["doi"]}')
    
    # search
    search_result_bd = search_by_baidu('linaclotide', 11)
    search_result_pm = search_by_pubmed('linaclotide', read_json('./data_tmp/id.json')['edu_email'], 11)
    
    # download
    dl_result = download_by_scihub('./data_tmp/', title = search_result_bd[0]['title'])
    download_by_scihub('./data_tmp/', search_result_bd[0]['doi'], ris["title"], file_full_name = f'{ris["title"]:s}.pdf')
    
    # extract section
    pdf_path = replace_invalid_path_chr("./data_tmp/{:s}.pdf".format(ris["title"]))
    sections = get_english_part_of_bookmarks(get_section_bookmarks(pdf_path))
    paper, section = convert_pdf_to_txt(pdf_path), rand_choose(sections, 0)
    print(sections, section, get_section_from_paper(paper, section, keys=sections))
