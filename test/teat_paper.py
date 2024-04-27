'''
Date: 2023-10-18 22:21:51
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-01-09 22:49:30
Description: 
'''

from mbapy.file import replace_invalid_path_chr
from mbapy.paper import *

if __name__ == '__main__':
    # dev code
    from mbapy.base import rand_choose
    from mbapy.file import convert_pdf_to_txt, read_json

    # RIS parse
    ris = parse_ris('./data_tmp/savedrecs.ris', '')
    ris = rand_choose(ris)
    print(f'title: {ris["title"]}\ndoi: {ris["doi"]}')
    
    # search
    # search_result = search_by_baidu('linaclotide', 11)
    search_result = search_by_pubmed('linaclotide', read_json('./data_tmp/id.json')['edu_email'], 11)
    search_result2 = search(ris["title"])
    
    # download
    dl_result = download_by_scihub('./data_tmp/', title = search_result[0]['title'])
    download_by_scihub('./data_tmp/', '10.1097/j.pain.0000000000001905', ris["title"], file_full_name = f'{ris["title"]:s}.pdf')
    
    # extract section
    pdf_path = replace_invalid_path_chr("./data_tmp/{:s}.pdf".format(ris["title"]))
    sections = get_english_part_of_bookmarks(get_section_bookmarks(pdf_path))
    paper, section = convert_pdf_to_txt(pdf_path), rand_choose(sections, 0)
    print(sections, section, get_section_from_paper(paper, section, keys=sections))