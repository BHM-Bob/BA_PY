'''
Date: 2023-07-07 20:51:46
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-07-12 00:17:56
FilePath: \BA_PY\mbapy\paper.py
Description: 
'''
import os, re
from typing import List, Dict

import rispy
from scihub_cn.scihub import SciHub
import PyPDF2

if __name__ == '__main__':
    # dev mode
    from mbapy.base import *
    from mbapy.file import replace_invalid_path_chr, convert_pdf_to_txt, read_text, opts_file
    import mbapy.web as web
else:
    # release mode
    from .base import *
    from .file import replace_invalid_path_chr, convert_pdf_to_txt, opts_file
    from . import web

scihub = SciHub()

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
    with open('./data_tmp/savedrecs.ris', 'r', encoding='utf-8') as bibliography_file:
        ris = rispy.load(bibliography_file)
        if fill_none_doi is not None:
            for r in ris:
                if 'doi' not in r:
                    r['doi'] = fill_none_doi
        return ris
    
@parameter_checker(check_parameters_len, raise_err = False)
def search_by_baidu(query:str, limit:int = 1):
    """
    This function is used to search for articles on Baidu Scholar using a given query.
    
    Parameters:
        query (str): The search query.
        limit (int, optional): The maximum number of articles to retrieve. Defaults to 1.
        
    Returns:
        list: A list of dictionaries containing information about the articles found.
            Each dictionary has the following keys:
            - title (str): The title of the article.
            - abstract (str): The abstract of the article.
            - keyword (list): A list of keywords associated with the article.
            - doi (str): The DOI (Digital Object Identifier) of the article.
    """
    from lxml import etree
    def _get_a_search_page(query:str, page:int = 0):
        res = scihub.sess.request(method='GET', url='https://xueshu.baidu.com/s',
                                params={'wd': query, 'pn': page, 'filter': 'sc_type%3D%7B1%7D'},
                                proxies=scihub.proxies)
        s = etree.HTML(res.text)
        return s.xpath("//div[@class='sc_content']/h3/a/@href")
    
    def _parse_links(links:list):
        results = []
        for link in links:
            res = scihub.sess.request(method='GET', url=link, proxies=scihub.proxies)
            s = etree.HTML(res.text)
            title = s.xpath("//div[@class='main-info']/h3/a/text()")
            if len(title) == 0:
                title = s.xpath("//div[@class='main-info']/h3/span/text()")
            title = get_default_for_bool(title, [''])[0].strip()
            abstract = s.xpath("//p[@class='abstract']/text()")
            abstract = get_default_for_bool(abstract, [''])[0].strip()
            keyword = s.xpath("//div[@class='kw_wr']/p[@class='kw_main']/span/a/text()")
            keyword = get_default_for_bool(keyword, [''])
            doi = s.xpath("//div[@class='doi_wr']/p[@class='kw_main']/text()")
            doi = get_default_for_bool(doi, [''])[0].strip()
            results.append({'title': title, 'abstract': abstract, 'keyword': keyword, 'doi': doi})
        return results
    
    links = _get_a_search_page(query, 0)
    if limit > 10:
        # 未登录的百度学术一页只显示10个结果
        page = 1
        while limit > 10:
            limit -= 10
            links += _get_a_search_page(query, page)[: (limit%10 if limit > 10 else limit)]
            page += 1
    return _parse_links(links)
        
def search(query:str, limit:int = 1, search_engine:str = 'publons'):
    """
    Search for a given query using a specified search engine and return the results.

    Parameters:
    - query (str): The query string to search for.
    - limit (int): The maximum number of results to return.
    - search_engine (str): The search engine to use. Default is 'baidu xueshu'.
         allows: 'baidu xueshu', 'science direct', 'publons', if not recognized, returns None

    Returns:
    - The search results as a list of dict, contain 'title', 'abstract', 'keyword' and 'doi'.
    """
    if search_engine == 'baidu xueshu':
        return search_by_baidu(query, limit)
    elif search_engine == 'science direct':
        if os.path.isfile(get_storage_path('science_direct_cookie.txt')):
            cookie = opts_file(get_storage_path('science_direct_cookie.txt'))
        else:
            import browser_cookie3
            cookie = browser_cookie3.load('https://www.sciencedirect.com/')
            opts_file(get_storage_path('science_direct_cookie.txt'), 'w', encoding='utf-8', data = cookie)
        return scihub.search_by_science_direct(query, cookie, limit)
    elif search_engine == 'publons':
        return scihub.search_by_publons([query], limit)
    else:
        return put_err(f'Unknown search engine: {search_engine}, returns None', None)

@parameter_checker(check_parameters_bool, raise_err = False)
def download_from_scihub_by_doi(doi):
    try:
        return scihub.fetch({'doi':doi})
    except:
        return put_err(f'Maybe DOI: {doi:s} does not exist. scihub fetch error', None)
            
@parameter_checker(check_parameters_bool, raise_err = False)
def download_from_scihub_by_title(title):
    scihub_url = scihub._get_available_scihub_urls()[0]
    res = scihub.sess.post(scihub_url, data = {'request': title}, proxies=scihub.proxies)
    doi = web.get_between(res.text, "doi:", "&nbsp", find_tail_from_head=True)
    # TODO : xpath do not work
    # dl = s.xpath("//div[@id='buttons']/ul/li/a[@href='#']")
    return download_from_scihub_by_doi(doi)
            
@parameter_checker(check_parameters_path, raise_err = False)
def download_by_scihub(dir: str, doi: str = None, title:str = None,
                       file_full_name:str = None, use_title_as_name: bool = True,
                       valid_path_chr:str = '_'):
    """
    Download a paper from Sci-Hub using its DOI.
    if file_full_name is None, use the paper's title as the file name, if not, use the paper's DOI as the file name.

    Args:
        doi (str): The DOI (Digital Object Identifier) of the paper.
        dir (str): The directory where the downloaded file will be saved.
        file_full_name (str, optional): The name of the downloaded file, include the file extension(.pdf). Defaults to None.
        use_title_as_name (bool, optional): Whether to use the paper's title as the file name. Defaults to True.
        valid_path_chr (str, optional): The character used to replace invalid characters in the file name. Defaults to '_'.

    Returns:
        dict or None: If successful, returns a dictionary containing information
            about the downloaded paper. If unsuccessful, returns None.
    """
    # check whether doi or title are specified
    if doi is None and title is None:
        return put_err('Either DOI or title must be specified, returns None', None)
    # download from Sci-Hub by DOI or title
    if doi:
        res, paper_info = download_from_scihub_by_doi(doi)
    else:
        res, paper_info = download_from_scihub_by_title(title)
    if type(res) == dict and 'err' in res:        
        return put_err(res['err'])
    if not res:
        return None
    # get the file name, save the file
    if file_full_name is not None:
        file_name = file_full_name
    else:
        file_name = ((title if title else paper_info.title) if use_title_as_name else doi) + '.pdf'
    file_name = replace_invalid_path_chr(file_name, valid_path_chr)
    scihub._save(res.content, os.path.join(dir, file_name))
    return paper_info
    
def _flatten_pdf_bookmarks(*bookmarks):
    """
        Parse a list of bookmarks and return a flattened list of all bookmarks.

        Args:
            *bookmarks (List[Any]): A variable number of bookmark lists.

        Returns:
            List[Any]: A flattened list of all bookmarks.
    """
    ret = []
    for bookmark in bookmarks:
        if isinstance(bookmark, list):
            ret = ret + _flatten_pdf_bookmarks(*bookmark)
        else:
            ret.append(bookmark)
    return ret

def has_sci_bookmarks(pdf_path:str = None, pdf_obj = None, section_names:List[str]=[]):
    """
    Checks if a PDF document has bookmarks for scientific sections.

    Parameters:
        pdf_obj: The PDF object(Being opened!). Defaults to None.
        pdf_path (str): The path to the PDF document. Defaults to None.
        section_names (list[str]): A list of section names to check for bookmarks. Defaults to an empty list.

    Returns:
        list[str] or bool: list of section names if the PDF has bookmarks, False otherwise.
    """
    def _get_outlines(pdf_obj):
        if hasattr(pdf_obj, 'outline') and pdf_obj.outline:
            return pdf_obj.outline
        else:
            return []
    # check parameters
    if pdf_path is None and pdf_obj is None:
        return put_err('pdf_path or pdf_obj is None', None)
    # get outlines
    if pdf_obj is not None:
        outlines = _get_outlines(pdf_obj)
    elif pdf_path is not None and os.path.isfile(pdf_path):
        with open(pdf_path, 'rb') as file:
            pdf_obj = PyPDF2.PdfReader(file)
            outlines = _get_outlines(pdf_obj)
    # check for valid bookmarks, get flat section list
    if len(outlines) == 0:
        return False
    else:
        outlines = _flatten_pdf_bookmarks(*outlines)
    # set default section names
    if not section_names:
        section_names = ['Abstract', 'Introduction', 'Materials', 'Methods',
                         'Results', 'Discussion', 'References']
    # check whether any of the section names is in the outlines
    for outline in outlines:
        for section_name in section_names:
            pattern = r'\b{}\b'.format(re.escape(section_name))
            if re.search(pattern, outline.title, re.IGNORECASE):
                return outlines
    return False

def get_sci_bookmarks_from_pdf(pdf_path:str = None, pdf_obj = None, section_names:List[str]=[]):
    """
    Returns a list of section names from a scientific PDF.

    Parameters:
        pdf_path (str): The path to the PDF file. Default is None.
        pdf_obj: The PDF object. Default is None.
        section_names (List[str]): A list of section names to search for.
            If None, all sections include 'Abstract', 'Introduction', 'Materials', 'Methods',
            'Results', 'Discussion', 'References' will be searched.

    Returns:
        List[str]: A list of section names found in the PDF.
    """
    # check parameters
    if pdf_path is None and pdf_obj is None:
        return put_err('pdf_path or pdf_obj is None', None)
    # set default section names
    if not section_names:
        section_names = ['Abstract', 'Introduction', 'Materials', 'Methods',
                         'Results', 'Discussion', 'References']
    # get pdf full txt
    if pdf_obj is not None:
        # extract text from pdf obj
        content = '\n'.join([page.extract_text() for page in pdf_obj.pages])
    elif pdf_path is not None and os.path.isfile(pdf_path):
        # get text from pdf file
        content = convert_pdf_to_txt(pdf_path)
    # get section titles
    ret = []
    for section in section_names:
        if content.find(section) != -1:
            ret.append(section)
    return ret
    
def get_section_bookmarks(pdf_path:str = None, pdf_obj = None):
    """
    Returns a list of titles of bookmark sections in a PDF.

    Parameters:
    - pdf_path (str): The path to the PDF file. Defaults to None.
    - pdf_obj: The PDF object(Being opened!). Defaults to None.

    Returns:
    - list: A list of titles of bookmark sections in the PDF.
    Returns None if there are no bookmark sections or if the PDF file does not exist.
    """
    def worker(pdf_obj):
        sections = has_sci_bookmarks(None, pdf_obj)
        if not sections:
            # do not has inner bookmarks, just parse from text
            return get_sci_bookmarks_from_pdf(None, pdf_obj)
        # has inner bookmarks, parse from outline
        return [section.title for section in sections]
    # check path
    if not os.path.isfile(pdf_path):
        return put_err(f'{pdf_path:s} does not exist', None)
    # get section titles
    if pdf_obj is None:
        with open(pdf_path, 'rb') as file:
            pdf_obj = PyPDF2.PdfReader(file)
            return worker(pdf_obj)
    else:
        return worker(pdf_obj)
    
def get_english_part_of_bookmarks(bookmarks:List[str]):
    """
    Retrieves the English part of the given list of bookmarks.

    Parameters:
        bookmarks (list[str]): A list of bookmarks.

    Returns:
        list[str]: A list containing only the English part of the bookmarks.
    """
    if bookmarks is None:
        return put_err('bookmarks is None', None)
    english_bookmarks = []
    for bookmark in bookmarks:
        match = re.search(r'[a-zA-Z]+[a-zA-Z\s\S]+', bookmark)
        english_bookmarks.append(match.group(0).strip() if match else bookmark)
    return english_bookmarks

def get_section_from_paper(paper:str, key:str,
                           keys:List[str] = ['Title', 'Authors', 'Abstract', 'Keywords',
                                             'Introduction', 'Materials & Methods',
                                             'Results', 'Discussion', 'References']):
    """
    extract section of a science paper by key
    
    Parameters:
        paper (str): a science paper.
        key (str): one of the sections in the paper.
            can be 'Title', 'Authors', 'Abstract', 'Keywords', 'Introduction',
            'Materials & Methods', 'Results', 'Discussion', 'References'
        keys (list[str], optional): a list of keys to extract.
            Defaults to ['Title', 'Authors', 'Abstract', 'Keywords', 'Introduction',
            'Materials & Methods', 'Results', 'Discussion', 'References'].
    """
    # 构建正则表达式模式，使用re.IGNORECASE标志进行不区分大小写的匹配
    if paper is None or key is None:
        return put_err('paper or key is None', None)
    # TODO : sometimes may fail
    pattern = r'\b{}\b.*?(?=\b{})'.format(key, keys[keys.index(key)+1] if key != keys[-1] else '$')
    # 使用正则表达式匹配内容
    match = re.search(pattern, paper, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0)
    else:
        return put_err(f'key "{key}" not found in paper', '')

def format_paper_from_txt(content:str,
                          struct:List[str] = ['Title', 'Authors', 'Abstract', 'Keywords',
                                              'Introduction', 'Materials & Methods',
                                              'Results', 'Discussion', 'References']):
    content = content.replace('\n', '')
    struction = {}
    for key in struct:
        struction[key] = ''

if __name__ == '__main__':
    # dev code
    from mbapy.base import rand_choose
    from mbapy.file import convert_pdf_to_txt
    
    # RIS parse
    ris = parse_ris('./data_tmp/savedrecs.ris', '')
    ris = rand_choose(ris)
    print(f'title: {ris["title"]}\ndoi: {ris["doi"]}')
    
    # search
    search_result = search_by_baidu('linaclotide', 11)
    search_result2 = search(ris["title"])
    
    # download
    dl_result = download_by_scihub('./data_tmp/', title = search_result[0]['title'])
    download_by_scihub(ris["doi"], './data_tmp/', file_full_name = f'{ris["title"]:s}.pdf')
    
    # extract section
    pdf_path = replace_invalid_path_chr("./data_tmp/{:s}.pdf".format(ris["title"]))
    sections = get_english_part_of_bookmarks(get_section_bookmarks(pdf_path))
    paper, section = convert_pdf_to_txt(pdf_path), rand_choose(sections, 0)
    print(sections, section, get_section_from_paper(paper, section, keys=sections))
