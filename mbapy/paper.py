'''
Date: 2023-07-07 20:51:46
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-07-16 19:05:08
FilePath: \BA_PY\mbapy\paper.py
Description: 
'''
import os, re, requests
from typing import List, Dict

from lxml import etree
import rispy
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
    
@parameter_checker(check_parameters_len, raise_err = False)
def search_IF(query:str, proxies = None):    
    """
    This function performs a search on the 'ablesci' website using the provided query. It retrieves the first result, extracts relevant information from the page, and returns a dictionary containing the full name, abbreviated name, subjects, impact factor, WOS query, and CAS query.

    Parameters:
    - query (str): The search query to be used.
    - proxies (Optional[dict]): A dictionary of proxy settings to be used for the request. Defaults to None.

    Returns:
    - dict: A dictionary containing the following information:
        - full_name (str): The full name of the journal.
        - abbr_name (str): The abbreviated name of the journal.
        - subjects (list[str]): A list of subjects covered by the journal.
        - impact_factor (str): The impact factor of the journal.
        - wos_q (str): The WOS query associated with the journal.
        - cas_q (str): The CAS query associated with the journal.
    
    If no results are found, an error message is returned.
    """
    base_url = 'https://www.ablesci.com/journal/index'
    res = session.request(method='GET', url=base_url, params={'keywords': query})
    s = etree.HTML(res.text)
    links = s.xpath("//a[@class='journal-name']//@href")
    if len(links) > 0:
        res = session.request(method='GET', url=links[0])
        page = etree.HTML(res.text)
        full_name = page.xpath("//tbody/tr[1]//td[2]/text()")[0]
        abbr_name = page.xpath("//tbody/tr[2]//td[2]/text()")[0]
        if query == full_name or query == abbr_name:
            subjects = page.xpath("//tbody/tr[3]/td[2]/div/span/@title")
            subjects = [subject.replace('\n', ' | ') for subject in subjects]
            impact_factor = page.xpath("//tbody/tr[6]/td[2]/span/text()")[0]
            wos_q = page.xpath("//tbody/tr[12]/td[2]/table/tbody/tr/td/div/span/text()")[0]
            cas_q = page.xpath("//tbody/tr[13]/td[2]/table/tbody/tr/td/div/span/text()")[0]
            return {'full_name': full_name, 'abbr_name': abbr_name, 'subjects': subjects,
                    'impact_factor': impact_factor, 'wos_q': wos_q, 'cas_q': cas_q}
    return put_err('No results found', None)
    
@parameter_checker(check_parameters_len, raise_err = False)
def search_by_baidu(query:str, limit:int = 1, proxies = None):
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
            - journal (str): The name of the journal associated with the article.
    """
    def _get_a_search_page(query:str, page:int = 0):
        res = session.request(method='GET', url='https://xueshu.baidu.com/s',
                              params={'wd': query, 'pn': page, 'filter': 'sc_type%3D%7B1%7D'})
        s = etree.HTML(res.text)
        return s.xpath("//div[@class='sc_content']/h3/a/@href")
    
    def _parse_info(xpath_search_key:str, xpath_obj, is_single: bool = True):
        search_result = xpath_obj.xpath(xpath_search_key)
        if is_single:
            return get_default_for_bool(search_result, [''])[0].strip()
        return get_default_for_bool(search_result, [''])
    
    def _parse_links(links:list):
        results = []
        for link in links:
            res = session.request(method='GET', url=link, proxies=proxies)
            s = etree.HTML(res.text)
            title = s.xpath("//div[@class='main-info']/h3/a/text()")
            if len(title) == 0:
                title = s.xpath("//div[@class='main-info']/h3/span/text()")
            title = get_default_for_bool(title, [''])[0].strip()
            abstract = _parse_info("//p[@class='abstract']/text()", s)
            keyword = _parse_info("//div[@class='kw_wr']/p[@class='kw_main']/span/a/text()", s, False)
            doi = _parse_info("//div[@class='doi_wr']/p[@class='kw_main']/text()", s)
            journal = _parse_info("//a[@class='journal_title']/text()", s)
            results.append({'title': title, 'abstract': abstract, 'keyword': keyword,
                            'doi': doi, 'journal': journal})
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

@parameter_checker(check_parameters_len, check_parameters_len, raise_err = False)
def search_by_pubmed(query:str, email:str = None, limit:int = 1):
    """
    Searches for articles in the PubMed database based on a query.

    Parameters:
    - query (str): The search query.
    - email (str, optional): The email address used for Entrez.email. Defaults to None.
    - limit (int, optional): The maximum number of results to return. Defaults to 1.

    Returns:
        list: A list of dictionaries containing information about the articles found.
            Each dictionary has the following keys:
            - title (str): The title of the article.
            - abstract (str): The abstract of the article.
            - doi (str): The DOI (Digital Object Identifier) of the article.
            - journal (str): The name of the journal associated with the article.

    Raises:
    - parameter_checker.ParameterLengthError: If the length of the parameters is not equal to check_parameters_len.
    """
    from Bio import Entrez
    Entrez.email = email  # 设置邮箱地址
    handle = Entrez.esearch(db='pubmed', term=query, retmax=limit)
    record = Entrez.read(handle)
    handle.close()
    pubmed_ids = record['IdList']

    results = []
    for pubmed_id in pubmed_ids:
        handle = Entrez.efetch(db='pubmed', id=pubmed_id, retmode='xml')
        record = Entrez.read(handle)
        handle.close()
        
        if len(record['PubmedArticle']) > 0:
            article_info = record['PubmedArticle'][0]['MedlineCitation']
            article = article_info['Article']
            title = str(article['ArticleTitle'])
            doi = str(article['ELocationID'][0]) if 'ELocationID' in article else ''
            abstract = '.\n'.join([str(element) for element in article['Abstract']['AbstractText']]) if 'Abstract' in article else ''
            journal = str(article['Journal']['Title']) if 'Journal' in article else ''
            keywords = article_info['KeywordList'][0] if 'KeywordList' in article_info else []
            keywords = [str(keyword) for keyword in keywords]
            results.append({'title': title, 'abstract': abstract, 'doi': doi, 'journal': journal, 'keywords': keywords})

    return results

search_by_pubmed('linaclotide', 'baohm20@lzu.edu.cn', 10)

@parameter_checker(check_parameters_len, raise_err = False)
def search_by_wos(query:str, limit:int = 1, proxies = None):
    pass    
        
def search(query:str, limit:int = 1, search_engine:str = 'baidu xueshu', email:str = None):
    """
    Search for a given query using a specified search engine and return the results.

    Parameters:
    - query (str): The query string to search for.
    - limit (int): The maximum number of results to return.
    - search_engine (str): The search engine to use. Default is 'baidu xueshu'.
         allows: 'baidu xueshu', 'pubmed', 'wos', if not recognized, returns None

    Returns:
    - The search results as a list of dict, contain 'title', 'abstract', 'keyword' and 'doi'.
    """
    if search_engine == 'baidu xueshu':
        return search_by_baidu(query, limit)
    elif search_engine == 'pubmed' and email is not None:
        return search_by_pubmed(query, email, limit)
    elif search_engine == 'wos':
        raise NotImplementedError
        # if os.path.isfile(get_storage_path('science_direct_cookie.txt')):
        #     cookie = opts_file(get_storage_path('science_direct_cookie.txt'))
        # else:
        #     import browser_cookie3
        #     cookie = browser_cookie3.load('https://www.sciencedirect.com/')
        #     opts_file(get_storage_path('science_direct_cookie.txt'), 'w', encoding='utf-8', data = cookie)
    else:
        return put_err(f'Unknown search engine: {search_engine}, returns None', None)

def _get_available_scihub_urls(proxies = None):
    '''
    Finds available scihub urls via http://tool.yovisun.com/scihub/
    '''
    links = []
    res = session.request(method='GET', url='http://tool.yovisun.com/scihub/', proxies=proxies)
    results = etree.HTML(res.text).xpath('//tr[@class="item"]')
    for result in results:
        # 我真的服了这个'latin1'编码，都没见过，还是问的chatGPT。。。
        status = result.xpath('.//td[@class="status"]/span[@class="label  label-success"]/text()')[0]
        status = status.encode('latin1').decode('utf-8')
        if status == '可用':
            ssl_link = result.xpath('.//td[@class="domainssl"]/a/@href')[0]
            links.append(ssl_link)
    return links

# avoid multiple requests
available_scihub_urls = None
def _update_available_scihub_urls():
    global available_scihub_urls
    available_scihub_urls = _get_available_scihub_urls() if available_scihub_urls is None else available_scihub_urls
    return available_scihub_urls    

def _download_from_scihub_webpage(webpage:requests.Response, proxies = None):
    """
    Downloads a file from the SciHub webpage.

    Args:
        webpage (requests.Response): The response object of the SciHub webpage.
        proxies (dict, optional): The proxies to be used for the request. Defaults to None.

    Returns:
        dict: A dictionary containing the title, DOI, and the response object of the download request.
    """
    def _get_valid_download_link(link:str):
        available_scihub_urls = _update_available_scihub_urls()
        if not link.startswith('http:'):
            if link.find('sci-hub') == -1:
                link = (available_scihub_urls[0]+'/') + link
            else:
                link = 'http:' + link
        return link
            
    results = etree.HTML(webpage.text)
    title = results.xpath('//div[@id="citation"]/i/text()')[0]
    doi = results.xpath('//div[@id="citation"]//following-sibling::text()')[0]
    download_link = results.xpath('//div[@id="buttons"]/button[1]/@onclick')[0].split("'")[1]
    valid_download_link = _get_valid_download_link(download_link)
    res = session.request(method='GET', url=valid_download_link, proxies=proxies)
    return {'title': title, 'doi': doi, 'res': res}

@parameter_checker(check_parameters_bool, raise_err = False)
def download_from_scihub_by_doi(doi:str, proxies = None):
    """
    Downloads a file from the Sci-Hub database using the DOI.

    Parameters:
        doi (str): The DOI of the file to download.
        proxies (dict): A dictionary of proxies to use for the request.

    Returns:
        a dictionary containing the title, DOI, and the response object of the download request.
        if meets error, returns None.

    Raises:
        Exception: If the DOI does not exist or if there is an error fetching the file from Sci-Hub.
    """
    # try:
    available_scihub_urls = _update_available_scihub_urls()
    res = session.request(method='GET', url=available_scihub_urls[0]+'/'+doi, proxies=proxies)
    return _download_from_scihub_webpage(res)
    # except:
    #     return put_err(f'Maybe DOI: {doi:s} does not exist. scihub fetch error', None)
            
@parameter_checker(check_parameters_bool, raise_err = False)
def download_from_scihub_by_title(title, proxies = None):
    """
    Downloads a document from Scihub by title.

    Parameters:
        title (str): The title of the document to be downloaded.
        proxies (dict, optional): A dictionary of proxies to be used for the HTTP request.

    Returns:
        a dictionary containing the title, DOI, and the response object of the download request.
        if meets error, returns None.

    Raises:
        Exception: If the document with the given title does not exist on Scihub.
    """
    try:
        available_scihub_urls = _update_available_scihub_urls()
        res = session.post(available_scihub_urls[0], data = {'request': title}, proxies=proxies)
        return _download_from_scihub_webpage(res)
    except:
        return put_err(f'Maybe TITLE: {title:s} does not exist. scihub fetch error', None)
            
def download_by_scihub(dir: str, doi: str = None, title:str = None,
                       file_full_name:str = None, use_title_as_name: bool = True,
                       valid_path_chr:str = '_'):
    """
    Download a paper from Sci-Hub using its DOI.
    if file_full_name is None, use the paper's title as the file name, if not, use the paper's DOI as the file name.

    Args:
        dir (str): The directory where the downloaded file will be saved.
        doi (str): The DOI (Digital Object Identifier) of the paper.
        file_full_name (str, optional): The name of the downloaded file, include the file extension(.pdf). Defaults to None.
        use_title_as_name (bool, optional): Whether to use the paper's title as the file name. Defaults to True.
        valid_path_chr (str, optional): The character used to replace invalid characters in the file name. Defaults to '_'.

    Returns:
        dict or None: If successful, returns a dictionary containing information
            about the downloaded paper. If unsuccessful, returns None.
    """
    # check dir exists, if not, create it
    if not check_parameters_path(dir):
        os.makedirs(dir)
    # check whether doi or title are specified
    if doi is None and title is None:
        return put_err('Either DOI or title must be specified, returns None', None)
    # download from Sci-Hub by DOI or title
    if doi:
        result = download_from_scihub_by_doi(doi)
    else:
        result = download_from_scihub_by_title(title)
    if result is None:
        return None
    # get the file name, save the file
    if file_full_name is not None:
        file_name = file_full_name
    else:
        file_name = ((title if title else result['title']) if use_title_as_name else doi) + '.pdf'
    file_name = replace_invalid_path_chr(file_name, valid_path_chr)
    opts_file(os.path.join(dir, file_name), 'wb', data = result['res'].content)
    return result
    
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
    from mbapy.file import convert_pdf_to_txt, read_json
    
    # RIS parse
    ris = parse_ris('./data_tmp/savedrecs.ris', '')
    ris = rand_choose(ris)
    print(f'title: {ris["title"]}\ndoi: {ris["doi"]}')
    
    # search impact factor
    print(search_IF('Nature Communications'))
    
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
