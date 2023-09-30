import os
import random
import re
import requests
from typing import Dict, List

import crossref_commons.retrieval
import rispy
from lxml import etree

if __name__ == '__main__':
    # dev mode
    import mbapy.web as web
    from mbapy.base import *
    from mbapy.file import (convert_pdf_to_txt, opts_file, read_text,
                            replace_invalid_path_chr)
else:
    # release mode
    from .. import web
    from ..base import *
    from ..file import convert_pdf_to_txt, opts_file, replace_invalid_path_chr

session = requests.Session()

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
def search_by_baidu(query:str, limit:int = 1, proxies = None,
                    use_browser = False, browser = None, use_undetected=True):
    """
    This function is used to search for articles on Baidu Scholar using a given query.
    
    Note: If use_browser_first is False, you may open baidu xueshu website before this function being called. 
    
    Parameters:
        query (str): The search query.
        limit (int, optional): The maximum number of articles to retrieve. Defaults to 1.
        use_browser_first (bool, optional): Whether to use selenium to get website for the first time, since it make more successful.
        
    Returns:
        list: A list of dictionaries containing information about the articles found.
            Each dictionary has the following keys:
            - title (str): The title of the article.
            - abstract (str): The abstract of the article.
            - keyword (list): A list of keywords associated with the article.
            - doi (str): The DOI (Digital Object Identifier) of the article.
            - journal (str): The name of the journal associated with the article.
    """
    def _get_a_search_page(query:str, page:int = 0, browser = None):
        url = f'https://xueshu.baidu.com/s?wd={query:s}&pn={page*10:d}&tn=SE_baiduxueshu_c1gjeupa&ie=utf-8&sc_hit=1&rsv_page={page:d}'
        if use_browser:
            bs, page = web.get_url_page_se(browser, url, return_html_text=True)
        else:
            page = web.get_url_page(url, 'utf-8')
        s = etree.HTML(page)
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
    
    if use_browser:
        if browser is None:
            browser = web.get_browser('Chrome', options=['--no-sandbox', '--headless'],
                                      use_undetected=use_undetected)
        browser.get('https://xueshu.baidu.com')
    
    links = _get_a_search_page(query, 0, browser)
    if limit > 10:
        # 未登录的百度学术一页只显示10个结果
        page = 1
        while limit > 10:
            limit -= 10
            links += _get_a_search_page(query, page, browser)[: (limit%10 if limit > 10 else limit)]
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

def _parse_wos_paper_link(browser):
    # get page
    time.sleep(5)
    web.scroll_browser(browser, 'bottom', 5)
    # parse page info
    s = etree.HTML(browser.page_source)
    title = web.parse_xpath_info('//h2[@class="title text--large"]/text()', s)
    authors = web.parse_xpath_info('//span[starts-with(@id, "author-")]//text()', s, False)
    doi = web.parse_xpath_info('//span[@data-ta="FullRTa-DOI"]/text()', s)
    date = web.parse_xpath_info('//span[@data-ta="FullRTa-indexedDate"]/text()', s)
    article_type = web.parse_xpath_info('//span[@data-ta="FullRTa-doctype-0"]/text()', s)
    abstruct_lst = web.parse_xpath_info('//div[@class="abstract--instance"]/p/text()', s, False)
    abstruct = '\n'.join(abstruct_lst)
    keywords = web.parse_xpath_info('//app-full-record-keywords[@class="ng-star-inserted"]//div//span//a//text()', s, False)
    jounal = web.parse_xpath_info('//mat-sidenav-content//span//a//text()', s)
    impact_factor = web.parse_xpath_info('//mat-sidenav-content//span//button//span//span//text()', s)
    jounal_subjects = web.parse_xpath_info('//div[@class="journal-content"]//div[5]//span[@class="value-wrap ng-star-inserted"]//text()', s, False)
    
    return {
        'title': title, 'authors': authors, 'doi': doi, 'date': date,
        'article_type': article_type, 'abstruct': abstruct, 'keywords': keywords,
        'jounal': jounal, 'jounal_subjects': jounal_subjects, 'impact_factor': impact_factor
    }

@parameter_checker(check_parameters_len, raise_err = False)
def search_by_wos(query:str, limit:int = 1,
                  browser = 'Chrome', browser_driver_path:str = None,
                  wait_per_link:int = 10, proxies = None):
    """
    Perform a search on the Web of Science (WOS) database using a given query string.
    
    Parameters:
        query (str): The query string to search for in the WOS database.
        limit (int, optional): The maximum number of search results to return. Defaults to 1.
        browser (str, optional): The browser to use for performing the search. Defaults to 'Chrome'.
        browser_driver_path (str, optional): The path to the browser driver executable. Defaults to None.
        proxies (optional): The proxies to use for the browser. Defaults to None.
    
    Returns:
        list: A list of dictionaries containing information about the search results. Each dictionary contains the following keys:
            - title (str): The title of the paper.
            - authors (str): The authors of the paper.
            - doi (str): The DOI of the paper.
            - date (str): The indexed date of the paper.
            - article_type (str): The type of the article (e.g., journal article, conference paper, etc.).
            - abstract (str): The abstract of the paper.
            - keywords (str): The keywords associated with the paper.
            - journal (str): The journal in which the paper was published.
            - journal_subjects (str): The subjects covered by the journal.
            - impact_factor (str): The impact factor of the journal.
    """
    def _get_lst_page(browser):
        """
        在浏览器打开一个新的列表页面后调用，返回列表页面的链接列表
        called after opening a new list page, returns the list page's link list
        """
        try:
            locator = (web.transfer_str2by('xpath'), '//div[@class="search-display"]')
            web.WebDriverWait(browser, 5).until(web.EC.presence_of_element_located(locator))
        finally:
            web.scroll_browser(browser, 'bottom', 5)
        web.wait_for_amount_elements(browser, 'xpath', '//div[@dir="ltr" and @class="ng-star-inserted"]/app-summary-title/h3/a', 45)
        return etree.HTML(browser.page_source).xpath('//div[@dir="ltr" and @class="ng-star-inserted"]/app-summary-title/h3/a/@href')
    # init browser, make search and get the first page
    browser = web.get_browser(browser, browser_driver_path, ['--no-sandbox'], True)
    browser.get("https://www.webofscience.com/wos/alldb/basic-search")
    web.click_browser(browser, '//button[@id="onetrust-accept-btn-handler"]', 'xpath')
    web.send_browser_key(browser, query+'\n', '//input[@name="search-main-box"]', 'xpath')
    
    lst_page_handle = browser.current_window_handle
    sum_searched = 0
    while sum_searched <= limit:
        links = _get_lst_page(browser)
        time.sleep(3+3*random.random())
        info = []
        # browser.execute_script("window.open('','_blank');") # 被浏览器拦截
        browser.switch_to.new_window('https://www.webofscience.com'+links[0])
        time.sleep(10+3*random.random())
        for link in links:
            try:
                browser.get('https://www.webofscience.com'+link)
                time.sleep(wait_per_link+3*random.random())
                info.append(_parse_wos_paper_link(browser))
                sum_searched += 1
            except:
                pass
        browser.close()
        browser.switch_to.window(lst_page_handle)
        web.click_browser(browser, '//span[@class="mat-button-wrapper"]/mat-icon[text()="chevron_right"][1]', 'xpath')
            
    return info
        
def search(query:str, limit:int = 1, search_engine:str = 'baidu xueshu',
           email:str = None, browser:str = 'Chrome', browser_driver_path:str = web.CHROME_DRIVER_PATH,
           use_browser = False, use_undetected=True):
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
        return search_by_baidu(query, limit, use_browser=use_browser, use_undetected=use_undetected)
    elif search_engine == 'pubmed' and email is not None:
        return search_by_pubmed(query, email, limit)
    elif search_engine == 'wos':
        return search_by_wos(query, limit, browser, browser_driver_path)
    else:
        return put_err(f'Unknown search engine: {search_engine}, returns None', None)

@parameter_checker(check_parameters_len, raise_err = False)
def get_reference_by_doi(doi:str):
    """
    Return:
        - dict:
            - 'key': '2019041211085787000_b1-0711081'
            - 'doi-asserted-by': 'crossref'
            - 'first-page': '1480'
            - 'DOI': '10.1053/j.gastro.2005.11.061'
            - 'article-title'(may not exist): 'Functional bowel disorders'
            - 'volume': '130'
            - 'author': 'Longstreth'
            - 'year': '2006'
            - 'journal-title': 'Gastroenterology'
    """
    try:
        return crossref_commons.retrieval.get_publication_as_json(doi)['reference']
    except:
        return put_err(f'can not retrive from crossref with doi:{doi}, return None', None)


if __name__ == '__main__':
    # dev code
    from mbapy.file import read_json

    # search
    search_result_bd = search_by_baidu('linaclotide', 11, use_browser=True, use_undetected=True)
    search_result_pm = search_by_pubmed('linaclotide', read_json('./data_tmp/id.json')['edu_email'], 11)
    search_result_wos = search_by_wos("linaclotide", 61, browser_driver_path=web.CHROMEDRIVERPATH)