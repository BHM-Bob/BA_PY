import os
import re
from typing import Dict, List

import requests
import rispy
from lxml import etree

if __name__ == '__main__':
    # dev mode
    import mbapy.web as web
    from mbapy.base import *
    from mbapy.file import (convert_pdf_to_txt, opts_file, read_text,
                            get_valid_file_path)
else:
    # release mode
    from .. import web
    from ..base import *
    from ..file import convert_pdf_to_txt, opts_file, get_valid_file_path
            

session = web.get_requests_retry_session()

def _get_available_scihub_urls(proxies = None):
    '''
    Finds available scihub urls via http://tool.yovisun.com/scihub/
    '''
    links = []
    res = session.request(method='GET', url='http://tool.yovisun.com/scihub/', proxies=proxies)
    results = etree.HTML(res.text).xpath('//tr[@class="item"]')
    for result in results:
        # 真的服了这个'latin1'编码，都没见过。。。
        status = result.xpath('.//td[@class="status"]/span[@class="label  label-success"]/text()')[0]
        status = status.encode('latin1').decode('utf-8')
        if status == '可用':
            ssl_link = result.xpath('.//td[@class="domainssl"]/a/@href')[0]
            links.append(ssl_link)
    return links

# avoid multiple requests
available_scihub_urls = None
def _update_available_scihub_urls():
    """
    Updates the list of available SciHub URLs.
    This function updates the global variable `available_scihub_urls` by calling the `_get_available_scihub_urls()` function if `available_scihub_urls` is None. Otherwise, it returns the current value of `available_scihub_urls`.

    Returns:
        list: The updated list of available SciHub URLs.
    """
    global available_scihub_urls
    available_scihub_urls = _get_available_scihub_urls() if available_scihub_urls is None else available_scihub_urls
    return available_scihub_urls   

def get_clean_doi(doi:str):
    doi_match = re.search(r'10\.[a-zA-Z0-9./]+', doi)
    if doi_match:
        return doi_match.group()
    else:
        return ''   

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
    # get right title and doi from sci-hub webpage is not required
    try:
        title = results.xpath('//div[@id="citation"]/i/text()')[0]
        doi = results.xpath('//div[@id="citation"]//following-sibling::text()')[0]
    except:
        try:
            paper_info = results.xpath('//div[@id="citation"]//text()')[0]
            doi = get_clean_doi(paper_info)
            title = doi.replace('/', '_')
        except:
            title = None
            doi = None
    # get right download link is required
    try:
        download_link = results.xpath('//div[@id="buttons"]//@onclick')[0].split("'")[1]
        valid_download_link = _get_valid_download_link(download_link)
        res = session.get(url = valid_download_link, proxies=proxies, stream=False, timeout=60)
        return {'title': title, 'doi': get_clean_doi(doi), 'res': res}
    except:
        return None

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
        - dir (str): The directory where the downloaded file will be saved.
        - doi (str): The DOI (Digital Object Identifier) of the paper.
        - file_full_name (str, optional): The name of the downloaded file, include the file extension(.pdf). Defaults to None.
        - use_title_as_name (bool, optional): Whether to use the paper's title as the file name. Defaults to True.
        - valid_path_chr (str, optional): The character used to replace invalid characters in the file name. Defaults to '_'.

    Returns:
        dict or None: If successful, returns a dictionary containing information
            about the downloaded paper. If unsuccessful, returns None.
            
    Notes:
        - if doi is None and can't get doi from sci-hub webpage, doi will be set as %Y%m%d.%H%M%S
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
        return put_err(f"can't download with \ndoi:{doi}\ntitle:{title}\n, returns None", None)
    # deal with err title and doi
    if result['title'] is None:
        result['title'] = title if title is not None else doi.replace('/', '_')
    if result['doi'] is None:
        result['doi'] = doi if doi is not None else get_fmt_time("%Y%m%d.%H%M%S")
    # get the file name, save the file
    if file_full_name is not None:
        file_name = file_full_name
    else:
        file_name = ((title if title else result['title']) if use_title_as_name else doi.replace('/', '_')) + '.pdf'
    file_name = file_name.replace('/', ' or ')
    file_name = get_valid_file_path(file_name, valid_path_chr)
    file_path = os.path.join(dir, file_name)
    opts_file(file_path, 'wb', data = result['res'].content)
    result['file_name'] = file_name
    result['file_path'] = file_path
    return result
    
if __name__ == '__main__':
    # dev code
    from mbapy.base import rand_choose
    from mbapy.file import convert_pdf_to_txt, read_json

    # download
    title = 'Linaclotide: a novel compound for the treatment of irritable bowel syndrome with constipation'
    doi = '10.1517/14656566.2013.833605'
    dl_result = download_by_scihub('./data_tmp/papers/', doi = '10.1517/14656566.2013.833605')
    download_by_scihub('./data_tmp/', doi, title)