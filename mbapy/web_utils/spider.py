import itertools
import requests
import time
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union, Callable

from lxml import etree
from tqdm import tqdm

if __name__ == '__main__':
    from mbapy.base import (Configs, check_parameters_len,
                            check_parameters_path, put_err)
    from mbapy.file import get_valid_file_path
    from mbapy.web_utils.request import get_url_page_s, random_sleep
    from mbapy.web_utils.task import CoroutinePool, TaskStatus
else:
    from ..base import (Configs, check_parameters_len, check_parameters_path,
                        put_err)
    from ..file import get_valid_file_path
    from .request import get_url_page_s, random_sleep
    from .task import CoroutinePool, TaskStatus
    
    
def install_headers(agent: str = None):
    opener = urllib.request.build_opener()
    agent = agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36'
    opener.addheaders = [('User-agent', agent)]
    urllib.request.install_opener(opener)

async def get_web_html_async(url: str, encoding: str = 'utf-8'):
    """async version of get_url_page_s"""
    return get_url_page_s(url, encoding)

async def retrieve_file_async(url: str, file_path: str):
    """async version of download_file"""
    install_headers()
    return urllib.request.urlretrieve(url, file_path)

@dataclass
class AsyncResult:
    async_pool: CoroutinePool
    name: str
    result: Any = TaskStatus.NOT_RETURNED
    def get(self):
        """block to get result"""
        # short cut to get result
        if self.result is not TaskStatus.NOT_RETURNED:
            return self.result
        # wait for result
        self.result = self.async_pool.query_task(self.name)
        while self.result == TaskStatus.NOT_FINISHED:
            self.result = self.async_pool.query_task(self.name)
            time.sleep(0.1)
        # will also return TaskStatus.NOT_SUCCEED
        return self.result
        
def only_sleep(seconds: float = 1, rand = True, max: float = 5):
    """sleep for seconds, with random sleep up to max seconds, return True."""
    if rand:
        random_sleep(max, seconds)
    else:
        time.sleep(seconds)
    return True

def text_fn(x: Union[str, List[str], etree._Element, List[etree._Element]]):
    """
    get text from element or list of elements
    """
    if isinstance(x, etree._Element):
        return x.text.strip()
    elif isinstance(x, list) and len(x) > 0 and isinstance(x[0], etree._Element):
        return [i.text.strip() for i in x]
    else:
        return x
    
def Compose(lst: List[Callable]):
    """
    Compose a list of functions into a single function.
    """
    def inner(*args, **kwargs):
        res = lst[0](*args, **kwargs)
        for f in lst[1:]:
            res = f(res)
        return res
    return inner
    
@dataclass
class BasePage:
    """page container, store each web-page and its parsed data."""
    name: str = '' # name of this page, could be empty
    xpath: str = '' # xpath expression to extract data
    _async_task_pool: CoroutinePool = None # async task pool to execute async tasks
    # result generally is a list of result, top-level list is for each web-page, and it ONLY store one kind item.
    result: List[Any] = field(default_factory = lambda: [])
    result_page_html: List[str] = field(default_factory = lambda: []) # web-page html of parsed result
    result_page_xpath: List[etree._Element] = field(default_factory = lambda: []) # web-page xpath object of parsed result
    father_page: 'BasePage' = None # father page of this page
    next_pages: Dict[str, 'BasePage'] = field(default_factory = lambda: {}) # next pages of this page
    before_func: Callable[['BasePage'], bool] = None # before function to execute before parsing, check if need to parse this page
    after_func: Callable[['BasePage'], bool] = None # after function to execute after parsing, do something after parsing this page
    def add_next_page(self, name:str, page: 'BasePage') -> None:
        if isinstance(name, str):
            self.next_pages[name] = page
            return True
        else:
            return put_err('name should be str, skip and return False', False)
    def parse(self, results: List[List[Union[str, etree._Element]]] = None) -> None:
        """
        parse data from results, override by subclass.
        In BasePage.perform, it will call this function to parse data and store in self.result..
        """
    def _process_parsed_data(self, *args, **kwargs):
        """
        process parsed data, could be override by subclass
        """
    def perform(self, *args, results: List[List[Union[str, etree._Element]]] = None, **kwargs):
        self.result =  self.parse(results)
        self._process_parsed_data(*args, **kwargs)
        return self.result
    
class PagePage(BasePage):
    """
    Only START and store a new web page or a list of web pages.
    """
    def __init__(self, url: List[Union[str, List[str]]] = None,
                 web_get_fn: Callable[[Any], str] = get_web_html_async,
                 *args, **kwargs) -> None:
        """
        Parameters:
            - url: List[str | List[str]], url to get web page, could be empty if get from father_page.result
            - web_get_fn: Callable[[Any], str], function to get web page, default is get_url_page_s
            - args: any, args for web_get_fn
            - kwargs: any, kwargs for web_get_fn
                - each_delay: float, delay seconds between each request, default is 1.
        
        NOTE:
            - item of url() could be str of list of str.
        """
        super().__init__()
        self.url = url
        self.web_get_fn = web_get_fn
        self.each_delay = kwargs.get('each_delay', 0.5)
        self.max_each_delay = kwargs.get('max_each_delay', 1)
        self.args = args
        self.kwargs = kwargs
    def _process_url(self, results: List[List[Union[str, etree._Element]]] = None):
        # NOTE:result always be list(page) for list(items)
        # get url from results or father_page.result_page_xpath
        if self.url is None:
            if results is None:
                results = self.father_page.result
        elif isinstance(self.url, list) and len(self.url) > 0 and isinstance(self.url[0], str):
            results = [self.url] # => list of list of urls
        else:
            results = self.url
        return results        
    def parse(self, results: List[List[Union[str, etree._Element]]] = None):
        """
        get new web-page(s) from self.url, results(as links) or self.father_page.result_page_xpath(as links)
        Note: a page store one kind item, the content of this item could be url or list of urls.
        """
        results = self._process_url(results)
        # get web-page(s) and parse
        for page in tqdm(results, leave=False, desc=f'{self.name} get Page'): # a page store one kind item, a list of urls
            self.result.append([])
            self.result_page_html.append([])
            self.result_page_xpath.append([])
            for url in tqdm(page, leave=False, desc=f'{self.name} get Item'):
                task_name = self._async_task_pool.add_task(None, self.web_get_fn, url, *self.args, **self.kwargs)
                self.result[-1].append(AsyncResult(self._async_task_pool, task_name))
                self.result_page_html[-1].append(self.result[-1][-1])
                # NOTE: do not append xpath object because html is async.
                random_sleep(self.max_each_delay, self.each_delay)
        return self.result
    
class UrlIdxPagesPage(PagePage):
    """
    special page to START and parse pages, store web pages for further parsing.
    get page url from given base url, make each page by given function.
    
    Note:
        - url_fn should return '' if no more url to get.
        - This class also represents a page container, 
            so the result also be list(page) for list(items) for links(result).
    """
    def __init__(self, base_url: str, url_fn: Callable[[str, int], str],
                 web_get_fn: Callable[[Any], str] = get_web_html_async,
                 *args, **kwargs) -> None:
        super().__init__([], web_get_fn=web_get_fn, *args, **kwargs)
        self.base_url = base_url
        self.url_fn = url_fn
    def parse(self, results: List[List[Union[str, etree._Element]]]):
        is_valid, idx = True, 0
        while is_valid:
            url = self.url_fn(self.base_url, idx)
            idx += 1
            if url == '' or url is None:
                is_valid = False
            else:
                # url is a list of list of links, top-level is each page,
                # second-level is each item(link of this page)
                self.url.append([url])
        return super().parse()
    def _process_parsed_data(self, *args, **kwargs):
        return self
    
class DownloadPage(PagePage):
    def __init__(self, url: List[Union[str, List[str]]] = None, folder: str = None,
                 wait_all: bool = True,
                 web_get_fn: Callable[[Any], str] = retrieve_file_async,
                 *args, **kwargs) -> None:
        super().__init__(url, web_get_fn, *args, **kwargs)
        self.folder = folder
        self.wait_all = wait_all
        self.page_folder_name: List[str] = None # folder name of each page, could be empty
        self.item_file_name: List[List[str]] = None # file name of each item, could be empty
    def parse(self, results: List[List[Union[str, etree._Element]]] = None):
        """
        download file from self.url, results(as links) or self.father_page.result_page_xpath(as links)
        Note: a page store one kind item, the content of this item could be url or list of urls.
        """
        results = self._process_url(results)
        # get web-page(s) and parse
        # NOTE: enumerate will let tqdm be unwarw of the total number of pages and items.
        for page_idx, page in zip(range(len(results)),
                                  tqdm(results, leave=False, desc=f'{self.name} fetch Page')):
            # a page store one kind item, a list of urls
            self.result.append([])
            if self.page_folder_name is None:
                page_folder = os.path.join(self.folder, str(page_idx))
            else:
                page_folder = os.path.join(self.folder, self.page_folder_name[page_idx])
            for url_idx, url in zip(range(len(page)),
                                    tqdm(page, leave=False, desc=f'{self.name} fetch Item')):
                if self.item_file_name is None:
                    file_name = requests.get(url).headers.get('Content-Disposition', None)
                    if file_name is None:
                        file_name = os.path.basename(url)
                    file_path = os.path.join(page_folder, file_name)
                else:
                    file_path = os.path.join(page_folder, self.item_file_name[page_idx][url_idx])
                # create folder if not exists
                if not os.path.exists(page_folder):
                    os.makedirs(page_folder)
                # create fetch task
                file_path = get_valid_file_path(file_path)
                task_name = self._async_task_pool.add_task(None, self.web_get_fn, url, file_path, *self.args, **self.kwargs)
                self.result[-1].append(AsyncResult(self._async_task_pool, task_name))
                random_sleep(self.max_each_delay, self.each_delay)
        # wait all tasks finished
        if self.wait_all:
            sum_tasks = sum([len(i) for i in self.result])
            not_finished, last_not_finished = sum_tasks, sum_tasks
            with tqdm(total=sum_tasks, leave=False, desc=f'{self.name} wait all') as pbar:
                while not_finished > 0:
                    not_finished = 0
                    for page in self.result:
                        for i in page:
                            if self._async_task_pool.query_task(i.name) == TaskStatus.NOT_FINISHED:
                                not_finished += 1
                    pbar.update(last_not_finished - not_finished)
                    last_not_finished = not_finished
                    time.sleep(0.1)
        return self.result
    
class ItemsPage(BasePage):
    """
    Only parse and store data of THE FATHER PAGE.
    """
    def __init__(self, xpath: str,
                 single_page_items_fn: Callable[[str], Any] = lambda x: x,
                 *args, **kwargs) -> None:
        super().__init__(xpath=xpath)
        self.single_page_items_fn = single_page_items_fn
        self.args = args
        self.kwargs = kwargs
    def parse(self, results: List[List[Union[str, etree._Element]]] = None):
        """
        parse data from results, override by subclass.
        """
        if results is None:
            results = self.father_page.result
        # detect available result and transfer to xpath object
        for page in tqdm(results, leave=False, desc=f'{self.name} parse Page'): # page is a result container of one kind item
            self.result.append([])
            self.result_page_xpath.append([])
            # exactly, page ONLY contains one kind item, so it should be a list of a result or results
            for r in tqdm(page, leave=False, desc=f'{self.name} parse Item'):
                if isinstance(r, AsyncResult): # async result
                    r = r.get() # get result from async task pool
                    if r == TaskStatus.NOT_SUCCEEDED:
                        continue # skip this item if failed
                if isinstance(r, str): # html
                    r = etree.HTML(r)
                if isinstance(r, etree._Element): # xpath object
                    self.result_page_xpath[-1].append(r) # Only store xpath object, not html
                    self.result[-1].extend(r.xpath(self.xpath))
        return self.result
    def _process_parsed_data(self, *args, **kwargs):
        results = []
        for xpath in tqdm(self.result, leave=False, desc=f'{self.name} process Item'):
            results.append(self.single_page_items_fn(
                xpath, *self.args, **self.kwargs))
        self.result = results
        return self
    
# TODO: add records
@dataclass
class Actions:
    pages: Dict[str, BasePage] = field(default_factory = lambda: {})
    results: Dict = None
    _async_task_pool: CoroutinePool = CoroutinePool()
    @staticmethod
    def get_page(name: str, father_pages: Dict[str, BasePage] = None,
                 father_page: BasePage = None) -> BasePage:
        def _extract(n: str, d: Dict[str, BasePage]):
            if n in d:
                return d[n]
            else:
                for k, v in d.items():
                    ret = _extract(n, v.next_pages)
                    if ret is not None:
                        return ret
            return None
        # check parameters
        if not isinstance(name, str):
            return put_err('name should be str, skip and return None', None)
        # try extract from father_page
        if isinstance(father_pages, dict):
            ret = _extract(name, father_pages)
        # try extract from father_page.next_pages
        if ret is None and issubclass(type(father_page), BasePage):
            ret = _extract(name, father_page.next_pages)
        # all failed, return None
        return ret
    def add_page(self, page: BasePage, father: str = '', name: str = '',
                 before_func: Callable[['Actions', 'BasePage'], bool] = None,
                 after_func: Callable[['Actions', 'BasePage'], bool] = None) -> 'Actions':
        # check parameters
        if not isinstance(name, str) or not isinstance(father, str):
            return put_err('name and father should be str, skip and return self', self)
        if not issubclass(type(page), BasePage):
            return put_err('page should be BasePage, skip and return self', self)
        if before_func is not None and not isinstance(before_func, Callable):
            return put_err('before_func should be None or Callable, skip and return self', self)
        if after_func is not None and not isinstance(after_func, Callable):
            return put_err('after_func should be None or Callable, skip and return self', self)
        # get valid father
        if father != '':
            father = self.get_page(father, self.pages)
        # add before_func and after_func
        page.before_func = before_func
        page.after_func = after_func
        # add page to pages
        page.name = name
        page._async_task_pool = self._async_task_pool
        if father is None or father == '':
            self.pages[name] = page
        else:
            father.add_next_page(name, page)
            page.father_page = father
        return self
    def del_page(self, name: str) -> None:
        raise NotImplementedError()
    def perform(self, *args, **kwargs):
        def _perform(page: Dict[str, BasePage], results: Dict, *args, **kwargs):
            for n, p in page.items():
                # check before_func
                if p.before_func is None or p.before_func(self, p):
                    # perform this page to make it's own result
                    result = p.perform(*args, **kwargs)
                    # execute after_func if exists
                    if p.after_func is not None:
                        p.after_func(self, p)
                    # perform each next_page recursively
                    if p.next_pages is not None and len(p.next_pages) > 0:
                        results[n] = {}
                        results[n] = _perform(p.next_pages, results[n], *args, **kwargs)
                    else: # or perform itself if no next_page
                        results[n] = result
            return results
        self._async_task_pool.run()
        self.results = {}
        self.results = _perform(self.pages, self.results, *args, **kwargs)
        return self.results

    
