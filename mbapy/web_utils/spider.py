import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union, Callable

from lxml import etree

if __name__ == '__main__':
    from mbapy.base import (Configs, check_parameters_len,
                            check_parameters_path, put_err)
    from mbapy.web_utils.request import get_url_page_s
else:
    from ..base import (Configs, check_parameters_len, check_parameters_path,
                        put_err)
    from .request import get_url_page_s
    
@dataclass
class BasePage:
    xpath: str = '' # xpath expression to extract data
    result: List[Any] = field(default_factory = lambda: []) # general result of the page, could be any type
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
    def parse(self, results: List[Union[str, etree._Element]] = None) -> None:
        """
        parse data from results, override by subclass.
        In BasePage.perform, it will call this function to parse data and store in self.result..
        """
    def _process_parsed_data(self, *args, **kwargs):
        """
        process parsed data, could be override by subclass
        """
    def perform(self, *args, results: List[Union[str, etree._Element]] = None, **kwargs):
        self.result =  self.parse(results)
        self._process_parsed_data(*args, **kwargs)
        return self.result
    
class PagePage(BasePage):
    """
    Only START and store a new web page.
    """
    def __init__(self, url: str,
                 web_get_fn: Callable[[Any], str] = get_url_page_s,
                 *args, **kwargs) -> None:
        super().__init__()
        self.url = url
        self.web_get_fn = web_get_fn
        self.args = args
        self.kwargs = kwargs
    def parse(self, results: List[Union[str, etree._Element]] = None):
        """
        get new web-page(s) from self.url, results(as links) or self.father_page.result_page_xpath(as links)
        """
        if self.url == '':
            if results is None:
                results = self.father_page.result
        else:
            results = [self.url]
        self.result = [self.web_get_fn(r, *self.args, **self.kwargs) for r in results]
        self.result_page_html = [r for r in self.result if isinstance(r, str)]
        self.result_page_xpath = [etree.HTML(r) for r in self.result_page_html if isinstance(r, str)]
        return self
    
class UrlIdxPagesPage(PagePage):
    """
    special page to START and parse pages, store web pages for further parsing.
    get page url from given base url, make each page by given function.
    """
    def __init__(self, base_url: str, url_fn: Callable[[str, int], str],
                 web_get_fn: Callable[[Any], str] = get_url_page_s,
                 *args, **kwargs) -> None:
        super().__init__('', web_get_fn=web_get_fn, *args, **kwargs)
        self.base_url = base_url
        self.url_fn = url_fn
    def parse(self, results: List[Union[str, etree._Element]]):
        raise NotImplementedError()
    def _process_parsed_data(self, *args, **kwargs):
        is_valid, results = True, []
        while is_valid:
            super()._process_parsed_data(self.url_fn(self.url, len(self.result_links)))
            if self.result is not None:
                results.append(self.result)
            else:
                is_valid = False
        self.result = results
        return self
    
class ItemsPage(BasePage):
    """
    Only parse and store data of THE FATHER PAGE.
    """
    def __init__(self, xpath: str,
                 single_item_fn: Callable[[str], Any] = lambda x: x,
                 *args, **kwargs) -> None:
        super().__init__(xpath=xpath)
        self.single_item_fn = single_item_fn
        self.args = args
        self.kwargs = kwargs
    def parse(self, results: List[Union[str, etree._Element]] = None):
        """
        parse data from results, override by subclass.
        """
        if results is None:
            results = self.father_page.result_page_xpath
        if len(results) == 0:
            results = self.father_page.result_page_html # TODO: check if this is correct
        # detect available result and transfer to xpath object
        for r in results:
            if isinstance(r, str): # html
                r = etree.HTML(r)
            if isinstance(r, etree._Element): # xpath object
                self.result_page_xpath.append(r)
                self.result_page_html.append(etree.tostring(r, encoding='unicode')) # TODO: check if this is correct
                self.result.append(r.xpath(self.xpath))
        return self.result
    def _process_parsed_data(self, *args, **kwargs):
        results = []
        for xpath in self.result:
            results.append(self.single_item_fn(
                xpath, *self.args, **self.kwargs))
        self.result = results
        return self
    
@dataclass
class Actions:
    pages: Dict[str, BasePage] = field(default_factory = lambda: {})
    results: Dict = None
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
                 before_func: Callable[['Actions'], bool] = None,
                 after_func: Callable[['Actions'], bool] = None) -> 'Actions':
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
                if p.before_func is None or p.before_func(self):
                    # perform this page to make it's own result
                    result = p.perform(*args, **kwargs)
                    # perform each next_page recursively
                    if p.next_pages is not None and len(p.next_pages) > 0:
                        results[n] = {}
                        results[n] = _perform(p.next_pages, results[n],
                                              *args, **kwargs)
                    else: # or perform itself if no next_page
                        results[n] = result
                # execute after_func if exists                  
                if p.after_func is not None:
                    p.after_func(self)
            return results
        self.results = {}
        self.results = _perform(self.pages, self.results, *args, **kwargs)
        return self.results
    
    
if __name__ == '__main__':
    # dev code
    ip_xpath = '//*[@id="list"]/div[2]/table/tbody/tr[3]/td[1]'
    port = '//*[@id="list"]/div[2]/table/tbody/tr[1]/td[2]'
    text_fn = lambda x: x[0].text
    action = Actions()
    action.add_page(page=PagePage(url='https://www.kuaidaili.com/free/inha/'), name='main')
    action.add_page(page=ItemsPage(xpath = ip_xpath, single_item_fn = text_fn), name='ip', father='main')
    action.add_page(page=ItemsPage(xpath = port, single_item_fn = text_fn), name='port', father='main')
    results: Dict = action.perform()