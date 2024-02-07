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
    result = None # general result of the page, could be any type
    result_links: List[str] = field(default_factory = lambda: []) # links of parsed result
    result_page_html: List[str] = field(default_factory = lambda: []) # web-page html of parsed result
    result_page_xpath: List[etree._Element] = field(default_factory = lambda: []) # web-page xpath object of parsed result
    father_page: 'BasePage' = None # father page of this page
    next_pages: Dict[str, 'BasePage'] = field(default_factory = lambda: {}) # next pages of this page
    before_func: Callable[['BasePage'], bool] = None # before function to execute before parsing, check if need to parse this page
    after_func: Callable[['BasePage'], bool] = None # after function to execute after parsing, do something after parsing this page
    def parse(self, html: str = None, xpath_obj: etree._Element = None) -> None:
        if html is not None and isinstance(html, str):
            xpath_obj = etree.HTML(html)
        if xpath_obj is not None and isinstance(xpath_obj, etree._Element):
            self.results = xpath_obj.xpath(self.xpath)
            return self.results
        else:
            # do not waring, return None
            return None
    def add_next_page(self, name:str, page: 'BasePage') -> None:
        if isinstance(name, str):
            self.next_pages[name] = page
            return True
        else:
            return put_err('name should be str, skip and return False', False)
    def _process_parsed_data(self):
        """
        process parsed data, could be override by subclass
        """
        pass
    def perform(self, html: str = None, xpath_obj: etree._Element = None):
        self.result =  self.parse(html, xpath_obj)
        self._process_parsed_data()
        return self.result
    
class PagePage(BasePage):
    """
    Only get and store one web page.
    """
    def __init__(self, url: str,
                 web_get_fn: Callable[[Any], str] = get_url_page_s,
                 *args, **kwargs) -> None:
        super().__init__()
        self.url = url
        self.web_get_fn = web_get_fn
        self.args = args
        self.kwargs = kwargs
    def _process_parsed_data(self, url: str = None):
        url = self.url if not isinstance(url, str) else url
        self.result = self.web_get_fn(url, *self.args, **self.kwargs)
        self.result_page_html.append(self.web_get_fn(self.url, *self.args, **self.kwargs))
        self.result_page_xpath.append(etree.HTML(self.result_page_html[0]))
        return self
    
class UrlIdxPagesPage(PagePage):
    """
    special page to parse pages, store web pages for further parsing.
    get page url from given base url, make each page by given function.
    """
    def __init__(self, base_url: str, url_fn: Callable[[str, int], str],
                 web_get_fn: Callable[[Any], str] = get_url_page_s,
                 *args, **kwargs) -> None:
        super().__init__('', web_get_fn=web_get_fn, *args, **kwargs)
        self.base_url = base_url
        self.url_fn = url_fn
    def _process_parsed_data(self):
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
    Only store data of one page.
    """
    def __init__(self, xpath: str) -> None:
        super().__init__(xpath=xpath)
    
class LinksPage(ItemsPage):
    """
    Special page to parse links, links stand for web pages.
    """
    def __init__(self, xpath: str,
                 single_link_str_fn: Callable[[str], str] = lambda x: x,
                 web_get_fn: Callable[[Any], str] = get_url_page_s,
                 *args, **kwargs) -> None:
        super().__init__(xpath=xpath)
        self.single_link_str_fn = single_link_str_fn
        self.web_get_fn = web_get_fn
        self.args = args
        self.kwargs = kwargs
    def _process_parsed_data(self):
        if isinstance(self.result, list):
            for link in self.result:
                if isinstance(link, str):
                    link = self.single_link_str_fn(link)
                    self.result_links.append(self.single_link_str_fn(link))
                    self.result_page_html.append(self.web_get_fn(
                        self.result_links[-1], *self.args, **self.kwargs))
                    self.result_page_xpath.append(
                        etree.HTML(self.result_page_html[-1]))
        return self
    
class TextsPage(BasePage):
    """
    Only store text data of one page.
    """
    def __init__(self) -> None:
        super().__init__()
        pass
    
class ImagesPage(BasePage):
    """
    Only store image data of one page.
    """
    def __init__(self) -> None:
        super().__init__()
        pass
    
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
    def del_page(self, name: str) -> None:
        raise NotImplementedError()
    def perform(self, html: str = None, xpath_obj: etree._Element = None):
        def _perform(page: Dict[str, BasePage], results: Dict,
                     html: str = None, xpath_obj: etree._Element = None):
            for n, p in page.items():
                # check before_func
                if p.before_func is None or p.before_func(self):
                    # perform each next_page recursively
                    if p.next_pages is not None and len(p.next_pages) > 0:
                        results[n] = {}
                        results[n] = _perform(p.next_pages, results[n],
                                              html, xpath_obj)
                    else: # or perform itself if no next_page
                        results[n] = p.perform(html, xpath_obj)
                # execute after_func if exists                  
                if p.after_func is not None:
                    p.after_func(self)
            return results
        self.results = {}
        self.results = _perform(self.pages, self.results, html, xpath_obj)
        return self.results
                
    
    
if __name__ == '__main__':
    # dev code
    action = Actions()
    action.add_page(page=PagePage(url='https://www.kuaidaili.com/free/inha/'), name='main')
    action.add_page(page=LinksPage(xpath = ''), name='links', father='main')
    results: Dict = action.perform()