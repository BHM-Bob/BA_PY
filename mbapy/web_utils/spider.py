import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Callable

from lxml import etree

if __name__ == '__main__':
    from mbapy.base import (Configs, check_parameters_len,
                            check_parameters_path, put_err)
else:
    from ..base import (Configs, check_parameters_len, check_parameters_path,
                        put_err)
    
@dataclass
class BasePage:
    xpath: str = '' # xpath expression to extract data
    result = None # general result of the page, could be any type
    result_links: List[str] = field(default_factory = lambda: []) # links of parsed result
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
            return put_err(f'html or xpath_obj should be str or etree._Element')
    def add_next_page(self, name:str, page: 'BasePage') -> None:
        if isinstance(name, str):
            self.next_pages[name] = page
            return True
        else:
            return put_err('name should be str, skip and return False', False)
    def perform(self, html: str = None, xpath_obj: etree._Element = None):
        return self.parse(html, xpath_obj)        
    
class PagesPage(BasePage):
    def __init__(self, ) -> None:
        super().__init__()
        pass
    
class ItemsPage(BasePage):
    def __init__(self) -> None:
        super().__init__()
        pass
    
class LinksPage(BasePage):
    def __init__(self) -> None:
        super().__init__()
        pass
    
class TextsPage(BasePage):
    def __init__(self) -> None:
        super().__init__()
        pass
    
class ImagesPage(BasePage):
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
        if not isinstance(page, BasePage):
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
                        for k, v in p.next_pages.items():
                            self.results[n][k] = _perform(v, html, xpath_obj)
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
    action.add_page(page=PagesPage(), name='main')
    action.add_page(page=ItemsPage(), name='s-page', father='main')
    action.add_page(page=TextsPage(), name='s-title', father='s-page')
    action.add_page(page=LinksPage(), name='s-link', father='s-page')
    results: Dict = action.perform()