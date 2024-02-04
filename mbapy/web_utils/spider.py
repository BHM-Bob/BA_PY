import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
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
    xpath: str = ''
    results: List = None
    father_page: 'BasePage' = None
    next_pages: Dict[str, 'BasePage'] = None
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
        pass
    
class LinksPage(BasePage):
    def __init__(self) -> None:
        pass
    
class ItemsPage(BasePage):
    def __init__(self) -> None:
        pass
    
class TextsPage(BasePage):
    def __init__(self) -> None:
        pass
    
class ImagesPage(BasePage):
    def __init__(self) -> None:
        pass
    
class Actions:
    def __init__(self) -> None:
        pass
    def add_action(self, page: BasePage,father: str = '', name: str = '',
                   before_func: Callable[['Actions'], bool] = None,
                   after_func: Callable[['Actions'], bool] = None) -> 'Actions':
        pass
    def del_action(self, name: str) -> None:
        pass
    def perform(self):
        pass