
import re

from bs4 import BeautifulSoup
from lxml import etree

if __name__ == '__main__':
    from mbapy.base import Configs, get_default_for_bool, put_err

    # functon assembly
else:
    from ..base import Configs, get_default_for_bool, put_err

    # functon assembly

def get_between(string:str, head:str, tail:str,
               headRFind:bool = False, tailRFind:bool = True,
               ret_head:bool = False, ret_tail:bool = False,
               find_tail_from_head = False):
    """
    Finds and returns a substring between two given strings in a larger string.

    Parameters:
        string (str): The larger string to search within.
        head (str): The starting string to search for.
        tail (str): The ending string to search for.
        headRFind (bool, optional): If True, searches for the last occurrence of head. Defaults to False.
        tailRFind (bool, optional): If True, searches for the last occurrence of tail. Defaults to True.
        ret_head (bool, optional): If True, includes the head in the returned substring. Defaults to False.
        ret_tail (bool, optional): If True, includes the tail in the returned substring. Defaults to False.
        find_tail_from_head (bool, optional): If True, searches for the tail starting from the position of the head. Defaults to False.

    Returns:
        str: The substring between the head and tail. If the head or tail is not found, an error message is returned.
    """
    headIdx = string.rfind(head) if headRFind else string.find(head)
    if find_tail_from_head:
        tailIdx = string[headIdx+len(head):].find(tail) + headIdx + len(head)
    else:
        tailIdx = string.rfind(tail) if tailRFind else string.find(tail)
    if headIdx == -1 or tailIdx == -1:
        return put_err(f"{head if headIdx == -1 else tail:s} not found, return string", string)
    if headIdx == tailIdx:
        return put_err(f"headIdx == tailIdx with head:{head:s} and string:{string:s}, return ''", '')
    idx1 = headIdx if ret_head else headIdx+len(head)
    idx2 = tailIdx+len(tail) if ret_tail else tailIdx
    return string[idx1:idx2]

def get_between_re(string:str, head:str, tail:str,
                   head_r:bool = False, tail_r:bool = True,
                   ret_head:bool = False, ret_tail:bool = False):
    """
    Return the substring between two given patterns in a string.

    Parameters:
        string (str): The input string.
        head (str): The starting pattern.
        tail (str): The ending pattern.
        head_r (bool, optional): If True, include the head pattern in the result. Defaults to False.
        tail_r (bool, optional): If True, include the tail pattern in the result. Defaults to True.
        ret_head (bool, optional): If True, return the head pattern along with the substring. Defaults to False.
        ret_tail (bool, optional): If True, return the tail pattern along with the substring. Defaults to False.

    Returns:
        str: The substring between the head and tail patterns.

    Raises:
        ValueError: If the head or tail pattern is not found in the string.

    Examples:
        >>> get_between_re("Hello world!", "Hello", "!")
        ' world'
        >>> get_between_re("Hello world!", "Hello", "!", head_r=True, ret_tail=True)
        'Hello world!'
    """
    h = re.compile(head).search(string) if len(head) > 0 else ''
    t = re.compile(tail).search(string)
    if h is None or t is None:
        return put_err(f"not found with head:{head:s} and tail:{tail:s}, return string", string)
    else:
        h, t = h.group(0) if h != '' else '', t.group(0)
    return get_between(string, h, t, head_r, tail_r, ret_head, ret_tail)

def parse_xpath_info(xpath_search_key:str, xpath_obj, is_single: bool = True):
    search_result = xpath_obj.xpath(xpath_search_key)
    if is_single:
        return get_default_for_bool(search_result, [''])[0].strip()
    return get_default_for_bool(search_result, [''])


__all__ = [
    'get_between',
    'get_between_re',
    'parse_xpath_info',
]