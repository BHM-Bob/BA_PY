.. module:: module_name

   :platform: Unix, Windows
   :synopsis: This module provides functions for manipulating strings and parsing XML using XPath.

Module Description
------------------

This module contains functions for manipulating strings and parsing XML using XPath. It includes the following functions:

- `get_between`: Finds and returns a substring between two given strings in a larger string.
- `get_between_re`: Returns the substring between two given patterns in a string using regular expressions.
- `parse_xpath_info`: Parses XML using XPath and returns the search result.

Function Details
----------------

.. function:: get_between(string:str, head:str, tail:str, headRFind:bool=False, tailRFind:bool=True, ret_head:bool=False, ret_tail:bool=False, find_tail_from_head:bool=False)

   Finds and returns a substring between two given strings in a larger string.

   :param string: The larger string to search within.
   :type string: str
   :param head: The starting string to search for.
   :type head: str
   :param tail: The ending string to search for.
   :type tail: str
   :param headRFind: If True, searches for the last occurrence of head. Defaults to False.
   :type headRFind: bool, optional
   :param tailRFind: If True, searches for the last occurrence of tail. Defaults to True.
   :type tailRFind: bool, optional
   :param ret_head: If True, includes the head in the returned substring. Defaults to False.
   :type ret_head: bool, optional
   :param ret_tail: If True, includes the tail in the returned substring. Defaults to False.
   :type ret_tail: bool, optional
   :param find_tail_from_head: If True, searches for the tail starting from the position of the head. Defaults to False.
   :type find_tail_from_head: bool, optional
   :return: The substring between the head and tail. If the head or tail is not found, an error message is returned.
   :rtype: str

.. function:: get_between_re(string:str, head:str, tail:str, head_r:bool=False, tail_r:bool=True, ret_head:bool=False, ret_tail:bool=False)

   Returns the substring between two given patterns in a string using regular expressions.

   :param string: The input string.
   :type string: str
   :param head: The starting pattern.
   :type head: str
   :param tail: The ending pattern.
   :type tail: str
   :param head_r: If True, include the head pattern in the result. Defaults to False.
   :type head_r: bool, optional
   :param tail_r: If True, include the tail pattern in the result. Defaults to True.
   :type tail_r: bool, optional
   :param ret_head: If True, return the head pattern along with the substring. Defaults to False.
   :type ret_head: bool, optional
   :param ret_tail: If True, return the tail pattern along with the substring. Defaults to False.
   :type ret_tail: bool, optional
   :return: The substring between the head and tail patterns.
   :rtype: str
   :raises ValueError: If the head or tail pattern is not found in the string.
   :examples:
      - Example 1: get_between_re("Hello world!", "Hello", "!") returns ' world'
      - Example 2: get_between_re("Hello world!", "Hello", "!", head_r=True, ret_tail=True) returns 'Hello world!'

.. function:: parse_xpath_info(xpath_search_key:str, xpath_obj, is_single:bool=True)

   Parses XML using XPath and returns the search result.

   :param xpath_search_key: The XPath expression to search for.
   :type xpath_search_key: str
   :param xpath_obj: The XML object to search within.
   :type xpath_obj: lxml.etree._Element
   :param is_single: If True, returns a single result. Defaults to True.
   :type is_single: bool, optional
   :return: The search result.
   :rtype: str