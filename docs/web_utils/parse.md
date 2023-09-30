# mbapy.web_utils.parse

This module provides utility functions for string manipulation and parsing.  

## Functions

### get_between(string:str, head:str, tail:str, headRFind:bool = False, tailRFind:bool = True, ret_head:bool = False, ret_tail:bool = False, find_tail_from_head = False) -> str

Finds and returns a substring between two given strings in a larger string.  

Parameters:  
- string (str): The larger string to search within.  
- head (str): The starting string to search for.  
- tail (str): The ending string to search for.  
- headRFind (bool, optional): If True, searches for the last occurrence of head. Defaults to False.  
- tailRFind (bool, optional): If True, searches for the last occurrence of tail. Defaults to True.  
- ret_head (bool, optional): If True, includes the head in the returned substring. Defaults to False.  
- ret_tail (bool, optional): If True, includes the tail in the returned substring. Defaults to False.  
- find_tail_from_head (bool, optional): If True, searches for the tail starting from the position of the head. Defaults to False.  

Returns:  
- str: The substring between the head and tail. If the head or tail is not found, an error message is returned.  

Example:  
```python
get_between("Hello world!", "Hello", "!")
# Output: ' world'
```

### get_between_re(string:str, head:str, tail:str, head_r:bool = False, tail_r:bool = True, ret_head:bool = False, ret_tail:bool = False) -> str

Return the substring between two given patterns in a string.  

Parameters:  
- string (str): The input string.  
- head (str): The starting pattern.  
- tail (str): The ending pattern.  
- head_r (bool, optional): If True, include the head pattern in the result. Defaults to False.  
- tail_r (bool, optional): If True, include the tail pattern in the result. Defaults to True.  
- ret_head (bool, optional): If True, return the head pattern along with the substring. Defaults to False.  
- ret_tail (bool, optional): If True, return the tail pattern along with the substring. Defaults to False.  

Returns:  
- str: The substring between the head and tail patterns.  

Raises:  
- ValueError: If the head or tail pattern is not found in the string.  

Examples:  
```python
get_between_re("Hello world!", "Hello", "!")
# Output: ' world'
```

### parse_xpath_info(xpath_search_key:str, xpath_obj, is_single: bool = True) -> str

Parse the result of an XPath search and return the desired information.  

Parameters:  
- xpath_search_key (str): The XPath search key.  
- xpath_obj: The object on which to perform the XPath search.  
- is_single (bool, optional): If True, return a single result. Defaults to True.  

Returns:  
- str: The parsed information.  

Example:  
```python
parse_xpath_info("//div[@class='title']", xpath_obj)
```
