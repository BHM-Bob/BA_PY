# mbapy.web_utils.request

This module provides functions for web scraping and browser automation.  

## Functions

### get_url_page(url:str, coding = 'gbk') -> str

Given a URL and a character encoding, this function returns the decoded content of the page.  

Parameters:  
- url (str): The URL to be visited.  
- coding (str, optional): The character encoding of the page. Defaults to 'gbk'.  

Returns:  
- str: The decoded content of the page.  

Example:  
```python
get_url_page('https://www.example.com', 'utf-8')
```

### get_url_page_s(url:str, coding = 'gbk') -> str

Returns the HTML page content from the given URL.  

Parameters:  
- url (str): The URL of the web page to retrieve.  
- coding (str, optional): The encoding of the HTML content. Defaults to 'gbk'.  

Returns:  
- str: The HTML page content.  

Example:  
```python
get_url_page_s('https://www.example.com', 'utf-8')
```

### get_url_page_b(url:str, return_html_text:bool = False, debug:bool = False, coding = 'gbk') -> BeautifulSoup object or tuple

This function takes a URL and returns the HTML page of the URL in a BeautifulSoup object. It has the option to return a string of the HTML text as well. It also takes optional arguments for debugging and specifying the coding of the page to be retrieved.  

Parameters:  
- url (str): The URL to retrieve.  
- return_html_text (bool): Whether or not to return the HTML text as a string. Defaults to False.  
- debug (bool): Whether to use debug mode. Defaults to False.  
- coding (str): The coding of the page to retrieve. Defaults to 'gbk'.  

Returns:  
- BeautifulSoup object: A BeautifulSoup object representing the HTML page of the URL.  
- str (optional): If `return_html_text` is True, it returns a string of the HTML text.  

Example:  
```python
get_url_page_b('https://www.example.com', return_html_text=True, debug=True, coding='utf-8')
```

### get_url_page_se(browser, url:str, return_html_text:bool = False, debug = False) -> BeautifulSoup object or tuple

Retrieves the HTML source code of a webpage given its URL using a webdriver instance.  

Parameters:  
- browser: The webdriver instance.  
- url (str): The URL of the webpage to retrieve.  
- return_html_text (bool): Whether or not to return the HTML source code as a string.  
- debug (bool): Whether or not to enable debug mode.  

Returns:  
- BeautifulSoup object: A BeautifulSoup object representing the parsed HTML.  
- str (optional): If `return_html_text` is True, it returns a string of the HTML source code.  

Example:  
```python
from selenium import webdriver

browser = webdriver.Chrome()
get_url_page_se(browser, 'https://www.example.com', return_html_text=True, debug=True)
```

### get_browser(browser:str, browser_driver_path:str = None, options =['--no-sandbox', '--headless', f"--user-agent={Configs.web.chrome_driver_path:s}"], use_undetected:bool = False) -> Browser

Initializes and returns a Selenium browser instance based on the specified browser name and driver path.  

Parameters:  
- browser (str): The name of the browser. Currently supported values are 'Edge' and 'Chrome'.  
- browser_driver_path (str, optional): The path to the browser driver executable. Defaults to None.  
- options (list, optional): A list of additional options to be passed to the browser. Defaults to ['--no-sandbox', '--headless'].  
- use_undetected (bool): Whether or not to use undetected_chromedriver. Defaults to False.  

Returns:  
- Browser: An instance of the Selenium browser based on the specified browser name and options.  

Example:  
```python
browser = get_browser('Chrome', browser_driver_path='path/to/chromedriver.exe')
```

### add_cookies(browser, cookies_path:str = None, cookies_string:str = None) -> None

Adds cookies to the browser.  

Parameters:  
- browser (object): The browser object to add the cookies to.  
- cookies_path (str, optional): The path to the file containing the cookies. Defaults to None.  
- cookies_string (str, optional): A string of cookies to add. Defaults to None.  

Returns:  
- None

Raises:  
- ValueError: If no cookies are specified.  

Description:  
This function adds cookies to the specified browser object. It can add cookies from a file or from a string. If the `cookies_path` parameter is provided, the function will check if the file exists and is valid before parsing and adding the cookies. If the `cookies_string` parameter is provided, the function will directly parse and add the cookies from the string. If neither `cookies_path` nor `cookies_string` is provided, a `ValueError` will be raised.  

The function internally uses the `_parse_cookies` function to parse and add the cookies.  

Example Usage:  
```python
# Add cookies from a file
add_cookies(browser, cookies_path="cookies.txt")

# Add cookies from a string
add_cookies(browser, cookies_string="cookie1=value1; cookie2=value2")
```

### transfer_str2by(by:str) -> By

Transfers a string representation of a 'By' identifier to the corresponding 'By' object.  

Parameters:  
- by (str): The string representation of the 'By' identifier. Supported values are 'class', 'css', and 'xpath'.  

Returns:  
- By: The corresponding 'By' object.  

Raises:  
- ValueError: If the 'by' parameter is not one of the valid 'By' identifier strings.  

Example:  
```python
transfer_str2by('class')
```

### wait_for_amount_elements(browser, by, element, count, timeout=10) -> list

Waits for a specified number of elements to be present on the page.  

Args:  
- browser (WebDriver): The WebDriver instance used to interact with the browser.  
- by (str): The method used to locate the elements (e.g. "class", "css", "xpath").  
- element (str): The value used to locate the elements (e.g. the ID, class name, or xpath expression).  
- count (int): The number of elements to wait for.  
- timeout (int, optional): The maximum amount of time (in seconds) to wait for the elements to be present. Defaults to 10.  

Returns:  
- list: A list of WebElement objects representing the elements found.  

Raises:  
- TimeoutException: If the elements are not found within the specified timeout.  

Example:  
```python
wait_for_amount_elements(browser, 'class', 'example', 5, timeout=20)
```

### send_browser_key(browser, keys:str, element:str, by:str = 'class', wait:int = 5) -> None

Sends a sequence of keystrokes to a specified element in a web browser.  

Args:  
- browser (WebDriver): The web browser instance.  
- keys (str): The sequence of keystrokes to send.  
- element (str): The identifier of the element to send the keystrokes to.  
- by (str, optional): The method used to locate the element. Defaults to 'class'.  
- wait (int, optional): The maximum time in seconds to wait for the element to be present. Defaults to 5.  

Returns:  
- None

Example:  
```python
send_browser_key(browser, 'example', 'input', by='class', wait=10)
```

### click_browser(browser, element:str, by:str = 'class', wait = 5) -> None

Clicks on a specified element in a browser.  

Args:  
- browser: The browser object on which to perform the click.  
- element (str): The identifier of the element to click on.  
- by (str, optional): The method to locate the element. Defaults to 'class'.  
- wait (int, optional): The maximum time to wait for the element to be present. Defaults to 5.  

Returns:  
- None

Example:  
```python
click_browser(browser, 'button', by='class', wait=10)
```

### scroll_browser(browser, scroll='bottom', duration=0) -> None

Scrolls the browser window either to the bottom or by a specific amount.  

Parameters:  
- browser (object): The browser object.  
- scroll (str|int): The scroll behavior. If set to 'bottom', the browser will be scrolled to the bottom. If set to an integer, the browser will be scrolled by that amount (in pixels).  
- duration (int): The duration (in seconds) for which the scroll operation should be performed. If set to 0, the scroll operation will be performed instantly.  

Returns:  
- None

Raises:  
- ValueError: If the scroll type is unknown.  

Example:  
```python
scroll_browser(browser, scroll='bottom', duration=5)
```