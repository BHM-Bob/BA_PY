# mbapy.web_utils.request

This module provides functions for web scraping and browser automation.  

### random_sleep -> None
**General Description**

This function introduces a random sleep interval between `min_t` and `max_t` seconds.

#### Params
- max_t (int): The maximum time in seconds for the random sleep interval. Defaults to 10.
- min_t (int): The minimum time in seconds for the random sleep interval. Defaults to 1.

#### Returns
None

#### Notes
This function is useful for introducing random delays in a program, which can be helpful for simulating human-like behavior when making requests or performing other actions.

#### Example
```python
random_sleep(5, 2)
```

### get_requests_retry_session -> Session
**General Description**

This function creates and returns a session object with automatic request retry functionality.

#### Params
- retries (int): The number of times to retry a request in case of failure. Defaults to 3.
- backoff_factor (float): The factor by which the backoff time between retries increases. Defaults to 0.3.
- status_forcelist (tuple): The HTTP status codes that should trigger a retry. Defaults to (500, 502, 504).
- session (Session): An existing session object to use. If not provided, a new session object will be created.

#### Returns
- Session: The session object with retry functionality.

#### Notes
This function uses the `requests` library to create a session object with automatic request retry functionality based on the provided parameters.

#### Example
```python
session = get_requests_retry_session(retries=5, backoff_factor=0.5)
```

### get_url_page -> str
**General Description**

Given a URL and a character encoding, this function returns the decoded content of the page.

#### Params
- url (str): A string representing the URL to be visited.
- coding (str): A string representing the character encoding of the page. Default is 'gbk'.

#### Returns
- str: The decoded content of the page.

#### Notes
This function uses the `urllib` library to retrieve the content of the specified URL and decode it using the provided character encoding.

#### Example
```python
content = get_url_page('https://example.com', 'utf-8')
```

### get_url_page_s -> str
**General Description**

This function returns the HTML page content from the given URL. The function takes two parameters: `url` and `coding`. It tries to retrieve the HTML page content from the given URL using the `get_url_page` function, with the specified encoding. If it fails, it returns '-html-None'.

#### Params
- url (str): A string representing the URL of the web page to retrieve.
- coding (str): A string representing the encoding of the HTML content. Default is 'gbk'.

#### Returns
- str: The HTML page content from the given URL, or '-html-None' if retrieval fails.

#### Notes
This function is a wrapper around the `get_url_page` function and provides a fallback option if the retrieval fails.

### get_url_page_b -> BeautifulSoup object, str (optional)
**General Description**

This function takes a URL and returns the HTML page of the URL in a BeautifulSoup object. It has the option to return a string of the HTML text as well. It also takes optional arguments for debugging and specifying the coding of the page to be retrieved.

#### Params
- url (str): A string representing the URL to retrieve.
- return_html_text (bool): A boolean indicating whether or not to return the HTML text as a string. Defaults to False.
- debug (bool): A boolean indicating whether to use debug mode. Defaults to False.
- coding: The coding of the page to retrieve. Defaults to 'gbk'.

#### Returns
- BeautifulSoup object: A BeautifulSoup object representing the HTML page of the URL.
- str (optional): If `return_html_text` is True, it returns a string of the HTML text.

#### Notes
This function uses the `get_url_page` and `BeautifulSoup` to retrieve the HTML content of the specified URL and parse it into a BeautifulSoup object. It also provides the option to return the HTML text as a string.

### get_url_page_se -> BeautifulSoup object, str (optional)
**General Description**

Retrieves the HTML source code of a webpage given its URL using a webdriver instance.

#### Params
- browser: The webdriver instance.
- url (str): The URL of the webpage to retrieve.
- return_html_text (bool): Whether or not to return the HTML source code as a string.
- debug (bool): Whether or not to enable debug mode.

#### Returns
- If `return_html_text` is True, returns a tuple containing a BeautifulSoup object representing the parsed HTML and the raw HTML source code as a string. Otherwise, returns a BeautifulSoup object representing the parsed HTML.

#### Notes
This function uses a webdriver instance to retrieve the HTML source code of a webpage and parse it into a BeautifulSoup object. It provides the option to return the HTML source code as a string.

### get_browser -> Browser
**General Description**

Initializes and returns a Selenium browser instance based on the specified browser name and driver path.

#### Params
- browser (str): The name of the browser. Currently supported values are 'Edge' and 'Chrome'.
- browser_driver_path (str, optional): The path to the browser driver executable. Defaults to None.
- options (list, optional): A list of additional options to be passed to the browser. Defaults to ['--no-sandbox', '--headless'].
- use_undetected (bool): Whether to use undetected_chromedriver or not.

#### Returns
- Browser: An instance of the Selenium browser based on the specified browser name and options.

#### Notes
This function initializes and returns a Selenium browser instance based on the specified browser name and driver path. It also provides the option to use undetected_chromedriver.

### add_cookies -> None
**General Description**

Adds cookies to the browser.

#### Params
- browser (object): The browser object to add the cookies to.
- cookies_path (str, optional): The path to the file containing the cookies. Defaults to None.
- cookies_string (str, optional): A string of cookies to add. Defaults to None.

#### Returns
None

#### Raises
- ValueError: If no cookies are specified.

#### Description
This function adds cookies to the specified browser object. It can add cookies from a file or from a string. If the `cookies_path` parameter is provided, the function will check if the file exists and is valid before parsing and adding the cookies. If the `cookies_string` parameter is provided, the function will directly parse and add the cookies from the string. If neither `cookies_path` nor `cookies_string` is provided, a `ValueError` will be raised.

### transfer_str2by -> By
**General Description**

Transfers a string representation of a 'By' identifier to the corresponding 'By' object.

#### Params
- by (str): The string representation of the 'By' identifier.
    - support class('By.CLASS_NAME'), css('By.CSS_SELECTOR'), xpath('By.XPATH')

#### Returns
- By: The corresponding 'By' object.

#### Raises
- ValueError: If the 'by' parameter is not one of the valid 'By' identifier strings.

### wait_for_amount_elements -> list
**General Description**

Waits for a specified number of elements to be present on the page.

#### Params
- browser (WebDriver): The WebDriver instance used to interact with the browser.
- by (str): The method used to locate the elements (e.g. "class", "css", "xpath").
- element (str): The value used to locate the elements (e.g. the ID, class name, or xpath expression).
- count (int): The number of elements to wait for.
- timeout (int, optional): The maximum amount of time (in seconds) to wait for the elements to be present. Defaults to 10.

#### Returns
- list: A list of WebElement objects representing the elements found.

#### Raises
- TimeoutException: If the elements are not found within the specified timeout.

### send_browser_key -> None
**General Description**

Sends a sequence of keystrokes to a specified element in a web browser.

#### Params
- browser (WebDriver): The web browser instance.
- keys (str): The sequence of keystrokes to send.
- element (str): The identifier of the element to send the keystrokes to.
- by (str, optional): The method used to locate the element. Defaults to 'class'.
- wait (int, optional): The maximum time in seconds to wait for the element to be present. Defaults to 5.

#### Returns
None

### click_browser -> None
**General Description**

Clicks on a specified element in a browser.

#### Params
- browser: The browser object on which to perform the click.
- element (str): The identifier of the element to click on.
- by (str, optional): The method to locate the element. Defaults to 'class'.
- wait (int, optional): The maximum time to wait for the element to be present. Defaults to 5.

#### Returns
None

### scroll_browser -> None
**General Description**

Scroll the browser window either to the bottom or by a specific amount.

#### Params
- browser (object): The browser object.
- scroll (str|int): The scroll behavior. If set to 'bottom', the browser will be scrolled to the bottom. If set to an integer, the browser will be scrolled by that amount (in pixels).
- duration (int): The duration (in seconds) for which the scroll operation should be performed. If set to 0, the scroll operation will be performed instantly.

#### Returns
None

#### Raises
- ValueError: If the scroll type is unknown.

### download_streamly -> None
**General Description**

Downloads a file from the given URL to the specified path using a streaming approach.

#### Params
- url (str): The URL of the file to be downloaded.
- path (str): The path where the downloaded file will be saved.
- session (object): The session object used for making the HTTP request.

#### Returns
None

### BrowserActionWarpper -> function
**General Description**

Decorator function and adds sleep functionality before and after the function call.

#### Warpped functon args
- self: The instance of the class that the function is a method of.
- *args: Positional arguments to be passed to the wrapped function.
- sleep_before (Union[None, int, float, Tuple[int, int]]): Optional. The amount of time to sleep before the function call.
- sleep_after (Union[None, int, float, Tuple[int, int]]): Optional. The amount of time to sleep after the function call.
- **kwargs: Keyword arguments to be passed to the wrapped function.

#### Returns
The return value of the wrapped function.

### BrowserElementActionWarpper -> function
**General Description**

Decorator function, find the element and adds sleep functionality before and after the function call.

#### Warpped functon args
- self: The instance of the class.
- *args: Positional arguments passed to the wrapped function.
- element (Union[None, str, ElementType]): The element to locate on the page. Defaults to None.
- by (str): The locator strategy to use. Defaults to 'xpath'.
- executor (str): The script executor to use. Defaults to 'JS'.
- time_out (int): The maximum time to wait for the element to be present. Defaults to 5.
- multi_idx (int): The index of the element to interact with in case multiple elements are found. Defaults to 0.
- sleep_before (Union[None, int, float, Tuple[int, int]]): The sleep interval before executing the wrapped function. Defaults to None.
- sleep_after (Union[None, int, float, Tuple[int, int]]): The sleep interval after executing the wrapped function. Defaults to (3, 1).
- **kwargs: Keyword arguments passed to the wrapped function.

#### Returns
The return value of the wrapped function.

#### Raises
- TypeError: If `element` is not of type `None`, `str`, or `ElementType`.
- TimeoutException: If the element is not found within the specified time out.

### Browser
**General Description**

Browser class for web automation.

#### Attributes
- browser_name (str): The name of the browser.
- options (List[str]): A list of additional options to be passed to the browser.
- use_undetected (bool): Whether to use undetected_chromedriver or not.
- browser (Browser): The Selenium browser instance.

#### Methods
- get(self, url: str, sleep_before: Union[None, int, float, Tuple[int, int]] = None, sleep_after: Union[None, int, float, Tuple[int, int]] = (10, 5))
- execute_script(self, script: str, *args)
- click(self, element: Union[str, ElementType], by: str = 'xpath', executor: str = 'element', 
        time_out: int = 5, multi_idx: int = 0, sleep_before: Union[None, int, float, Tuple[int, int]] = None, 
        sleep_after: Union[None, int, float, Tuple[int, int]] = (3, 1))
- send_key(self, key, element: Union[str, ElementType], by: str = 'xpath', executor: str = 'element', 
        time_out: int = 5, multi_idx: int = 0, sleep_before: Union[None, int, float, Tuple[int, int]] = None, 
        sleep_after: Union[None, int, float, Tuple[int, int]] = (3, 1))
- scroll_percent(self, dx: Union[str, float], dy: Union[str, float], duration: int,
