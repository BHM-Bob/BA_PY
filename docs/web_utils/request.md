Documentation
=============

Module Overview
---------------

This module provides a comprehensive set of functions and classes for web scraping and automation using Python. It integrates various libraries such as `requests`, `selenium`, and `BeautifulSoup` to facilitate tasks like fetching web pages, handling browser interactions, and parsing HTML content. The module also includes utility functions for random sleep intervals, retrying requests, and downloading files.

Functions
=========
`random_sleep(max_t: int = 10, min_t: int = 1)`
-----------------------------------------------
### Function Overview

Introduces a random sleep interval between `min_t` and `max_t` seconds to simulate human-like behavior or avoid overwhelming web servers.
### Parameters

*   `max_t` (int): The maximum sleep time in seconds. Defaults to 10.
*   `min_t` (int): The minimum sleep time in seconds. Defaults to 1.
### Return Value

None

### Notes

This function is useful for adding delays between requests to avoid being blocked by web servers.
### Example
```python
random_sleep(5, 2)  # Sleeps for a random time between 2 and 5 seconds
```
`get_requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None)`
-----------------------------------------------------------------------------------------------------------
### Function Overview

Creates a `requests.Session` object with automatic retry functionality for handling transient errors.
### Parameters

*   `retries` (int): The number of retry attempts. Defaults to 3.
*   `backoff_factor` (float): The delay factor between retries. Defaults to 0.3.
*   `status_forcelist` (tuple): A tuple of HTTP status codes that should trigger a retry. Defaults to (500, 502, 504).
*   `session` (requests.Session): An existing session object. If not provided, a new session is created.
### Return Value

A `requests.Session` object with retry functionality.
### Notes

This function helps in making HTTP requests more robust by automatically retrying failed requests.
### Example
```python
session = get_requests_retry_session(retries=5)
response = session.get("https://example.com")
```
`get_url_page(url: str, coding='utf-8')`
----------------------------------------
### Function Overview

Fetches the content of a web page from the given URL and decodes it using the specified encoding.
### Parameters

*   `url` (str): The URL of the web page to fetch.
*   `coding` (str): The character encoding of the page. Defaults to 'utf-8'.
### Return Value

The decoded content of the web page as a string.
### Notes

This function uses `urllib` to make the HTTP request and handle cookies.
### Example
```python
html_content = get_url_page("https://example.com")
print(html_content)
```
`get_url_page_s(url: str, coding='utf-8')`
------------------------------------------
### Function Overview

A simplified version of `get_url_page` that handles exceptions and returns `-html-None` if an error occurs.
### Parameters

*   `url` (str): The URL of the web page to fetch.
*   `coding` (str): The character encoding of the page. Defaults to 'utf-8'.
### Return Value

The decoded content of the web page as a string, or `-html-None` if an error occurs.
### Notes

This function is useful for cases where robust error handling is required.
### Example
```python
html_content = get_url_page_s("https://example.com")
print(html_content)
```
`get_url_page_b(url: str, return_html_text: bool = False, debug: bool = False, coding='utf-8')`
-----------------------------------------------------------------------------------------------
### Function Overview

Fetches the content of a web page and parses it using `BeautifulSoup`.
### Parameters

*   `url` (str): The URL of the web page to fetch.
*   `return_html_text` (bool): Whether to return the raw HTML text. Defaults to False.
*   `debug` (bool): Whether to enable debug mode. Defaults to False.
*   `coding` (str): The character encoding of the page. Defaults to 'utf-8'.
### Return Value

A `BeautifulSoup` object representing the parsed HTML. If `return_html_text` is True, returns a tuple containing the `BeautifulSoup` object and the raw HTML text.
### Notes

This function is useful for parsing HTML content for web scraping.
### Example
```python
soup = get_url_page_b("https://example.com")
print(soup.title.text)
```
`get_url_page_se(browser, url: str, return_html_text: bool = False, debug=False)`
---------------------------------------------------------------------------------
### Function Overview

Fetches the content of a web page using a Selenium WebDriver instance.
### Parameters

*   `browser`: The WebDriver instance.
*   `url` (str): The URL of the web page to fetch.
*   `return_html_text` (bool): Whether to return the raw HTML text. Defaults to False.
*   `debug` (bool): Whether to enable debug mode. Defaults to False.
### Return Value

A `BeautifulSoup` object representing the parsed HTML. If `return_html_text` is True, returns a tuple containing the `BeautifulSoup` object and the raw HTML text.
### Notes

This function is useful for fetching dynamic content that requires browser interaction.
### Example
```python
browser = get_browser("Chrome")
soup = get_url_page_se(browser, "https://example.com")
print(soup.title.text)
```
`get_browser(browser: str, browser_driver_path: str = None, options=['--no-sandbox', '--headless', f"--user-agent={Configs.web.chrome_driver_path:s}"], use_undetected: bool = False, download_path: str = None)`
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Function Overview

Initializes and returns a Selenium WebDriver instance based on the specified browser and options.
### Parameters

*   `browser` (str): The name of the browser (e.g., 'Chrome', 'Edge').
*   `browser_driver_path` (str): The path to the browser driver executable. Defaults to None.
*   `options` (list): A list of additional options for the browser. Defaults to `['--no-sandbox', '--headless']`.
*   `use_undetected` (bool): Whether to use undetected\_chromedriver. Defaults to False.
*   `download_path` (str): The path for downloaded files. Defaults to None.
### Return Value

A Selenium WebDriver instance.
### Notes

This function is useful for setting up a browser instance for web automation.
### Example
```python
browser = get_browser("Chrome", options=['--no-sandbox'])
```
`add_cookies(browser, cookies_path: str = None, cookies_string: str = None)`
----------------------------------------------------------------------------
### Function Overview

Adds cookies to a Selenium WebDriver instance from a file or a string.
### Parameters

*   `browser`: The WebDriver instance.
*   `cookies_path` (str): The path to a file containing cookies. Defaults to None.
*   `cookies_string` (str): A string of cookies. Defaults to None.
### Return Value

None

### Notes

This function is useful for managing cookies in web automation.
### Example
```python
add_cookies(browser, cookies_string="cookie1=value1; cookie2=value2")
```
`transfer_str2by(by: str)`
--------------------------
### Function Overview

Converts a string representation of a Selenium `By` locator to the corresponding `By` object.
### Parameters

*   `by` (str): The string representation of the locator (e.g., 'class', 'css', 'xpath').
### Return Value

The corresponding `By` object.
### Notes

This function simplifies the use of locators in Selenium.
### Example
```python
by = transfer_str2by("class")
```
`wait_for_amount_elements(browser, by, element, count, timeout=10)`
-------------------------------------------------------------------
### Function Overview

Waits for a specified number of elements to be present on the page.
### Parameters

*   `browser`: The WebDriver instance.
*   `by`: The locator strategy (e.g., `By.CLASS_NAME`, `By.CSS_SELECTOR`, `By.XPATH`).
*   `element` (str): The locator value.
*   `count` (int): The number of elements to wait for.
*   `timeout` (int): The maximum time to wait in seconds. Defaults to 10.
### Return Value

A list of `WebElement` objects representing the found elements.
### Notes

This function is useful for ensuring that a certain number of elements are present before proceeding.
### Example
```python
elements = wait_for_amount_elements(browser, By.CSS_SELECTOR, ".example-class", 3)
```
`send_browser_key(browser, keys: str, element: str, by: str = 'class', wait: int = 5)`
--------------------------------------------------------------------------------------
### Function Overview

Sends a sequence of keystrokes to a specified element in a browser.
### Parameters

*   `browser`: The WebDriver instance.
*   `keys` (str): The sequence of keystrokes to send.
*   `element` (str): The locator value of the element.
*   `by` (str): The locator strategy. Defaults to 'class'.
*   `wait` (int): The maximum time to wait for the element in seconds. Defaults to 5.
### Return Value

None

### Notes

This function is useful for interacting with web elements.
### Example
```python
send_browser_key(browser, "Hello, World!", "input-field", by="css")
```
`click_browser(browser, element: str, by: str = 'class', wait: int = 5)`
------------------------------------------------------------------------
### Function Overview

Clicks on a specified element in a browser.
### Parameters

*   `browser`: The WebDriver instance.
*   `element` (str): The locator value of the element.
*   `by` (str): The locator strategy. Defaults to 'class'.
*   `wait` (int): The maximum time to wait for the element in seconds. Defaults to 5.
### Return Value

None

### Notes

This function is useful for interacting with web elements.
### Example
```python
click_browser(browser, "button-class", by="css")
```
`scroll_browser(browser, scroll='bottom', duration=0)`
------------------------------------------------------
### Function Overview

Scrolls the browser window to the specified position.
### Parameters

*   `browser`: The WebDriver instance.
*   `scroll` (str|int): The scroll behavior. Can be 'bottom' or an integer value. Defaults to 'bottom'.
*   `duration` (int): The duration of the scroll in seconds. Defaults to 0.
### Return Value

None

### Notes

This function is useful for scrolling through web pages.
### Example
```python
scroll_browser(browser, scroll=500, duration=2)
```
`download_streamly(url: str, path: str, session)`
-------------------------------------------------
### Function Overview

Downloads a file from the given URL to the specified path using a streaming approach.
### Parameters

*   `url` (str): The URL of the file to download.
*   `path` (str): The local path to save the file.
*   `session`: The HTTP session object.
### Return Value

None

### Notes

This function is useful for downloading large files without loading them entirely into memory.
### Example
```python
session = requests.Session()
download_streamly("https://example.com/file.zip", "local_file.zip", session)
```

Classes
=======
`Browser`
---------
### Class Overview

A class for web automation using Selenium WebDriver.
### Initialization
```python
Browser(browser_name: str = 'Chrome', options: List[str] = ['--no-sandbox', '--headless'], use_undetected: bool = False, driver_path: str = None, download_path: str = None)
```
### Initialization Parameters

*   `browser_name` (str): The name of the browser (e.g., 'Chrome', 'Edge'). Defaults to 'Chrome'.
*   `options` (List\[str\]): A list of additional options for the browser. Defaults to `['--no-sandbox', '--headless']`.
*   `use_undetected` (bool): Whether to use undetected\_chromedriver. Defaults to False.
*   `driver_path` (str): The path to the browser driver executable. Defaults to None.
*   `download_path` (str): The path for downloaded files. Defaults to None.
### Members

*   `browser_name` (str): The name of the browser.
*   `options` (List\[str\]): The list of browser options.
*   `use_undetected` (bool): Whether undetected\_chromedriver is used.
*   `download_path` (str): The path for downloaded files.
*   `browser`: The Selenium WebDriver instance.
### Methods

#### `get(url: str, sleep_before: Union[None, int, float, Tuple[int, int]] = None, sleep_after: Union[None, int, float, Tuple[int, int]] = (10, 5))`
##### Method Overview

Navigates to the specified URL with optional sleep intervals before and after the request.
##### Parameters

*   `url` (str): The URL to navigate to.
*   `sleep_before` (Union\[None, int, float, Tuple\[int, int\]\]): The sleep interval before navigating. Defaults to None.
*   `sleep_after` (Union\[None, int, float, Tuple\[int, int\]\]): The sleep interval after navigating. Defaults to (10, 5).
##### Return Value

The HTML content of the page as a string.
##### Notes

This method is useful for navigating to web pages with delays to simulate human behavior.
##### Example
```python
browser = Browser()
html_content = browser.get("https://example.com", sleep_after=(5, 3))
print(html_content)
```
#### `find_elements(element: str, by: str = 'xpath')`
##### Method Overview

Finds elements on the page using the specified locator strategy.
##### Parameters

*   `element` (str): The locator value.
*   `by` (str): The locator strategy. Defaults to 'xpath'.
##### Return Value

A list of `WebElement` objects representing the found elements.
##### Notes

This method is useful for locating elements on a web page.
##### Example
```python
elements = browser.find_elements(".example-class", by="css")
```
#### `wait_element(element: Union[str, List[str]], by: str = 'xpath', timeout: int = 600, check_fn: Callable = None)`
##### Method Overview

Waits for an element or a list of elements to be present on the page.
##### Parameters

*   `element` (Union\[str, List\[str\]\]): The locator value or a list of locator values.
*   `by` (str): The locator strategy. Defaults to 'xpath'.
*   `timeout` (int): The maximum time to wait in seconds. Defaults to 600.
*   `check_fn` (Callable): A custom function to check the presence of elements. Defaults to None.
##### Return Value

True if the elements are found within the timeout, False otherwise.
##### Notes

This method is useful for ensuring that elements are present before proceeding.
##### Example
```python
browser.wait_element([".element1", ".element2"], timeout=300)
```
#### `wait_text(element: Union[str, List[str]], text: str, by: str = 'xpath', timeout: int = 600, check_fn: Callable = None)`
##### Method Overview

Waits for an element to contain the specified text.
##### Parameters

*   `element` (Union\[str, List\[str\]\]): The locator value or a list of locator values.
*   `text` (str): The expected text.
*   `by` (str): The locator strategy. Defaults to 'xpath'.
*   `timeout` (int): The maximum time to wait in seconds. Defaults to 600.
*   `check_fn` (Callable): A custom function to check the text. Defaults to None.
##### Return Value

True if the text is found within the timeout, False otherwise.
##### Notes

This method is useful for waiting for specific text to appear on a web page.
##### Example
```python
browser.wait_text(".example-class", "Expected Text", timeout=300)
```
#### `execute_script(script: str, *args)`
##### Method Overview

Executes a JavaScript script in the context of the current page.
##### Parameters

*   `script` (str): The JavaScript script to execute.
*   `*args`: Arguments to pass to the script.
##### Return Value

The result of the script execution.
##### Notes

This method is useful for executing custom JavaScript on a web page.
##### Example
```python
result = browser.execute_script("return document.title;")
print(result)
```
#### `click(element: Union[str, ElementType], by: str = 'xpath', executor: str = 'element', time_out: int = 5, multi_idx: int = 0, sleep_before: Union[None, int, float, Tuple[int, int]] = None, sleep_after: Union[None, int, float, Tuple[int, int]] = (3, 1))`
##### Method Overview

Clicks on a specified element with optional sleep intervals before and after the click.
##### Parameters

*   `element` (Union\[str, ElementType\]): The locator value or the element object.
*   `by` (str): The locator strategy. Defaults to 'xpath'.
*   `executor` (str): The executor to use for the click action. Defaults to 'element'.
*   `time_out` (int): The maximum time to wait for the element in seconds. Defaults to 5.
*   `multi_idx` (int): The index of the element if multiple elements match the locator. Defaults to 0.
*   `sleep_before` (Union\[None, int, float, Tuple\[int, int\]\]): The sleep interval before clicking. Defaults to None.
*   `sleep_after` (Union\[None, int, float, Tuple\[int, int\]\]): The sleep interval after clicking. Defaults to (3, 1).
##### Return Value

None

##### Notes

This method is useful for interacting with web elements.
##### Example
```python
browser.click(".example-class", by="css", sleep_after=(2, 1))
```
#### `send_key(key, element: Union[str, ElementType], by: str = 'xpath', executor: str = 'element', time_out: int = 5, multi_idx: int = 0, sleep_before: Union[None, int, float, Tuple[int, int]] = None, sleep_after: Union[None, int, float, Tuple[int, int]] = (3, 1))`
##### Method Overview

Sends a sequence of keystrokes to a specified element with optional sleep intervals before and after the action.
##### Parameters

*   `key`: The sequence of keystrokes to send.
*   `element` (Union\[str, ElementType\]): The locator value or the element object.
*   `by` (str): The locator strategy. Defaults to 'xpath'.
*   `executor` (str): The executor to use for the key press action. Defaults to 'element'.
*   `time_out` (int): The maximum time to wait for the element in seconds. Defaults to 5.
*   `multi_idx` (int): The index of the element if multiple elements match the locator. Defaults to 0.
*   `sleep_before` (Union\[None, int, float, Tuple\[int, int\]\]): The sleep interval before sending keys. Defaults to None.
*   `sleep_after` (Union\[None, int, float, Tuple\[int, int\]\]): The sleep interval after sending keys. Defaults to (3, 1).
##### Return Value

None

##### Notes

This method is useful for interacting with web elements.
##### Example
```python
browser.send_key("Hello, World!", ".example-class", by="css", sleep_after=(2, 1))
```
#### `scroll_percent(dx: Union[str, float], dy: Union[str, float], duration: int, element: Union[None, str, ElementType], by: str = 'xpath', executor: str = 'JS', js_method: str = 'scrollBy', verbose: bool = False, time_out: int = 5, multi_idx: int = 0, sleep_before: Union[None, int, float, Tuple[int, int]] = None, sleep_after: Union[None, int, float, Tuple[int, int]] = (3, 1))`
##### Method Overview

Scrolls an element by a specified percentage of its scroll width and height.
##### Parameters

*   `dx` (Union\[str, float\]): The horizontal scroll percentage or 'bottom'.
*   `dy` (Union\[str, float\]): The vertical scroll percentage or 'bottom'.
*   `duration` (int): The duration of the scroll in seconds.
*   `element` (Union\[None, str, ElementType\]): The locator value or the element object.
*   `by` (str): The locator strategy. Defaults to 'xpath'.
*   `executor` (str): The executor to use for the scroll action. Defaults to 'JS'.
*   `js_method` (str): The JavaScript method to use for scrolling. Defaults to 'scrollBy'.
*   `verbose` (bool): Whether to print debug information. Defaults to False.
*   `time_out` (int): The maximum time to wait for the element in seconds. Defaults to 5.
*   `multi_idx` (int): The index of the element if multiple elements match the locator. Defaults to 0.
*   `sleep_before` (Union\[None, int, float, Tuple\[int, int\]\]): The sleep interval before scrolling. Defaults to None.
*   `sleep_after` (Union\[None, int, float, Tuple\[int, int\]\]): The sleep interval after scrolling. Defaults to (3, 1).
##### Return Value

None

##### Notes

This method is useful for scrolling through web pages or elements.
##### Example
```python
browser.scroll_percent(0, 0.5, 2, ".example-class", by="css", sleep_after=(2, 1))
```