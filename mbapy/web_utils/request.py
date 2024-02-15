import http.cookiejar
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import wraps
from typing import List, Tuple, Union

import numpy as np
import requests
import selenium
from bs4 import BeautifulSoup
from lxml import etree
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

if __name__ == '__main__':
    from mbapy.base import (Configs, check_parameters_len,
                            check_parameters_path, put_err)
    from mbapy.file import (opts_file, read_excel, read_json, save_excel,
                            save_json)
else:
    from ..base import (Configs, check_parameters_len, check_parameters_path,
                        put_err)
    from ..file import opts_file, read_excel, read_json, save_excel, save_json
    

def random_sleep(max_t: int = 10, min_t: int = 1):
    if isinstance(max_t, int) or isinstance(min_t, int):
        return time.sleep(random.uniform(float(min_t), float(max_t)))
    time.sleep(random.randint(min_t, max_t))

def get_requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    """
    Create and return a session object with automatic request retry functionality.

    Parameters:
        - retries (int): The number of times to retry a request in case of failure. Defaults to 3.
        - backoff_factor (float): The factor by which the backoff time between retries increases. Defaults to 0.3.
        - status_forcelist (tuple): The HTTP status codes that should trigger a retry. Defaults to (500, 502, 504).
        - session (Session): An existing session object to use. If not provided, a new session object will be created.

    Returns:
    - Session: The session object with retry functionality.
    """
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_url_page(url:str, coding = 'utf-8'):
    """
    Given a url and a coding, this function returns the decoded content of the page.
    :param url: A string representing the URL to be visited.
    :param coding: A string representing the character encoding of the page. Default is utf-8.
    :return: A string representing the decoded content of the page.
    """
    req = urllib.request.Request(url)
    # Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36
    # Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.36 Edg/86.0.622.63
    req.add_header("User-Agent",
                   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36")
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    urllib.request.install_opener(opener)
    return opener.open(req,timeout = 30).read().decode(coding,errors = 'ignore')

def get_url_page_s(url:str, coding = 'utf-8'):
    """
    Returns the HTML page content from the given URL. The function takes two parameters:
     - url: A string that represents the URL of the web page to retrieve.
     - coding: A string that represents the encoding of the HTML content. Default is 'utf-8'.
    The function tries to retrieve the HTML page content from the given URL using the get_url_page function,
    with the specified encoding. If it fails, it returns '-html-None'.
    """
    try:
        return get_url_page(url, coding)
    except:
        return '-html-None'
    
def get_url_page_b(url:str, return_html_text:bool = False, debug:bool = False, coding = 'utf-8'):
    """
    This function takes a URL and returns the HTML page of the URL in a BeautifulSoup object. It has the option to return a string of the HTML text as well. It also takes optional arguments for debugging and specifying the coding of the page to be retrieved. 

    Args:
        url (str): A string representing the URL to retrieve.
        return_html_text (bool): A boolean indicating whether or not to return the HTML text as a string. Defaults to False.
        debug (bool): A boolean indicating whether to use debug mode. Defaults to False.
        coding: The coding of the page to retrieve. Defaults to 'utf-8'.

    Returns:
        BeautifulSoup object: A BeautifulSoup object representing the HTML page of the URL.
        str (optional): If `return_html_text` is True, it returns a string of the HTML text.
    """
    if debug:
        html = get_url_page(url, coding)
    else:
        html = get_url_page_s(url, coding)
    if return_html_text:
        return BeautifulSoup(html, 'html.parser'), html
    return BeautifulSoup(html, 'html.parser')

def get_url_page_se(browser, url:str, return_html_text:bool = False, debug = False):
    """
    Retrieves the HTML source code of a webpage given its URL using a webdriver instance.

    Args:
        browser: The webdriver instance.
        url (str): The URL of the webpage to retrieve.
        return_html_text (bool): Whether or not to return the HTML source code as a string.
        debug (bool): Whether or not to enable debug mode.

    Returns:
        If `return_html_text` is True, returns a tuple containing a BeautifulSoup object 
        representing the parsed HTML and the raw HTML source code as a string. 
        Otherwise, returns a BeautifulSoup object representing the parsed HTML.
    """
    browser.get(url)
    html = browser.page_source
    if return_html_text:
        return BeautifulSoup(html, 'html.parser'), html
    return BeautifulSoup(html, 'html.parser')

def get_browser(browser:str, browser_driver_path:str = None,
                options =['--no-sandbox', '--headless', f"--user-agent={Configs.web.chrome_driver_path:s}"],
                use_undetected:bool = False):
    """
    Initializes and returns a Selenium browser instance based on the specified browser name and driver path.

    Parameters:
        browser (str): The name of the browser. Currently supported values are 'Edge' and 'Chrome'.
        browser_driver_path (str, optional): The path to the browser driver executable. Defaults to None.
        options (list, optional): A list of additional options to be passed to the browser. Defaults to ['--no-sandbox', '--headless'].
        
    Returns:
        Browser: An instance of the Selenium browser based on the specified browser name and options.
    """
    # get browser driver
    if browser == 'Edge':
        from selenium.webdriver import Edge as Browser
        from selenium.webdriver.edge.options import Options
        from selenium.webdriver.edge.service import Service
    elif browser == 'Chrome':
        from selenium.webdriver.chrome.service import Service
        if use_undetected:
            from undetected_chromedriver import Chrome as Browser
            from undetected_chromedriver import ChromeOptions as Options
        else:
            from selenium.webdriver import Chrome as Browser
            from selenium.webdriver.chrome.options import Options
    else:
        return put_err(f'Unsupported browser {browser}', None)
    # set options
    opts = Options()
    for option in options:
        opts.add_argument(option)
    # set Browser kwargs
    kwargs = {'options': opts}
    if browser_driver_path is not None:
        kwargs['service'] = Service(browser_driver_path)
    # return browser instance
    try:
        return Browser(**kwargs)
    except:
        if browser_driver_path is not None and\
            'AppData/Local/Google/Chrome/Application' in browser_driver_path:
            try:
                another_path = 'C:/Program Files/Google/Chrome/chromedriver.exe'
                kwargs['service'] = Service(another_path)
                put_err(f'can not use chromedriver with default setting, try {another_path}')
                return Browser(**kwargs)
            except:
                put_err(f'can not use chromedriver with default setting, return None')
                # just raise error
                return None
                

def add_cookies(browser, cookies_path:str = None, cookies_string:str = None):
    """
    Adds cookies to the browser.

    Parameters:
        browser (object): The browser object to add the cookies to.
        cookies_path (str, optional): The path to the file containing the cookies. Defaults to None.
        cookies_string (str, optional): A string of cookies to add. Defaults to None.

    Returns:
        None: This function does not return anything.

    Raises:
        ValueError: If no cookies are specified.

    Description:
        This function adds cookies to the specified browser object. It can add cookies from a file or from a string. If 
        the `cookies_path` parameter is provided, the function will check if the file exists and is valid before 
        parsing and adding the cookies. If the `cookies_string` parameter is provided, the function will directly parse 
        and add the cookies from the string. If neither `cookies_path` nor `cookies_string` is provided, a `ValueError` 
        will be raised.

        The function internally uses the `_parse_cookies` function to parse and add the cookies.

    Example Usage:
        >>> # Add cookies from a file
        >>> add_cookies(browser, cookies_path="cookies.txt")

        >>> # Add cookies from a string
        >>> add_cookies(browser, cookies_string="cookie1=value1; cookie2=value2")
    """
    def _parse_cookies(browser, cookies_string:str):
        for cookie in cookies_string.split(";"):
            name, value = cookie.strip().split("=", 1)
            browser.add_cookie({"name": name, "value": value})
    
    if cookies_path is not None and check_parameters_path(cookies_path):
        _parse_cookies(browser, opts_file(cookies_path))
    elif cookies_string is not None and check_parameters_len(cookies_string):
        _parse_cookies(browser, cookies_string)
    else:
        return put_err("No cookies specified", None)

def transfer_str2by(by:str):
    """
    Transfers a string representation of a 'By' identifier to the corresponding 'By' object.
    
    Parameters:
        by (str): The string representation of the 'By' identifier.
            support class('By.CLASS_NAME'), css('By.CSS_SELECTOR'), xpath('By.XPATH')
        
    Returns:
        By: The corresponding 'By' object.
        
    Raises:
        ValueError: If the 'by' parameter is not one of the valid 'By' identifier strings.
    """
    if by == 'class':
        return By.CLASS_NAME
    elif by == 'css':
        return By.CSS_SELECTOR
    elif by == 'xpath':
        return By.XPATH
    else:
        return put_err(f"Unknown By identifier {by:s}", None)
    
def wait_for_amount_elements(browser, by, element, count, timeout=10):
    """
    Waits for a specified number of elements to be present on the page.

    Args:
        browser (WebDriver): The WebDriver instance used to interact with the browser.
        by (str): The method used to locate the elements (e.g. "class", "css", "xpath").
        element (str): The value used to locate the elements (e.g. the ID, class name, or xpath expression).
        count (int): The number of elements to wait for.
        timeout (int, optional): The maximum amount of time (in seconds) to wait for the elements to be present. Defaults to 10.

    Returns:
        list: A list of WebElement objects representing the elements found.

    Raises:
        TimeoutException: If the elements are not found within the specified timeout.
    """
    wait = WebDriverWait(browser, timeout)
    by = transfer_str2by(by)
    try:
        elements = wait.until(lambda browser: len(browser.find_elements(by, element)) >= count)
    except:
        elements = browser.find_elements(by, element)
    return elements
    
def send_browser_key(browser, keys:str, element:str, by:str = 'class', wait:int = 5):
    """
    Sends a sequence of keystrokes to a specified element in a web browser.

    Args:
        browser (WebDriver): The web browser instance.
        keys (str): The sequence of keystrokes to send.
        element (str): The identifier of the element to send the keystrokes to.
        by (str, optional): The method used to locate the element. Defaults to 'class'.
        wait (int, optional): The maximum time in seconds to wait for the element to be present. Defaults to 5.

    Returns:
        None
    """
    by = transfer_str2by(by)
    try:
        WebDriverWait(browser, wait).until(EC.presence_of_element_located((by, element)))
    finally:
        element = browser.find_element(by, element)
        element.send_keys(keys)
        
def click_browser(browser, element:str, by:str = 'class', wait = 5):
    """
    Clicks on a specified element in a browser.

    Args:
        browser: The browser object on which to perform the click.
        element (str): The identifier of the element to click on.
        by (str, optional): The method to locate the element. Defaults to 'class'.
        wait (int, optional): The maximum time to wait for the element to be present. Defaults to 5.

    Returns:
        None
    """
    by = transfer_str2by(by)
    try:
        WebDriverWait(browser, wait).until(EC.presence_of_element_located((by, element)))
    finally:
        element = browser.find_element(by, element)
        # browser.execute_script("arguments[0].click();", element)
        ActionChains(browser).move_to_element(element).click().perform()

def scroll_browser(browser, scroll='bottom', duration=0):
    """
    Scroll the browser window either to the bottom or by a specific amount.

    Parameters:
        browser (object): The browser object.
        scroll (str|int): The scroll behavior. If set to 'bottom', the browser will be scrolled to the bottom. 
                          If set to an integer, the browser will be scrolled by that amount (in pixels).
        duration (int): The duration (in seconds) for which the scroll operation should be performed. 
                        If set to 0, the scroll operation will be performed instantly.
    Returns:
        None: This function does not return anything.
    Raises:
        ValueError: If the scroll type is unknown.
    """
    scrolled_length = 0
    end_time = time.time() + duration
    if isinstance(scroll, str) and scroll == 'bottom':
        if duration > 0:
            while time.time() < end_time:
                doc_height = browser.execute_script("return document.body.scrollHeight")
                last_heigth = doc_height - scrolled_length
                scroll_per_frame = last_heigth / (end_time - time.time()) / 10  # 假设每秒10帧
                scrolled_length += scroll_per_frame
                browser.execute_script(f"window.scrollBy(0, {scroll_per_frame});")
                time.sleep(1 / 10)  # 等待1/10秒，模拟每秒10帧
        # if duration == 0 or if duration is not enough
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    elif isinstance(scroll, int) and scroll > 0:
        if duration > 0:
            while time.time() < end_time:
                scroll_height = browser.execute_script("return document.body.scrollHeight")
                if scroll > scroll_height:
                    scroll = scroll_height
                scroll_per_frame = scroll / (end_time - time.time()) / 10  # 假设每秒10帧
                browser.execute_script(f"window.scrollBy(0, {scroll_per_frame});")
                time.sleep(1 / 10)  # 等待1/10秒，模拟每秒10帧
        else:
            if scroll > browser.execute_script("return window.innerHeight"):
                scroll = browser.execute_script("return window.innerHeight")
            browser.execute_script(f"window.scrollBy(0, {scroll});")
    else:
        return put_err(f"Unknown scroll type {scroll:s}", None)


def download_streamly(url: str, path: str, session):
    """
    Downloads a file from the given URL to the specified path using a streaming approach.

    Parameters:
        url (str): The URL of the file to be downloaded.
        path (str): The path where the downloaded file will be saved.
        session (object): The session object used for making the HTTP request.

    Returns:
        None
    """
    from tqdm import tqdm
    resp = session.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(path, 'wb') as file, tqdm(
        desc=path,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
            
            
ElementType = selenium.webdriver.remote.webelement.WebElement

def BrowserActionWarpper(func):
    """
    Decorator function and adds sleep functionality before and after the function call.

    Warpped functon args:
        self: The instance of the class that the function is a method of.
        *args: Positional arguments to be passed to the wrapped function.
        sleep_before (Union[None, int, float, Tuple[int, int]]): Optional. The amount of time to sleep before the function call.
            If an int or float, the function will sleep for a random amount of time between `sleep_before-1` and `sleep_before+1`.
            If a tuple of two ints, the function will sleep for a random amount of time between the two values in the tuple.
        sleep_after (Union[None, int, float, Tuple[int, int]]): Optional. The amount of time to sleep after the function call.
            If an int or float, the function will sleep for a random amount of time between `sleep_after-1` and `sleep_after+1`.
            If a tuple of two ints, the function will sleep for a random amount of time between the two values in the tuple.
        **kwargs: Keyword arguments to be passed to the wrapped function.

    Returns:
        The return value of the wrapped function.
    """
    @wraps(func)
    def core_wrapper(self, *args, sleep_before: Union[None, int, float, Tuple[int, int]] = None,
                     sleep_after: Union[None, int, float, Tuple[int, int]] = (3, 1), **kwargs):
        if sleep_before is not None:
            if isinstance(sleep_before, int) or isinstance(sleep_before, float):
                random_sleep(sleep_before+1, sleep_before-1)
            elif isinstance(sleep_before, tuple) and sleep_before[0] >= 0 and sleep_before[1] >= 0:
                random_sleep(*sleep_before)
        ret =  func(self, *args, **kwargs)
        if sleep_after is not None:
            if isinstance(sleep_after, int) or isinstance(sleep_after, float):
                random_sleep(sleep_after+1, sleep_after-1)
            elif isinstance(sleep_after, tuple) and sleep_after[0] >= 0 and sleep_after[1] >= 0:
                random_sleep(*sleep_after)
        return ret
    return core_wrapper

def BrowserElementActionWarpper(func):
    """
    Decorator function, find the element and adds sleep functionality before and after the function call.

    Warpped functon args:
        - self: The instance of the class.
        - *args: Positional arguments passed to the wrapped function.
        - element (Union[None, str, ElementType]): The element to locate on the page. Defaults to None.
        - by (str): The locator strategy to use. Defaults to 'xpath'.
        - executor (str): The script executor to use. Defaults to 'JS'.
        - time_out (int): The maximum time to wait for the element to be present. Defaults to 5.
        - multi_idx (int): The index of the element to interact with in case multiple elements are found. Defaults to 0.
        - sleep_before (Union[None, int, float, Tuple[int, int]]): The sleep interval before executing the wrapped function. 
            Defaults to None.
        - sleep_after (Union[None, int, float, Tuple[int, int]]): The sleep interval after executing the wrapped function. 
            Defaults to (3, 1).
        - **kwargs: Keyword arguments passed to the wrapped function.

    Returns:
        The return value of the wrapped function.

    Raises:
        - TypeError: If `element` is not of type `None`, `str`, or `ElementType`.
        - TimeoutException: If the element is not found within the specified time out.
    """
    @wraps(func)
    def core_wrapper(self, *args, element: Union[None, str, ElementType], by: str = 'xpath',
                     executor: str = 'JS', time_out: int = 5,
                     multi_idx: int = 0, sleep_before: Union[None, int, float, Tuple[int, int]] = None,
                     sleep_after: Union[None, int, float, Tuple[int, int]] = (3, 1), **kwargs):
        if element is None:
            element = 'document.body'
        elif element == 'window':
            pass
        else:
            by = self._get_by(by)
            try:
                WebDriverWait(self.browser, time_out).until(
                    EC.presence_of_element_located((by, element)))
            finally:
                elements = self.browser.find_elements(by, element)
                if len(elements) > 0:
                    element = elements[min(multi_idx, len(elements)-1)]
                else:
                    return put_err(f'{func.__name__} can not find element with expression: {element},\
                        do nothing and return None')
        if sleep_before is not None:
            if isinstance(sleep_before, int) or isinstance(sleep_before, float):
                random_sleep(sleep_before+1, sleep_before-1)
            elif isinstance(sleep_before, tuple) and sleep_before[0] >= 0 and sleep_before[1] >= 0:
                random_sleep(*sleep_before)
        ret =  func(self, *args, element = element, by = by, executor = executor,
                    time_out = time_out, multi_idx = multi_idx, **kwargs)
        if sleep_after is not None:
            if isinstance(sleep_after, int) or isinstance(sleep_after, float):
                random_sleep(sleep_after+1, sleep_after-1)
            elif isinstance(sleep_after, tuple) and sleep_after[0] >= 0 and sleep_after[1] >= 0:
                random_sleep(*sleep_after)
        return ret
    return core_wrapper

class Browser:
    """
    Browser class for web automation.
    
    Attributes:
        - browser_name (str): The name of the browser.
        - options (List[str]): A list of additional options to be passed to the browser.
        - use_undetected (bool): Whether to use undetected_chromedriver or not.
        - browser (Browser): The Selenium browser instance.
        
    Methods:
        - get(self, url: str, sleep_before: Union[None, int, float, Tuple[int, int]] = None, sleep_after: Union[None, int, float, Tuple[int, int]] = (10, 5))
        - execute_script(self, script: str, *args)
        - click(self, element: Union[str, ElementType], by: str = 'xpath', executor: str = 'element', 
                time_out: int = 5, multi_idx: int = 0, sleep_before: Union[None, int, float, Tuple[int, int]] = None, 
                sleep_after: Union[None, int, float, Tuple[int, int]] = (3, 1))
        - send_key(self, key, element: Union[str, ElementType], by: str = 'xpath', executor: str = 'element', 
                time_out: int = 5, multi_idx: int = 0, sleep_before: Union[None, int, float, Tuple[int, int]] = None, 
                sleep_after: Union[None, int, float, Tuple[int, int]] = (3, 1))
        - scroll_percent(self, dx: Union[str, float], dy: Union[str, float], duration: int,
    """
    def __init__(self, browser_name: str = 'Chrome',
                 options: List[str] = ['--no-sandbox', '--headless',
                                       f"--user-agent={Configs.web.chrome_driver_path:s}"],
                 use_undetected = False, driver_path = None) -> None:
        self.browser_name = browser_name
        self.options = options
        self.use_undetected = use_undetected
        
        self.browser = get_browser(browser_name, driver_path, options, use_undetected)
        
    def _get_by(self, by: str):
        if by in ['xpath', 'css', 'class']:
            return transfer_str2by(by)
        else:
            return by
        
    @BrowserActionWarpper
    def get(self, url: str,
            sleep_before: Union[None, int, float, Tuple[int, int]] = None,
            sleep_after: Union[None, int, float, Tuple[int, int]] = (10, 5)):
        return self.browser.get(url)
    
    def find_elements(self, element: str, by: str = 'xpath'):
        by = self._get_by(by)
        return self.browser.find_elements(by, element)
        
    def execute_script(self, script: str, *args):
        return self.browser.execute_script(script, *args)
        
    @BrowserElementActionWarpper
    def click(self, element: Union[str, ElementType], by: str = 'xpath',
              executor: str = 'element', time_out: int = 5, multi_idx: int = 0,
              sleep_before: Union[None, int, float, Tuple[int, int]] = None,
              sleep_after: Union[None, int, float, Tuple[int, int]] = (3, 1)):
        """
        Clicks on a specified element using different executors.

        Args:
            - element (Union[str, ElementType]): The element to be clicked. It can be either a string representing the element's xpath or the actual element object.
                - str: xpath, CSS, class expression to find the element.
                - ElementType: The actual element object.
            - by (str, optional): The locator strategy to find the element. Defaults to 'xpath'.
                - 'xpath', 'css', 'class': The locator strategy to find the element.
            - executor (str, optional): The executor to be used for the click operation. 
                - 'element', 'ActionChains', or 'JS'. Defaults to 'element'.
            - time_out (int, optional): The maximum time to wait for the element to be clickable. Defaults to 5.
            - multi_idx (int, optional): The index of the element to be clicked if there are multiple elements found with the same locator. Defaults to 0.
            - sleep_before (Union[None, int, float, Tuple[int, int]], optional): The time to sleep before clicking the element. It can be None, an integer or float representing the sleep time in seconds, or a tuple representing the sleep time range in seconds. Defaults to None.
            - sleep_after (Union[None, int, float, Tuple[int, int]], optional): The time to sleep after clicking the element. It can be None, an integer or float representing the sleep time in seconds, or a tuple representing the sleep time range in seconds. Defaults to (3, 1).
        """
        if executor == 'element':
            element.click()
        elif executor == 'ActionChains':
            ActionChains(self.browser).move_to_element(element).click().perform()
        elif executor == 'JS':
            self.execute_script("arguments[0].click();", element)
            
    @BrowserElementActionWarpper
    def send_key(self, key, element: Union[str, ElementType], by: str = 'xpath',
              executor: str = 'element', time_out: int = 5, multi_idx: int = 0,
              sleep_before: Union[None, int, float, Tuple[int, int]] = None,
              sleep_after: Union[None, int, float, Tuple[int, int]] = (3, 1)):
        """
        Performs a key press action on a specified element.

        Args:
            - key (str): The key to be pressed.
            - element (Union[str, ElementType]): The element to be clicked. It can be either a string representing the element's xpath or the actual element object.
                - str: xpath, CSS, class expression to find the element.
                - ElementType: The actual element object.
            - by (str, optional): The locator strategy to find the element. Defaults to 'xpath'.
                - 'xpath', 'css', 'class': The locator strategy to find the element.
            - executor (str, optional): The executor to be used for the click operation. 
                - 'element', 'ActionChains', or 'JS'. Defaults to 'element'.
            - executor (str, optional): The method of execution for the key press action. Valid options are 'element', 'ActionChains', and 'JS'. Defaults to 'element'.
            - time_out (int, optional): The maximum amount of time (in seconds) to wait for the element to be located. Defaults to 5.
            - multi_idx (int, optional): The index of the element in case multiple elements match the locator. Defaults to 0.
            - sleep_before (Union[None, int, float, Tuple[int, int]], optional): The amount of time to sleep before executing the action. This can be None, an integer or float representing the number of seconds, or a tuple representing a range of seconds. Defaults to None.
            - sleep_after (Union[None, int, float, Tuple[int, int]], optional): The amount of time to sleep after executing the action. This can be None, an integer or float representing the number of seconds, or a tuple representing a range of seconds. Defaults to (3, 1).
        """
        if executor == 'element':
            element.send_key(key)
        elif executor == 'ActionChains':
            ActionChains(self.browser).send_keys(key).click().perform()
        elif executor == 'JS':
            self.execute_script("arguments[0].value = arguments[1];", element, key)
            
    @BrowserElementActionWarpper
    def scroll_percent(self, dx: Union[str, float], dy: Union[str, float], duration: int,
                   element: Union[None, str, ElementType], by: str = 'xpath',
                   executor: str = 'JS', js_method: str = 'scrollBy',
                   verbose: bool = False, time_out: int = 5, multi_idx: int = 0,
                   sleep_before: Union[None, int, float, Tuple[int, int]] = None,
                   sleep_after: Union[None, int, float, Tuple[int, int]] = (3, 1)):
        """
        Scroll the element by a specific percentage of its scroll width and scroll height.

        Args:
            - dx (Union[str, float]): The percentage to scroll in the horizontal direction. If 'bottom' is passed, scroll to the bottom of the element.
            - dy (Union[str, float]): The percentage to scroll in the vertical direction. If 'bottom' is passed, scroll to the bottom of the element.
            - duration (int): The duration of the scrolling animation in seconds.
            - element (Union[str, ElementType]): The element to be clicked. It can be either a string representing the element's xpath or the actual element object.
                - None: wil be 'document.body', scroll the whole page.
                - 'window': scroll the whole page.
                - str: xpath, CSS, class expression to find the element.
                - ElementType: The actual element object.
            - by (str, optional): The locator strategy to find the element. Defaults to 'xpath'.
                - 'xpath', 'css', 'class': The locator strategy to find the element.
            - executor (str, optional): The executor to be used for the click operation. 
                - now only support 'JS'.
            - time_out (int): The maximum time to wait for the element to be located in seconds. Defaults to 5.
            - multi_idx (int): The zero-based index of the element to scroll if multiple elements are located. Defaults to 0.
            - sleep_before (Union[None, int, float, Tuple[int, int]]): The time to sleep before scrolling in seconds. Defaults to None.
            - sleep_after (Union[None, int, float, Tuple[int, int]]): The time to sleep after scrolling in seconds. Defaults to (3, 1).
        """
        if executor == 'JS':
            # get _get_scroll_, support bottom and int
            if dx == 'bottom':
                _get_scroll_width = lambda element : self.execute_script(
                    "return arguments[0].scrollWidth", element)
            else:
                _get_scroll_width = lambda element : dx * (self.execute_script(
                    "return arguments[0].scrollWidth", element) or 0.1)
            if dy == 'bottom':
                _get_scroll_height = lambda element : self.execute_script(
                    "return arguments[0].scrollHeight", element)
            else:
                _get_scroll_height = lambda element : dy * (self.execute_script(
                    "return arguments[0].scrollHeight", element) or 0.1)
            
            scrolled_len, last_len, scroll_len = np.zeros(2), np.zeros(2), np.zeros(2) # w, h
            end_time = time.time() + duration
            while time.time() < end_time:
                scroll_len[0] = _get_scroll_width(element) if dx != 0 else 0 # 适应动态加载
                scroll_len[1] = _get_scroll_height(element) if dy != 0 else 0 # 适应动态加载
                scrolled_len[0] = self.execute_script('return arguments[0].scrollLeft', element) if dx != 0 else 0
                scrolled_len[1] = self.execute_script('return arguments[0].scrollTop', element) if dx != 0 else 0
                if verbose:
                    print(f'scrolled_len: {scrolled_len}, last_len: {last_len}, scroll_len: {scroll_len}')
                last_len = scroll_len - scrolled_len
                last_frames = (end_time - time.time()) // 0.1 # 假设每秒10帧
                if last_frames > 0:
                    scroll_per_frame = last_len / last_frames
                    self.execute_script(f"arguments[0].{js_method}(arguments[1], arguments[2]);",
                                        element, int(scroll_per_frame[0]), int(scroll_per_frame[1]))
                    time.sleep(1 / 15) # 等待1/15秒
        else:
            return put_err(f'Not implemented with executor {executor},\
                do nothing and return None')


__all__ = [
    'random_sleep',
    'get_requests_retry_session',
    'get_url_page',
    'get_url_page_s',
    'get_url_page_b',
    'get_url_page_se',
    'get_browser',
    'add_cookies',
    'transfer_str2by',
    'wait_for_amount_elements',
    'send_browser_key',
    'click_browser',
    'scroll_browser',
    'download_streamly',
    'Browser',
]

if __name__ == '__main__':
    b = Browser(options=['--no-sandbox'], use_undetected= True, driver_path=Configs.web.chrome_driver_path)
    b.get('https://sci-hub.ren/', sleep_after=3)
    b.scroll_percent(0, 0.5, 5, element='//*[@id="info"]')
    b.send_key('mRNA Vaccine', element = '//*[@id="request"]')
    b.click(element = '//*[@id="enter"]/button')