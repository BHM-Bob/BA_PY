
import http.cookiejar
import requests
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from bs4 import BeautifulSoup
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

if __name__ == '__main__':
    from mbapy.base import (Configs, check_parameters_len,
                            check_parameters_path, put_err)
    from mbapy.file import (opts_file, read_excel, read_json, save_excel,
                            save_json)

    # functon assembly
else:
    from ..base import (Configs, check_parameters_len, check_parameters_path,
                        put_err)
    from ..file import opts_file, read_excel, read_json, save_excel, save_json

    # functon assembly

def random_sleep(max_t: float = 10, min_t: float = 1):
    time.sleep(np.random.uniform(min_t, max_t))

def get_requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
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

def get_url_page(url:str, coding = 'gbk'):
    """
    Given a url and a coding, this function returns the decoded content of the page.
    :param url: A string representing the URL to be visited.
    :param coding: A string representing the character encoding of the page. Default is gbk.
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

def get_url_page_s(url:str, coding = 'gbk'):
    """
    Returns the HTML page content from the given URL. The function takes two parameters:
     - url: A string that represents the URL of the web page to retrieve.
     - coding: A string that represents the encoding of the HTML content. Default is 'gbk'.
    The function tries to retrieve the HTML page content from the given URL using the get_url_page function,
    with the specified encoding. If it fails, it returns '-html-None'.
    """
    try:
        return get_url_page(url, coding)
    except:
        return '-html-None'
    
def get_url_page_b(url:str, return_html_text:bool = False, debug:bool = False, coding = 'gbk'):
    """
    This function takes a URL and returns the HTML page of the URL in a BeautifulSoup object. It has the option to return a string of the HTML text as well. It also takes optional arguments for debugging and specifying the coding of the page to be retrieved. 

    Args:
        url (str): A string representing the URL to retrieve.
        return_html_text (bool): A boolean indicating whether or not to return the HTML text as a string. Defaults to False.
        debug (bool): A boolean indicating whether to use debug mode. Defaults to False.
        coding: The coding of the page to retrieve. Defaults to 'gbk'.

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
    elif browser == 'Chrome':
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
        kwargs['executable_path'] = browser_driver_path
    # return browser instance
    return Browser(**kwargs)

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
        browser.execute_script("arguments[0].click();", element)
        from selenium.webdriver.common.action_chains import ActionChains

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