import _thread
import http.cookiejar
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from queue import Queue

from bs4 import BeautifulSoup
from lxml import etree
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

if __name__ == '__main__':
    from mbapy.base import put_err, check_parameters_path, check_parameters_len, get_default_for_bool
    from mbapy.file import save_json, read_json, save_excel, read_excel, opts_file
else:
    from .base import put_err, check_parameters_path, check_parameters_len, get_default_for_bool
    from .file import save_json, read_json, save_excel, read_excel, opts_file

CHROMEDRIVERPATH = r"C:\Users\Administrator\AppData\Local\Google\Chrome\Application\chromedriver.exe"
CHROME_DRIVER_PATH = CHROMEDRIVERPATH
BROWSER_HEAD = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'

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
def get_url_page(url:str, return_html_text:bool = False, debug:bool = False, coding = 'gbk'):
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
def get_url_page(browser, url:str, return_html_text:bool = False, debug = False):
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

def get_between(string:str, head:str, tail:str,
               headRFind:bool = False, tailRFind:bool = True,
               ret_head:bool = False, ret_tail:bool = False,
               find_tail_from_head = False):
    """
    Returns a substring of `string` that is between the last occurrence of `head` and the first 
    occurrence of `tail`. If `headRFind` is True, the last occurrence of `head` is used to find 
    the index of the start of the substring; otherwise, the first occurrence of `head` is used. 
    If `tailRFind` is True, the last occurrence of `tail` is used to find the index of the 
    end of the substring; otherwise, the first occurrence of `tail` is used. If `retHead` is 
    True, the returned substring includes the `head`; otherwise it starts after the `head`. 
    If `retTail` is True, the returned substring includes the `tail`; otherwise it ends before 
    the `tail`. If either `head` or `tail` is not found in `string`, an error message is returned. 

    :param string: The string to extract a substring from
    :type string: str
    :param head: The starting delimiter of the substring
    :type head: str
    :param tail: The ending delimiter of the substring
    :type tail: str
    :param headRFind: Whether to find the last (True) or first (False) occurrence of `head`
    :type headRFind: bool
    :param tailRFind: Whether to find the last (True) or first (False) occurrence of `tail`
    :type tailRFind: bool
    :param retHead: Whether to include the `head` in the returned substring
    :type retHead: bool
    :param retTail: Whether to include the `tail` in the returned substring
    :type retTail: bool
    :return: The substring between `head` and `tail` in `string`
    :rtype: str
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
    Returns the substring between the first occurrence of `head` and the first occurrence of `tail` in `string`.
    If `head_r` is True, returns the substring between the last occurrence of `head` and the first occurrence of `tail`.
    If `tail_r` is True, returns the substring between the first occurrence of `head` and the last occurrence of `tail`.
    If `ret_head` is True, includes the `head` substring in the returned substring.
    If `ret_tail` is True, includes the `tail` substring in the returned substring.
    
    :param string: The string to search for the substrings.
    :type string: str
    :param head: The left edge of the substring.
    :type head: str
    :param tail: The right edge of the substring.
    :type tail: str
    :param head_r: If True, returns the substring between the last occurrence of `head` and the first occurrence of `tail`.
    :type head_r: bool
    :param tail_r: If True, returns the substring between the first occurrence of `head` and the last occurrence of `tail`.
    :type tail_r: bool
    :param ret_head: If True, includes the `head` substring in the returned substring.
    :type ret_head: bool
    :param ret_tail: If True, includes the `tail` substring in the returned substring.
    :type ret_tail: bool
    :return: The substring between `head` and `tail` in `string`.
    :rtype: str
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

def get_browser(browser:str, browser_driver_path:str = None,
                options =['--no-sandbox', '--headless', f"--user-agent={BROWSER_HEAD:s}"],
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
        from selenium.webdriver.edge.options import Options
        from selenium.webdriver import Edge as Browser
    elif browser == 'Chrome':
        if use_undetected:
            from undetected_chromedriver import ChromeOptions as Options
            from undetected_chromedriver import Chrome as Browser
        else:
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver import Chrome as Browser
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
        # Add cookies from a file
        add_cookies(browser, cookies_path="cookies.txt")

        # Add cookies from a string
        add_cookies(browser, cookies_string="cookie1=value1; cookie2=value2")
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
    Sends a string of keys to an element on a webpage using Selenium WebDriver.
    
    Args:
    - browser: The WebDriver object to use.
    - keys: The string of keys to send to the element.
    - element: The identifier of the element to which the keys should be sent.
    - by: The method to use to find the element. Defaults to 'class'. Supported values are 'class', 'css', 'xpath'
    - wait: The maximum number of seconds to wait for the element to appear. Defaults to 5.
    
    Returns:
    - None
    """
    by = transfer_str2by(by)
    try:
        WebDriverWait(browser, wait).until(EC.presence_of_element_located((by, element)))
    finally:
        element = browser.find_element(by, element)
        element.send_keys(keys)
        
def click_browser(browser, element:str, by:str = 'class', wait = 5):
    """
    Clicks on a specified element in a browser using Selenium WebDriver.

    :param browser: The Selenium WebDriver to use.
    :type browser: selenium.webdriver
    :param element: The identifier for the element to be clicked.
    :type element: str
    :param by: The method used to locate the element (default is 'class').
        Valid options are 'class', 'id', 'name', 'xpath', 'css_selector', and 'link_text'.
    :type by: str
    :param wait: The number of seconds to wait for the element to appear before raising a TimeoutException.
    :type wait: int or float
    :return: None
    :rtype: None
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


def _wait_for_quit(statuesQue,):
    flag = 1
    while flag:
        s = input()
        if s == "e":
            statues_que_opts(statuesQue, "quit", "setValue", True)
            flag = 0
        else:
            statues_que_opts(statuesQue, "input", "setValue", s)
    return 0

def statues_que_opts(theQue, var_name, opts, *args):
    """opts contain:
    getValue: get varName value
    setValue: set varName value
    putValue: put varName value to theQue
    reduceBy: varName -= args[0]
    addBy: varName += args[0]
    """
    dataDict, ret = theQue.get(), None
    if var_name in dataDict.keys():
        if opts in ["getValue", "getVar"]:
            ret = dataDict[var_name]
        elif opts in ["setValue", "setVar"]:
            dataDict[var_name] = args[0]
        elif opts == "reduceBy":
            dataDict[var_name] -= args[0]
        elif opts == "addBy":
            dataDict[var_name] += args[0]
        else:
            print("do not support {" "} opts".format(opts))
    elif opts == "putValue":
        dataDict[var_name] = args[0]
    else:
        print("do not have {" "} var".format(var_name))
    theQue.put(dataDict)
    return ret

def get_input(promot:str = '', end = '\n'):
    if len(promot) > 0:
        print(promot, end = end)
    ret = statues_que_opts(statuesQue, "input", "getValue")
    while ret is None:
        time.sleep(0.1)
        ret = statues_que_opts(statuesQue, "input", "getValue")
    statues_que_opts(statuesQue, "input", "setValue", None)
    return ret

def show_prog_info(idx:int, sum:int = -1, freq:int = 10, otherInfo:str = ''):
    """
    Print progress information about a task at a certain frequency.

    :param idx: An integer representing the current progress of the task.
    :param sum: An optional integer representing the total size of the task. Default is -1.
    :param freq: An integer representing the frequency to print the progress information. Default is 10.
    :param otherInfo: An optional string with additional information to display. Default is an empty string.
    :return: None.
    """
    if idx % freq == 0:
        print(f'\r {idx:d} / {sum:d} | {otherInfo:s}', end = '')

class Timer:
    def __init__(self, ):
        self.lastTime = time.time()

    def OnlyUsed(self, ):
        return time.time() - self.lastTime

    def __call__(self) -> float:
        uesd = time.time() - self.lastTime
        self.lastTime = time.time()
        return uesd

class ThreadsPool:
    """self_func first para is a que for getting data,
    second is a que for send done data to main thread,
    third is que to send quited sig when get wait2quitSignal,
    fourth is other data \n
    _thread.start_new_thread(func, (self.ques[idx], self.sig, ))
    """
    def __init__(self, sum_threads:int, self_func, other_data, name = 'ThreadsPool') -> None:
        self.sumThreads = sum_threads
        self.sumTasks = 0
        self.name = name
        self.timer = Timer()
        self.sig = Queue()
        self.putDataQues = [ Queue() for _ in range(sum_threads) ]
        self.getDataQues = [ Queue() for _ in range(sum_threads) ]
        self.quePtr = 0
        for idx in range(sum_threads):
            _thread.start_new_thread(self_func,
                                     (self.putDataQues[idx],
                                      self.getDataQues[idx],
                                      self.sig,
                                      other_data, ))

    def put_task(self, data) -> None:
        self.putDataQues[self.quePtr].put(data)
        self.quePtr = ((self.quePtr + 1) if ((self.quePtr + 1) < self.sumThreads) else 0)
        self.sumTasks += 1
        
    def loop2quit(self, wait2quitSignal) -> list:
        """ be sure that all tasks sended, this func will
        send 'wait to quit' signal to every que,
        and start to loop waiting"""
        retList = []
        for idx in range(self.sumThreads):
            self.putDataQues[idx].put(wait2quitSignal)
        while self.sig._qsize() < self.sumThreads:
            sumTasksTillNow = sum([self.putDataQues[idx]._qsize() for idx in range(self.sumThreads)])
            print(f'\r{self.name:s}: {sumTasksTillNow:d} / {self.sumTasks:d} -- {self.timer.OnlyUsed():8.1f}s')
            for que in self.getDataQues:
                while not que.empty():
                    retList.append(que.get())
            time.sleep(1)
            if statues_que_opts(statuesQue, "quit", "getValue"):
                print('get quit sig')
                return retList            
        for que in self.getDataQues:
            while not que.empty():
                retList.append(que.get())
        return retList

def launch_sub_thread():
    """
    Launches a sub-thread to run a separate task concurrently with the main thread.

    This function creates a global `statuesQue` queue and puts a dictionary with the keys `quit` and `input` into the queue. The `quit` key is set to `False` and the `input` key is set to `None`. 
    The function then starts a new thread by calling the `_wait_for_quit` function with the `statuesQue` queue as an argument. 
    Finally, the function prints the message "web sub thread started".
    """
    global statuesQue
    statuesQue = Queue()
    statuesQue.put(
        {
            "quit": False,
            "input": None,
        }
    )
    _thread.start_new_thread(_wait_for_quit, (statuesQue,))
    print('mbapy::web: web sub thread started')

    
if __name__ == '__main__':
    # dev code
    pass
    