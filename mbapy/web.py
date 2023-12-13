'''
Date: 2023-07-31 21:34:48
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-08-02 18:37:20
FilePath: \BA_PY\mbapy\web.py
Description: 
'''

if __name__ == '__main__':
    # functon assembly
    from mbapy.web_utils.parse import *
    from mbapy.web_utils.request import *
    from mbapy.web_utils.task import *
else:
    # functon assembly
    from .web_utils.parse import *
    from .web_utils.request import *
    from .web_utils.task import *

CHROMEDRIVERPATH = r"C:\Users\Administrator\AppData\Local\Google\Chrome\Application\chromedriver.exe"
CHROME_DRIVER_PATH = CHROMEDRIVERPATH
BROWSER_HEAD = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'


__all__ = [
    'CHROMEDRIVERPATH',
    'CHROME_DRIVER_PATH',
    'BROWSER_HEAD',
    
    'get_between',
    'get_between_re',
    'parse_xpath_info',
    
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
    
    'Key2Action',
    'statuesQue',
    '_wait_for_quit',
    'statues_que_opts',
    'get_input',
    'launch_sub_thread',
    'show_prog_info',
    'Timer',
    'ThreadsPool',
]

if __name__ == '__main__':
    # dev code
    launch_sub_thread()
    print(get_input())