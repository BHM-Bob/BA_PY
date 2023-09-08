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


if __name__ == '__main__':
    # dev code
    launch_sub_thread()
    print(get_input())