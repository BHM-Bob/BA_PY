import _thread
import http.cookiejar
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from queue import Queue

import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from . import base
from mbapy.base import put_err
from mbapy.file import save_json, read_json, save_excel, read_excel

CHROMEDRIVERPATH = r"C:\Users\Administrator\AppData\Local\Google\Chrome\Application\chromedriver.exe"

def get_url_page(url:str, coding = 'gbk'):
    req = urllib.request.Request(url)
    # Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36
    # Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.36 Edg/86.0.622.63
    req.add_header("User-Agent",
                   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36")
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    urllib.request.install_opener(opener)
    return opener.open(req,timeout = 30).read().decode(coding,errors = 'ignore')
def get_url_page_s(url:str, coding = 'gbk'):
    try:
        return get_url_page(url, coding)
    except:
        return '-html-None'
def get_url_page(url:str, return_html_text:bool = False, debug:bool = False, coding = 'gbk'):
    if debug:
        html = get_url_page(url, coding)
    else:
        html = get_url_page_s(url, coding)
    if return_html_text:
        return BeautifulSoup(html, 'html.parser'), html
    return BeautifulSoup(html, 'html.parser')
def get_url_page(browser, url:str, return_html_text:bool = False, debug = False):
    browser.get(url)
    html = browser.page_source
    if return_html_text:
        return BeautifulSoup(html, 'html.parser'), html
    return BeautifulSoup(html, 'html.parser')

def get_between(string:str, head:str, tail:str,
               headRFind:bool = False, tailRFind:bool = True,
               retHead:bool = False, retTail:bool = False):
    headIdx = string.rfind(head) if headRFind else string.find(head)
    tailIdx = string.rfind(tail) if tailRFind else string.find(tail)
    if headIdx == -1 or tailIdx == -1:
        return put_err(f"{head if headIdx == -1 else tail:s} not found, return string", string)
    if headIdx == tailIdx:
        return put_err(f"headIdx == tailIdx with head:{head:s} and string:{string:s}, return string", string)
    idx1 = headIdx if retHead else headIdx+len(head)
    idx2 = tailIdx+len(tail) if retTail else tailIdx
    return string[idx1:idx2]
def get_between_re(string:str, head:str, tail:str,
               head_r:bool = False, tail_r:bool = True,
                ret_head:bool = False, ret_tail:bool = False):
    """support re
    """
    h = re.compile(head).search(string) if len(head) > 0 else ''
    t = re.compile(tail).search(string)
    if h is None or t is None:
        return put_err(f"not found with head:{head:s} and tail:{tail:s}, return string", string)
    else:
        h, t = h.group(0) if h != '' else '', t.group(0)
    return get_between(string, h, t, head_r, tail_r, ret_head, ret_tail)


def transfer_str2by(by:str):
    if by == 'class':
        return By.CLASS_NAME
    elif by == 'css':
        return By.CSS_SELECTOR
    else:
        raise Exception("unkown by : "+by)
def send_browser_key(browser, keys:str, element:str, by:str = 'class', wait:int = 5):
    by = transfer_str2by(by)
    try:
        elem = WebDriverWait(browser, wait).\
            until(EC.presence_of_element_located((by, element)))
    finally:
        elem = browser.find_element(by, 'which')  # Find the search box
        elem.send_keys(keys)        
def click_browser(browser, element:str, by:str = 'class', wait = 5):
    """by = calss | css
    """
    by = transfer_str2by(by)
    try:
        element = WebDriverWait(browser, wait).\
            until(EC.presence_of_element_located((by, element)))
    finally:
        rc = browser.find_element_by_class_name(element)


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

if base._Params['LAUNCH_WEB_SUB_THREAD']:
    statuesQue = Queue()
    statuesQue.put(
        {
            "quit": False,
            "input": None,
        }
    )
    _thread.start_new_thread(_wait_for_quit, (statuesQue,))