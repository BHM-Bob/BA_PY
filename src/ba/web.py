import _thread
import http.cookiejar
import json
import os
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

CHROMEDRIVERPATH = r"C:\Users\BHMfly\AppData\Local\Google\Chrome\Application\chromedriver.exe"

def geturlpage(strurl:str, strCode = 'gbk'):
    req = urllib.request.Request(strurl)
    # Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36
    # Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.36 Edg/86.0.622.63
    req.add_header("User-Agent",
                   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36")
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    urllib.request.install_opener(opener)
    return opener.open(strurl,timeout = 30).read().decode(strCode,errors = 'ignore')
def geturlpage_s1(strurl:str, strCode = 'gbk'):
    try:
        return geturlpage(strurl, strCode)
    except:
        return '-html-None'
def GetUrlPageR(strurl:str, returnHtmlText:bool = False, debug:bool = False, strCode = 'gbk'):
    if debug:
        html = geturlpage(strurl, strCode)
    else:
        html = geturlpage_s1(strurl, strCode)
    if returnHtmlText:
        return BeautifulSoup(html, 'html.parser'), html
    return BeautifulSoup(html, 'html.parser')
def GetUrlPageFromBrowser(browser, strurl:str, returnHtmlText:bool = False, debug = False):
    browser.get(strurl)
    html = browser.page_source
    if returnHtmlText:
        return BeautifulSoup(html, 'html.parser'), html
    return BeautifulSoup(html, 'html.parser')



def SaveObjAsJSON(path:str, obj, encoding:str = 'utf-8', forceUpdate = True):
    if forceUpdate or not os.path.isfile(path):
        json_str = json.dumps(obj, indent=1)
        with open(path, 'w' ,encoding=encoding, errors='ignore') as json_file:
            json_file.write(json_str)
def ReadObjFromJSON(path:str, encoding:str = 'utf-8', invalidPathReturn = None):
    if os.path.isfile(path):
        with open(path, 'r' ,encoding=encoding, errors='ignore') as json_file:
            json_str = json_file.read()
        return json.loads(json_str)
    return invalidPathReturn
def Save2DAsEXCEL(path:str, obj:list[list[str]], columns:list[str], encoding:str = 'utf-8', forceUpdate = True):
    if forceUpdate or not os.path.isfile(path):
        df = pd.DataFrame(obj, columns=columns)
        df.to_excel(path, encoding = encoding)
def ReadDFFromEXCEL(path:str, ignoreHead:bool = True,
                  ignoreFirstCol:bool = True, invalidPathReturn = None):
    if os.path.isfile(path):
        df = pd.read_excel(path, )
        return df
    return invalidPathReturn



def GetBetweenAndHT(string:str, head:str, tail:str):
    """ret include head and tail"""
    headIdx = string.find(head)
    tailIdx = string[headIdx+len(head):].find(tail)
    return string[headIdx:headIdx+tailIdx]
def GetBetweenFromHead(string:str, head:str, tail:str):
    """ret not include head and tail"""
    headIdx = string.find(head)
    tailIdx = string.find(tail)
    return string[headIdx+len(head):tailIdx]
def GetBetween(string:str, head:str, tail:str,
               headRFind:bool = False, tailRFind:bool = True,
               retHead:bool = False, retTail:bool = False):
    headIdx = string.rfind(head) if headRFind else string.find(head)
    tailIdx = string.rfind(tail) if tailRFind else string.find(tail)
    idx1 = headIdx if retHead else headIdx+len(head)
    idx2 = tailIdx+len(tail) if retTail else tailIdx
    return string[idx1:idx2]



def SendKeysByClassName(browser, className, keys, wait = 5):
    try:
        elem = WebDriverWait(browser, wait).\
            until(EC.presence_of_element_located((By.CLASS_NAME, className)))
    finally:
        elem = browser.find_element(By.NAME, 'which')  # Find the search box
        elem.send_keys(keys)
def SendKeysByCSS(browser, css, keys, wait = 5):
    try:
        elem = WebDriverWait(browser, wait).\
            until(EC.presence_of_element_located((By.CSS_SELECTOR, css)))
    finally:
        elem = browser.find_element(By.CSS_SELECTOR, css)  # Find the search box
        elem.send_keys(keys)
def ClickByClassName(browser, className, wait = 5):
    try:
        element = WebDriverWait(browser, wait).\
            until(EC.presence_of_element_located((By.CLASS_NAME, className)))
    finally:
        rc = browser.find_element_by_class_name(className)
        ActionChains(browser).click(rc).perform()
def ClickByCSS(browser, css, wait = 5):
    try:
        element = WebDriverWait(browser, 5).\
            until(EC.presence_of_element_located((By.CSS_SELECTOR, css)))
    finally:
        rc = browser.find_element_by_css_selector(css)
        ActionChains(browser).click(rc).perform()



def waitforquit(statuesQue,):
    flag = 1
    while flag:
        s = input()
        if s == "e":
            statuesQueOpts(statuesQue, "quit", "setValue", True)
            flag = 0
        else:
            statuesQueOpts(statuesQue, "input", "setValue", s)
    return 0

def statuesQueOpts(theQue, varName, opts, *args):
    """opts contain:
    getValue: get varName value
    setValue: set varName value
    putValue: put varName value to theQue
    reduceBy: varName -= args[0]
    addBy: varName += args[0]
    """
    dataDict, ret = theQue.get(), None
    if varName in dataDict.keys():
        if opts in ["getValue", "getVar"]:
            ret = dataDict[varName]
        elif opts in ["setValue", "setVar"]:
            dataDict[varName] = args[0]
        elif opts == "reduceBy":
            dataDict[varName] -= args[0]
        elif opts == "addBy":
            dataDict[varName] += args[0]
        else:
            print("do not support {" "} opts".format(opts))
    elif opts == "putValue":
        dataDict[varName] = args[0]
    else:
        print("do not have {" "} var".format(varName))
    theQue.put(dataDict)
    return ret

def GetInput(promot:str = '', end = '\n'):
    if len(promot) > 0:
        print(promot, end = end)
    ret = statuesQueOpts(statuesQue, "input", "getValue")
    while ret is None:
        time.sleep(0.1)
        ret = statuesQueOpts(statuesQue, "input", "getValue")
    statuesQueOpts(statuesQue, "input", "setValue", None)
    return ret

def ShowProgInfo(idx:int, sum:int = -1, freq:int = 10, otherInfo:str = ''):
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
    def __init__(self, sumThreads:int, self_func, otherData, name = 'ThreadsPool') -> None:
        self.sumThreads = sumThreads
        self.sumTasks = 0
        self.name = name
        self.timer = Timer()
        self.sig = Queue()
        self.putDataQues = [ Queue() for _ in range(sumThreads) ]
        self.getDataQues = [ Queue() for _ in range(sumThreads) ]
        self.quePtr = 0
        for idx in range(sumThreads):
            _thread.start_new_thread(self_func,
                                     (self.putDataQues[idx],
                                      self.getDataQues[idx],
                                      self.sig,
                                      otherData, ))

    def PutTask(self, data) -> None:
        self.putDataQues[self.quePtr].put(data)
        self.quePtr = ((self.quePtr + 1) if ((self.quePtr + 1) < self.sumThreads) else 0)
        self.sumTasks += 1
        
    def LoopToQuit(self, wait2quitSignal) -> list:
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
            if statuesQueOpts(statuesQue, "quit", "getValue"):
                print('get quit sig')
                return retList            
        for que in self.getDataQues:
            while not que.empty():
                retList.append(que.get())
        return retList

statuesQue = Queue()
statuesQue.put(
    {
        "quit": False,
        "input": None,
    }
)
_thread.start_new_thread(waitforquit, (statuesQue,))