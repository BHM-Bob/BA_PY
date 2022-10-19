'''
Author: BHM-Bob G 2262029386@qq.com
Date: 2022-05-15 20:10:04
LastEditors: BHM-Bob G
LastEditTime: 2022-07-12 18:23:04
Description: 
'''
#from urllib.request import urlopen
#from bs4 import BeautifulSoup
import time, os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ba.web import *

browser = webdriver.Chrome(CHROMEDRIVERPATH)
myID = ReadObjFromJSON('_ID.json')['chaoxing']
browser.get('http://i.chaoxing.com/base?t=1606058368568')
SendKeysByCSS(browser, 'input[type="text"]', myID['id'])
SendKeysByCSS(browser, 'input[type="password"]', myID['pw'] + Keys.RETURN)
time.sleep(1)
browser.get('https://mooc1-2.chaoxing.com/visit/stucoursemiddle?courseid=222808673&clazzid=50993177&vc=1&cpi=148111833&ismooc2=1')
ClickByCSS(browser, 'a[title="章节"]')
browser.get('https://mooc1.chaoxing.com/mycourse/studentstudy?chapterId=519527060&courseId=222808673&clazzid=50993177&enc=3aeaa3aaab1564725074e1a64b07e482&mooc2=1&cpi=148111833&openc=bffd2d293ce1ec7b0ce8bf622a78eff9')
browser.get('https://mooc1.chaoxing.com/ztnodedetailcontroller/visitnodedetail?courseId=214230846&knowledgeId=342794026&_from_=222808673_50993177_147627979_3aeaa3aaab1564725074e1a64b07e482&rtag=&nohead=1')
direction = 1
while statuesQueOpts(statuesQue, "quit", "getValue") == False:
    for i in range(99):
        browser.execute_script(f"window.scrollBy(0, 10)")
        time.sleep(1)
    try:
        if direction == 1:
            ClickByCSS(browser, 'a[class="ml40 nodeItem r"]')
        else:
            ClickByCSS(browser, 'a[class="nodeItem l"]')
    except:
        direction *= (-1)
    
browser.quit()

