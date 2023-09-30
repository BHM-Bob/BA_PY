'''
Author: BHM-Bob G 2262029386@qq.com
Date: 2022-05-15 20:10:04
LastEditors: BHM-Bob
LastEditTime: 2022-10-19 22:53:28
Description: 
'''
import sys
import time

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

sys.path.append(r'../../../../mbapy/')
from web import *

browser = webdriver.Chrome(CHROMEDRIVERPATH)
myID = read_json('_ID.json')['chaoxing']
browser.get('http://i.chaoxing.com/base?t=1606058368568')
send_browser_key(browser, 'input[type="text"]', myID['id'], by = 'css')
send_browser_key(browser, 'input[type="password"]', myID['pw'] + Keys.RETURN, by = 'css')
time.sleep(1)
browser.get('https://mooc1-2.chaoxing.com/visit/stucoursemiddle?courseid=222808673&clazzid=50993177&vc=1&cpi=148111833&ismooc2=1')
click_browser(browser, 'a[title="章节"]', by = 'css')
browser.get('https://mooc1.chaoxing.com/mycourse/studentstudy?chapterId=519527060&courseId=222808673&clazzid=50993177&enc=3aeaa3aaab1564725074e1a64b07e482&mooc2=1&cpi=148111833&openc=bffd2d293ce1ec7b0ce8bf622a78eff9')
browser.get('https://mooc1.chaoxing.com/ztnodedetailcontroller/visitnodedetail?courseId=214230846&knowledgeId=342794026&_from_=222808673_50993177_147627979_3aeaa3aaab1564725074e1a64b07e482&rtag=&nohead=1')
direction = 1
while statues_que_opts(statuesQue, "quit", "getValue") == False:
    for i in range(99):
        browser.execute_script(f"window.scrollBy(0, 10)")
        time.sleep(1)
    try:
        if direction == 1:
            click_browser(browser, 'a[class="ml40 nodeItem r"]', by = 'css')
        else:
            click_browser(browser, 'a[class="nodeItem l"]', by = 'css')
    except:
        direction *= (-1)
    
browser.quit()

