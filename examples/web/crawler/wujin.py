'''
Author: BHM-Bob G 2262029386@qq.com
Date: 2022-05-12 13:22:15
LastEditors: BHM-Bob
LastEditTime: 2022-12-10 11:23:58
Description: 
'''
import os
import re
import sys
#from urllib.request import urlopen
#from bs4 import BeautifulSoup
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

sys.path.append(r'../../../../mbapy/')
from web import *

dataRoot = "D:\AI\DataSet\AlphaMedia\mediaDataBase_v2"
dbPath = os.path.join(dataRoot, "contentsInfo.xlsx")
url = "http://www.basechem.org/search"
url2 = "http://www.basechem.org"
db = []
browser = webdriver.Chrome(CHROMEDRIVERPATH)

d = read_json(r"D:\AI\DataSet\AlphaMedia\mediaDataBase_v2\raw_cFreq0.json")
contents = d['content2pos']
f2n = d['contentFullName2Name']
d2 = [l[0] for l in db]

def GetListContent(data, key1, key2, way = 'H&T'):
    for key1Idx, d1 in enumerate(data):
        if key1 in d1.text:
            for key2Idx, d2 in enumerate(data[key1Idx+1:]):
                if key2 in d2.text:
                    idx1, idx2 = key1Idx, key2Idx+key1Idx+2
                    if 'H' not in way:
                        idx1 += 1
                    if 'T' not in way:
                        idx2 -= 1
                    return ' | '.join(
                        [ d.text for d in data[idx1:idx2] ]
                    )
    return ''

def SearchAKey(key:str, browser):
    browser.get(url)
    time.sleep(1)
    elem = browser.find_element(By.NAME, 'q')  # Find the search box
    elem.send_keys(key + Keys.RETURN)
    bs = BeautifulSoup(browser.page_source, 'html.parser')
    items = bs.findAll('h2', {'class':"margin-bottom-20", })
    # l[0].contents[0].attrs => {'href': '/chemical/41545', 'title': '琥珀酸-2,3-3H'}
    if items:
        browser.get(url2+items[0].contents[0].attrs['href'])
        time.sleep(1)
        bs = BeautifulSoup(browser.page_source, 'html.parser')
        name = bs.findAll('span', {'class':'pull-left', })[0].\
            contents[0].text
        formula = bs.findAll('div', {'class':'col-sm-8', })[0].\
            contents[1].contents[2].contents[3].contents[3].text
        mr = bs.findAll('div', {'class':'col-sm-8', })[0].\
            contents[1].contents[2].contents[5].contents[3].text
        explaination = bs.findAll('div', {'class':'col-sm-8', })[0].\
            contents[1].contents[2].contents[7].contents[3].text
        explaination = re.sub('\n *', '', explaination)
        try:
            mr = float(mr)
        except:
            mr = '-1'
        wx = bs.findAll('div', {'class':'c-content', })[0].contents
        character = GetListContent(wx, '1.','2.','H').replace('\n', '')
        density = GetListContent(wx, '2.','3.','H').replace('\n', '')
        solobility = GetListContent(wx, '18.','\n','H').replace('\n', '')
        usage = GetListContent(wx, '用途','安全信息','&').replace('\n', '')
        return [key, name, explaination, formula, mr, character,
                density, solobility, usage]
    return [key, '', '', '', '', '', '', '', '']
        
    
for idx, content in enumerate(contents.keys()):
    show_prog_info(idx, len(contents))
    if content not in d2:
        result = SearchAKey(content, browser)
        if result[-1] == -1:
            result = SearchAKey(f2n[content], browser)
        db.append(result)
    
browser.quit()
save_excel(dbPath, db, columns=['key', 'name', 'explaination', 'formula', 'mr',
                                   'character', 'density', 'solobility', 'usage'],
            encoding='gbk')
print("All Done")
