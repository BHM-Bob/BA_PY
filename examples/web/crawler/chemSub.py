'''
Author: BHM-Bob G 2262029386@qq.com
Date: 2022-05-26 23:51:16
LastEditors: BHM-Bob G
LastEditTime: 2022-05-29 17:31:54
Description: get media contents info
'''
import ssl
import sys

ssl._create_default_https_context = ssl._create_unverified_context
import os
import re
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

sys.path.append(r'../../../../mbapy/')
from web import *

from mbapy.web import *

desired_capabilities = DesiredCapabilities.CHROME # 修改页面加载策略
desired_capabilities["pageLoadStrategy"] = "none" 

dataRoot = r"D:\AI\DataSet\AlphaMedia\mediaDataBase_v2"
dbPath = os.path.join(dataRoot, "^record.json")
url = "http://chemsub.online.fr/"
url2 = "http://www.basechem.org"
db = read_json(dbPath) if os.path.isfile(dbPath) else []
browser = webdriver.Chrome(CHROMEDRIVERPATH)
contents = read_json(os.path.join(dataRoot, 'raw_cFreq0.json'))['content2pos']

for idx, content in enumerate(contents.keys()):
    show_prog_info(idx, len(contents))
    browser.get(url)
    print('got page')
    send_browser_key(browser, content, 'input[type=text][name=which][id=which]', 'css')
    click_browser(browser, 'input[id=inphow01][type=submit][name=how]', 'css')
    bs = BeautifulSoup(browser.page_source, 'html.parser')
    items = bs.findAll('td', {'class':"list", })
    if len(items) > 0:
        click_browser(browser, 'a[target=_top][class=list]', 'css')
        bs = BeautifulSoup(browser.page_source, 'html.parser')
        names, chNames = bs.findAll('tr', {'class':'vtop', }), []
        for idx, nDot in enumerate(names[10:]):
            if '中文' in nDot.text:
                for nDot2 in names[11+idx:]:
                    if not 'IUPAC' in nDot2.text:
                        chNames.append(nDot2.text)
                    else:
                        break
        name = '|'.join(chNames)
        pass
    else:
        pass
    # l[0].contents[0].attrs => {'href': '/chemical/41545', 'title': '琥珀酸-2,3-3H'}
    if items:
        print(url2+items[0].contents[0].attrs['href'])
        time.sleep(1)
        browser.get(url2+items[0].contents[0].attrs['href'])
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
        db.append([content, name, explaination, formula, mr])
    else:
        db.append([content, content, 'None', 'None', -1])
    
browser.quit()
save_json(dbPath, db, encoding = 'gbk')
print("All Done")
