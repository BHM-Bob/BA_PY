import time
import math

from mbapy import web
from mbapy.file import read_json
from mbapy.game import BaseInfo

import pyautogui
import numpy as np


link2handle = {}
search_link = 'https://pss-system.cponline.cnipa.gov.cn/seniorSearch'
result_link = 'https://pss-system.cponline.cnipa.gov.cn/retrieveList?prevPageTit=gaoji'
doc_link = 'https://pss-system.cponline.cnipa.gov.cn/documents/detail?prevPageTit=gaoji'
user_data = read_json('./data_tmp/id.json')['cnipa']
records_path = './data_tmp/patents.json'

class Records(BaseInfo):
    def __init__(self) -> None:
        super().__init__()
        self.patents = {} # request_id - content

def update_link2handle(b):
    global link2handle
    if len(link2handle) == 3 and search_link in link2handle and result_link in link2handle and doc_link in link2handle:
        return link2handle
    for handle in b.window_handles:
        b.switch_to.window(handle)
        link2handle[b.current_url] = handle
    return link2handle
        
def switch_window(name: str, b):
    update_link2handle(b)
    if name == 'search':
        b.switch_to.window(link2handle[search_link])
    elif name == 'result':
        b.switch_to.window(link2handle[result_link])
    elif name == 'doc':
        b.switch_to.window(link2handle[doc_link])

# 初始化服务
web.launch_sub_thread()
records = Records().from_json(records_path)
# 初始化浏览器并设置启用允许弹出式窗口(TODO: 目前只想到了pyautogui)
b = web.get_browser('Chrome', options = ['--no-sandbox'], use_undetected=True)
b.maximize_window()
b.get("chrome://settings/content/popups")
web.random_sleep(3)
choice_box = pyautogui.locateOnScreen('./data_tmp/imgs/chrome popups.png', confidence=0.9)
pyautogui.moveTo(pyautogui.center(choice_box))
pyautogui.click()

# 获取高级搜索页面
b.get('https://pss-system.cponline.cnipa.gov.cn/seniorSearch')
web.random_sleep(5, 3)

# 点击同意协议
web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[2]/div[8]/button', 'xpath')
web.random_sleep(5, 3)

# 点击登录按钮
web.click_browser(b, '//*[@id="app"]/div[1]/div[1]/div[3]/div[1]', 'xpath')
web.random_sleep(3)

# 点击二维码登录
# NOTE: 输入用户名密码似乎不稳定
web.click_browser(b, '//*[@id="userLayout"]/div/div[2]/div/button', 'xpath')
web.random_sleep(3)
captcha_ok = False
while not captcha_ok:
    if b.current_url == 'https://pss-system.cponline.cnipa.gov.cn/conventionalSearch':
        captcha_ok = True
    else:
        time.sleep(1)

# 点击检索方式下拉按钮
web.click_browser(b, '//*[@id="app"]/div[1]/div[1]/div[2]/div[1]/div/i', 'xpath')
web.random_sleep(3)

# 选择高级检索
web.click_browser(b, '//*[@id="app"]/div[1]/div[1]/div[2]/div[1]/ul/li[2]', 'xpath')
web.random_sleep(5, 3)

# 输入检索词
web.send_browser_key(b, user_data['query'], "//div[@class='el-row']/div[8]//input[@class='el-input__inner']", 'xpath')
web.random_sleep(3)

# 点击生成检索式按钮
web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[3]/div[2]/div/div[2]/div[2]/div/button[1]/span', 'xpath')
web.random_sleep(3)

# 点击检索按钮
web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[3]/div[2]/div/div[2]/div[2]/div/button[3]/i', 'xpath')
web.random_sleep(10, 5)

# 切换至结果窗口
switch_window('result', b)

# 获取总结果条目数并定义默认的每页结果数
sum_item = int(web.etree.HTML(b.page_source).xpath('//*[@id="app"]/div[1]/section/div/div/div[3]/div/div[1]/div[2]/div[1]/div[2]/span/text()')[0])
sum_item_per_page = 10

for page_idx in range(math.ceil(sum_item/sum_item_per_page)):
    # 切换至结果窗口
    switch_window('result', b)
    sum_items = len(web.etree.HTML(b.page_source).xpath("//li[@class='text2 color']/div/span"))
    for item_idx in range(sum_items):
        # 切换至结果窗口
        switch_window('result', b)
        web.random_sleep(10, 5)
        # 点击结果条目
        web.click_browser(b, f"(//li[@class='text2 color']/div/span)[{item_idx+1}]", 'xpath')
        web.random_sleep(10, 5)
        # 切换至文档窗口
        switch_window('doc', b)
        web.random_sleep(10, 5)
        # 获取专利信息
        tree = web.etree.HTML(b.page_source)
        patent_name = ''.join(tree.xpath('//*[@id="app"]/div[1]/section/div/div/div[2]/div[2]/div[2]/div[2]/div/div[3]/div[2]/div[6]/div/div[2]//text()'))
        if len(patent_name) == 0:
            # 如果点击失效，会直接切换页面，此时切回，再次点击 TODO: 递归 + try_time
            raise NotImplementedError
        request_id = tree.xpath('//*[@id="app"]/div[1]/section/div/div/div[2]/div[2]/div[2]/div[2]/div/div[1]/div[2]/div[1]/div/div[2]//text()')[0]
        request_date = tree.xpath('//*[@id="app"]/div[1]/section/div/div/div[2]/div[2]/div[2]/div[2]/div/div[1]/div[2]/div[2]/div/div[2]//text()')[0]
        public_id = tree.xpath('//*[@id="app"]/div[1]/section/div/div/div[2]/div[2]/div[2]/div[2]/div/div[3]/div[2]/div[1]/div/div[2]//text()')[0]
        public_date = tree.xpath('//*[@id="app"]/div[1]/section/div/div/div[2]/div[2]/div[2]/div[2]/div/div[3]/div[2]/div[2]/div/div[2]//text()')[0]
        abstruct = ' '.join(tree.xpath('//*[@id="cpp_content_i0"]//text()'))
        # 点击全文文本按钮
        web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[2]/div[2]/div[2]/div[1]/ul/li[2]', 'xpath')
        web.random_sleep(5, 3)
        # 获取全文文本
        tree = web.etree.HTML(b.page_source)
        full_text = '\n'.join(tree.xpath('//div[@class="fullText"]//text()')).replace(f'\n{user_data["query"][1:-1]}\n', f' {user_data["query"][1:-1]} ')
        # 点击法律状态按钮
        web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[2]/div[2]/div[2]/div[1]/ul/li[6]', 'xpath')
        web.random_sleep(3)
        tree = web.etree.HTML(b.page_source)
        table = tree.xpath('//*[@id="app"]/div[1]/section/div/div/div[2]/div[2]/div[2]/div[2]/div[1]/div[3]/table/tbody//text()')
        law_state = np.array(table).reshape(-1, 4).tolist()
        # 综合专利信息并保存至records
        records.patents[request_id] = {
            'request_id': request_id,
            'request_date': request_date,
            'public_id': public_id,
            'public_date': public_date,
            'abstruct': abstruct,
            'full_text': full_text,
            'law_state': law_state
        }
        
    # 切换至结果窗口
    switch_window('result', b)
    # 点击下一页
    web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[3]/div/div[1]/div[2]/div[2]/div[4]/div/div/span[3]/div/button[3]/i', 'xpath')
    
records.to_json(records_path)