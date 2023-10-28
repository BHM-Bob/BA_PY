import time
import math
from typing import Dict

from mbapy import web
from mbapy.file import read_json, save_json

import easyocr
import pyautogui
import numpy as np


link2handle = {}
search_link = 'https://pss-system.cponline.cnipa.gov.cn/seniorSearch'
result_link = 'https://pss-system.cponline.cnipa.gov.cn/retrieveList?prevPageTit=gaoji'
doc_link = 'https://pss-system.cponline.cnipa.gov.cn/documents/detail?prevPageTit=gaoji'
user_data = read_json('./data_tmp/id.json')['cnipa']
records_path = './data_tmp/patents.json'
patents = read_json(records_path, 'gbk', {}) # request_id - content
reader = easyocr.Reader(['en'], model_storage_directory='./data_tmp/nn_model/EasyOCR', download_enabled=False)

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
        
def select_item(b, item_idx: int, patents: Dict):
    """切换窗口并点击item_idx的结果, 切换至doc窗口"""
    # 切换至结果窗口
    switch_window('result', b)
    # 点击结果条目(如果该条目未收集)
    request_id_ele = b.find_element(web.transfer_str2by('xpath'),
                                    f"(//li[@class='text3']/div/span)[{item_idx+1}]")
    public_id_ele = b.find_element(web.transfer_str2by('xpath'),
                                    f"(//li[@class='text2 color']/div/span)[{item_idx+1}]")
    if request_id_ele.text in patents:
        return None
    else:
        print(request_id_ele.text)
    web.random_sleep(10, 5)
    b.execute_script("arguments[0].scrollIntoView();", public_id_ele)
    b.execute_script("arguments[0].click();", public_id_ele)
    web.random_sleep(10, 5)
    # 切换至文档窗口
    switch_window('doc', b)
    # 返回非None值
    return True
    
def download_item(b, try_times: int = 10):
    if try_times == 0:
        return None
    # 获取并输入验证码
    captcha_element = b.find_element(web.transfer_str2by('xpath'), "//div[@class='authcode']/img[1]")
    if captcha_element.size['width'] == 0 or captcha_element.size['height'] == 0:
        return None
    result = reader.readtext(captcha_element.screenshot_as_png)
    if len(result) == 0 or len(result[0][1]) != 4:
        # 重载验证码并重试
        web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[7]/div/div/div[2]/div[3]/span[2]', 'xpath')
        web.random_sleep(3, 2)
        return download_item(b, try_times - 1)
    web.send_browser_key(b, web.Keys.CONTROL + 'a', '//*[@id="app"]/div[1]/section/div/div/div[7]/div/div/div[2]/div[3]/div/input', 'xpath')
    web.send_browser_key(b, result[0][1], '//*[@id="app"]/div[1]/section/div/div/div[7]/div/div/div[2]/div[3]/div/input', 'xpath')
    # 点击确定
    web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[7]/div/div/div[3]/span/button[3]/span', 'xpath')
    web.random_sleep(5, 3)
    # 查看是否验证失败
    download_settings_box = pyautogui.locateOnScreen('./data_tmp/imgs/cnipa download settings.png', confidence=0.99)
    if download_settings_box is not None:
        # 重载验证码并重试
        web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[7]/div/div/div[2]/div[3]/span[2]', 'xpath')
        web.random_sleep(3, 2)
        return download_item(b, try_times - 1)   
        
def get_patent_info(b):
    """已经切换至文档窗口后调用此函数, 返回专利名, 申请号, 公开号, 法律状态和全文文本信息"""
    # 切换至文档窗口
    switch_window('doc', b)
    web.random_sleep(3, 2)
    # 获取专利信息
    tree = web.etree.HTML(b.page_source)
    info_table = tree.xpath('//div[@class="doc-con"]/div/div[@class="basic-con"]')
    info_table = [tree.xpath(f'(//div[@class="doc-con"]/div/div[@class="basic-con"])[{i+1}]//text()') for i in range(len((info_table)))]
    info_dict = {single_info[0]: ''.join(single_info[1:]) for single_info in info_table}
    # 点击全文文本按钮
    web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[2]/div[2]/div[2]/div[1]/ul/li[2]', 'xpath')
    web.random_sleep(15, 10)
    # 获取全文文本
    tree = web.etree.HTML(b.page_source)
    full_text = '\n'.join(tree.xpath('//div[@class="fullText"]//text()')).replace(f'\n{user_data["query"][1:-1]}\n', f' {user_data["query"][1:-1]} ')
    # 获取法律状态
    # TODO: 适配多页, NOTE: UNTESTED 目前只是调整至单页40条
    # 点击法律状态按钮
    web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[2]/div[2]/div[2]/div[1]/ul/li[6]', 'xpath')
    web.random_sleep(3)
    # 点击页数下拉按钮
    web.scroll_browser(b, duration=3)
    web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[2]/div[2]/div[2]/div[2]/div[2]/div/div/span[2]/div/div[1]/span/span/i', 'xpath')
    # 点击40页选项
    web.random_sleep(2)
    web.click_browser(b, "//li[.='40 条/页']", 'xpath')
    # 下滑
    web.random_sleep(2)
    web.scroll_browser(b, duration=3)
    tree = web.etree.HTML(b.page_source)
    law_table = tree.xpath('//*[@id="app"]/div[1]/section/div/div/div[2]/div[2]/div[2]/div[2]/div[1]/div[3]/table/tbody//text()')
    law_state = np.array(law_table).reshape(-1, 4).tolist()
    # 点击同族按钮
    web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[2]/div[2]/div[2]/div[1]/ul/li[7]', 'xpath')
    # TODO: 适配多页, NOTE: UNTESTED 目前只是调整至单页40条
    # 点击页数下拉按钮
    web.scroll_browser(b, duration=3)
    web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[2]/div[2]/div[2]/div[2]/div[2]/div/div/span[2]/div/div[1]/span/span/i', 'xpath')
    # 点击40页选项
    web.random_sleep(2)
    web.click_browser(b, "//li[.='40 条/页']", 'xpath')
    # 下滑
    web.random_sleep(2)
    web.scroll_browser(b, duration=3)
    tree = web.etree.HTML(b.page_source)
    cluster_table = tree.xpath('//*[@id="app"]/div[1]/section/div/div/div[2]/div[2]/div[2]/div[2]/div[1]/div/div[3]/table/tbody//text()')
    # 点击下载按钮
    while pyautogui.locateOnScreen('./data_tmp/imgs/cnipa download settings.png', confidence=0.99) is None:
        web.click_browser(b, "//div[@class='doc-title-container']//span[1]/span[.='下载']", 'xpath')
        web.random_sleep(3)
    # 点击录著项目复选框按钮
    web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[7]/div/div/div[2]/label[1]/span[1]/span', 'xpath')
    # 点击全文文本复选框按钮
    web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[7]/div/div/div[2]/label[2]/span[1]/span', 'xpath')
    # 点击全文图像复选框按钮
    web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[7]/div/div/div[2]/label[3]/span[1]/span', 'xpath')
    # 处理验证码并下载文件
    web.random_sleep(3)
    download_item(b, try_times=10)
    # 返回综合后的信息
    return info_table[0][1], {
        'info_dict': info_dict,
        'full_text': full_text,
        'law_state': law_state,
        'cluster': cluster_table
    }
    
def get_single_patent(b, item_idx: int, patents: Dict, try_times: int = 5):
    if try_times == 0:
        return None
    # 选择结果条目
    if select_item(b, item_idx, patents) is None:
        return None
    # 获取专利信息
    patent_info = get_patent_info(b)
    if patent_info is None:        
        return get_single_patent(b, item_idx, patents, try_times-1)
    return patent_info


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
web.random_sleep(2)
web.scroll_browser(b, duration=3)

# 获取总结果条目数并定义默认的每页结果数
tree = web.etree.HTML(b.page_source)
sum_item = int(tree.xpath('//*[@id="app"]/div[1]/section/div/div/div[3]/div/div[1]/div[2]/div[1]/div[2]/span/text()')[0])
sum_item_per_page = 10
now_page_idx = int(tree.xpath('//li[@class="number active"]//text()')[0])
sum_page = math.ceil(sum_item/sum_item_per_page)

# NOTE: 点击下一页可能会失效，故页面数+1
while now_page_idx <= sum_page:
    # 切换至结果窗口并获取该页的条目数
    switch_window('result', b)
    sum_items = len(web.etree.HTML(b.page_source).xpath("//li[@class='text2 color']/div/span"))
    
    # 挨个获取专利信息
    for item_idx in range(sum_items):
        patent_info = get_single_patent(b, item_idx, patents, try_times=5)
        if patent_info is not None:
            patents[patent_info[0]] = patent_info[1]
            # 每次都保存
            save_json(records_path, patents, 'gbk')
            
    # 切换至结果窗口并点击下一页
    switch_window('result', b)
    web.random_sleep(2)
    web.scroll_browser(b, duration=3)
    tree = web.etree.HTML(b.page_source)
    now_page_idx = int(tree.xpath('//li[@class="number active"]//text()')[0])
    if now_page_idx == sum_page:
        break
    else:
        web.click_browser(b, '//*[@id="app"]/div[1]/section/div/div/div[3]/div/div[1]/div[2]/div[2]/div[4]/div/div/span[3]/div/button[3]/i', 'xpath')
        web.random_sleep(5, 3)