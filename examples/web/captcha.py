'''
Date: 2023-10-17 19:15:15
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-10-17 21:00:19
Description: 
'''
import io

import easyocr
import numpy as np
from PIL import Image

import mbapy.web as web
from mbapy.web_utils.parse import etree

url = 'https://sso.lzu.edu.cn/login?service=https%3A%2F%2Fmy.lzu.edu.cn%2Fmylzu%2Fhome'

b = web.get_browser('Chrome', options=['--no-sandbox'], use_undetected=True)
b.get(url)
web.random_sleep(3)
tree = etree.HTML(b.page_source)
captcha_element = b.find_element(web.transfer_str2by('xpath'),
                                 '//img[@id="verification-code-img"]')
captcha_rect = captcha_element.rect
# captcha_image = Image.open(io.BytesIO(captcha_element.screenshot_as_png))

reader = easyocr.Reader(['en'], model_storage_directory='./data_tmp/nn_model/EasyOCR', download_enabled=False)
result = reader.readtext(captcha_element.screenshot_as_png)

print(result)