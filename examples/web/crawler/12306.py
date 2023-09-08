import time

from mbapy.web import *

user_data = {
    'name': 'xxxx',
    'pw': 'xxxx',
    'id_4': 'xxxx',
}

launch_sub_thread()
b = get_browser('Chrome', options = ['--no-sandbox'], use_undetected=True)
time.sleep(2)
# 打开主网址
b.get('https://www.12306.cn/index/index.html')
# 登录（模拟输入用户名和密码）
time.sleep(2)
click_browser(b, "//a[text()='登录']", 'xpath')
time.sleep(2)
send_browser_key(b, user_data['name'], "//input[@title='请输入用户名或邮箱或手机号']", 'xpath')
time.sleep(2)
send_browser_key(b, user_data['pw'], "//input[@title='请输入密码']", 'xpath')
time.sleep(2)
click_browser(b, "//a[text()='立即登录']", 'xpath')
time.sleep(2)
# 验证（模拟输入验证码）
click_browser(b, "//a[text()='获取验证码']", 'xpath')
time.sleep(2)
user_data['message_code'] = get_input()
send_browser_key(b, user_data['id_4'], "//input[@placeholder='请输入登录账号绑定的证件号后4位']", 'xpath')
time.sleep(2)
send_browser_key(b, user_data['message_code'], "//input[@placeholder='请输入验证码']", 'xpath')
time.sleep(2)
click_browser(b, "//a[@id='sureClick']", 'xpath')
time.sleep(2)
# 打开查询页面（预获取）
b.get('https://kyfw.12306.cn/otn/leftTicket/init?linktypeid=dc&fs=%E5%85%B0%E5%B7%9E,LZJ&ts=%E8%8A%9C%E6%B9%96,WHH&date=2023-08-05&flag=Y,N,Y')
# 等待预定按钮出现（超时999999秒）
wait_for_amount_elements(b, 'xpath', "//a[text()='预订']", 1, 999999)
# 点击预订按钮
click_browser(b, "//a[text()='预订']", 'xpath')
# 选择购票人
click_browser(b, "//label[@for='normalPassenger_0']", 'xpath')
# 不买学生票
click_browser(b, "//a[@id='dialog_xsertcj_cancel']", 'xpath')
# 提交订单
click_browser(b, "//a[@id='submitOrder_id']", 'xpath')
