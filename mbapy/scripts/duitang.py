'''
Date: 2024-02-14 21:40:00
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-02-15 22:18:38
Description: 
'''

import argparse
import os
from typing import Dict, List

from bs4 import BeautifulSoup
from tqdm import tqdm
from selenium.webdriver.common.keys import Keys

if __name__ == '__main__':
    from mbapy.base import put_err, Configs, get_fmt_time
    from mbapy.file import read_json, save_json
    from mbapy.scripts._script_utils_ import clean_path, show_args
    from mbapy.web import Browser, TaskPool, get_url_page_s, random_sleep
    from mbapy.web_utils.spider import retrieve_file_async
else:
    from ..base import put_err, Configs, get_fmt_time
    from ..file import read_json, save_json
    from ._script_utils_ import clean_path, show_args
    from ..web import Browser, TaskPool, get_url_page_s, random_sleep
    from ..web_utils.spider import retrieve_file_async
    

def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser(description = 'download images from duitang.com')
    args_paser.add_argument('-q', '--query', type=str,
                            help="search query string, warp with '' if contains space. Default is %(default)s.")
    args_paser.add_argument('-n', '--num-pages', type=int, default=5,
                            help="number of pages to download. Default is %(default)s.")
    args_paser.add_argument('-o', '--output', type=str, default='.',
                            help='output dir path, default is %(default)s.')
    args_paser.add_argument('-t', '--type', type = str, default='jpg', choices=['jpg', 'avif'],
                            help='file-type of image files. Default is %(default)s.')
    args_paser.add_argument('-g', '--gui', action='store_true', default=False,
                            help='FLAG: use Selenium GUI to download images. Default is %(default)s.')
    args_paser.add_argument('-u', '--undetected-chromedriver', action='store_true', default=False,
                            help='FLAG: use undetected chromedriver to setup selenium. Default is %(default)s.')
    args = args_paser.parse_args(sys_args)
    
    args.output = clean_path(args.output)
    show_args(args, ['query', 'num_pages', 'output', 'type', 'gui', 'undetected_chromedriver'])
    os.makedirs(args.output, exist_ok=True)
    
    # prepare avif support
    if args.type == 'avif':
        from pillow_heif import register_avif_opener
        register_avif_opener()
    
    # setup selenium
    # NOTE: use header to enable PC browser mode
    b_options = ['--no-sandbox', f"--user-agent={Configs.web.request_header}"]
    b = Browser(options=b_options + ([] if args.gui else ['--headless']),
                use_undetected=args.undetected_chromedriver)
    
    # setup coroutine pool
    pool = TaskPool().start()
    
    # get base page and search
    base_url = 'https://www.duitang.com/'
    b.get(base_url)
    b.send_key(key = args.query, element='//*[@id="kw"]')
    b.click(element='//*[@id="dt-search"]/form/button')
    b.click(element='/html/body/div[9]/div/div[1]/a', sleep_before=(10, 5))
    # b.click()
    
    # get image urls for each page
    urls = read_json(path=os.path.join(args.output, '__mbapy-cli_urls__.json'),
                     invalidPathReturn=[])
    for _ in range(args.num_pages):
        # scroll to bottom to load all images
        for _ in range(15):
            b.execute_script("window.scrollBy(0, 10000)")
            random_sleep(1, 2)
        # parse image urls
        bs = BeautifulSoup(b.browser.page_source, 'html.parser')
        links = bs.findAll('a', {'target':'_blank', 'class':'a'})
        links = [i.attrs['href'] for i in links]
        names = bs.findAll('div', {'class':'g'})
        names = [i.text.replace('/','').strip() for i in names]
        for link, name in zip(links, names):
            if base_url[:-1] + link not in urls:
                urls.append(base_url[:-1] + link)
                subpage = get_url_page_s(base_url[:-1] + link)
                bs = BeautifulSoup(subpage, 'html.parser')
                img_link = bs.findAll('a', {'target':'_blank', 'class':'img-out'})[0].attrs['href']
                file_path = os.path.join(args.output, f"{name} {get_fmt_time()}.jpg")
                pool.add_task(None, retrieve_file_async, img_link, file_path,
                            {'User-Agent': Configs.web.request_header})
        # go to next page
        b.click(element='//a[@class="woo-nxt"]')
    # save urls record
    save_json(path=os.path.join(args.output, '__mbapy-cli_urls__.json'),
              obj=urls)

# dev code
# main(['-q', r'xxx', '-g', '-o', r'E:\HC\SpiderSD\xxx-DuiTang'])

if __name__ == '__main__':
    main()