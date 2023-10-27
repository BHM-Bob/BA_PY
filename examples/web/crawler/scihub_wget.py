'''
Date: 2023-10-17 21:07:06
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-10-17 21:27:32
Description: 
'''
import os
from argparse import ArgumentParser

import wget

from mbapy.base import put_log
from mbapy.sci_utils.paper_download import (_get_scihub_valid_download_link,
                                            _update_available_scihub_urls,
                                            session)
from mbapy.web_utils.parse import etree

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-i', '--doi', type=str, default='10.1002/btpr.3138', help='doi')
    args.add_argument('-d', '--dir', type=str, default='./', help='output dir')
    args.add_argument('-o', '--out', type=str, default='paper.pdf', help='file name')
    args = args.parse_args()

    available_scihub_urls = _update_available_scihub_urls()
    res = session.request(method='GET', url=available_scihub_urls[0]+'/'+args.doi)
    results = etree.HTML(res.text)
    download_link = results.xpath('//div[@id="buttons"]//@onclick')[0].split("'")[1]
    valid_download_link = _get_scihub_valid_download_link(download_link)
    put_log(f'get valid download link: {valid_download_link}')
    
    
    file_name = wget.filename_from_url(valid_download_link)
    if args.out != 'paper.pdf':
        target_name = os.path.join(args.dir, args.out)
    else:
        target_name = os.path.join(args.dir, file_name)
    put_log(f'file name: {target_name}')
    file_name = wget.download(valid_download_link, out=target_name)

