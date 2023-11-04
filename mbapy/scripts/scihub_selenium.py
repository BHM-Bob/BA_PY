'''
Date: 2023-11-04 18:43:38
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-11-04 22:38:51
Description: 
'''
import argparse
import os
import random
import time
import urllib
from typing import Any, Dict

import requests
import tqdm
import wget

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'True'
from mbapy.base import (check_parameters_path, get_storage_path, put_err,
                        put_log)
from mbapy.file import get_valid_file_path, opts_file
from mbapy.game import BaseInfo
from mbapy.paper import (_get_scihub_valid_download_link,
                         _update_available_scihub_urls, get_reference_by_doi,
                         parse_ris, session)
from mbapy.web import (Browser, launch_sub_thread, random_sleep,
                       statues_que_opts, statuesQue)


class Record(BaseInfo):
    def __init__(self) -> None:
        super().__init__()
        self.doi: Dict[str, Dict[str, str]] = {} # doi(str) - download_info(Dict[str, str])
        self.center_paper: Dict[str, Dict[str, Any]] = {} # doi(str) - center_paper_info(Dict[str, str|Dict[str, str]])
        
        
def download_by_scihub(b: Browser, dir:str, doi: str, avaliable_scihub_url: str):
    b.get(url = f"{avaliable_scihub_url}/{doi}")
    # check if there has captcha
    if b.find_elements('//*[@id="h-captcha"]'):
        if '--headless' in b.browser.options.arguments:
            # headless mode, no GUI to let user pass captcha manually, return None
            return None
        else:
            # let user pass captcha manually
            while b.find_elements('//*[@id="h-captcha"]'):
                put_log("NOTE: Please pass captcha manually")
                random_sleep(5, 3)
    # get download link and download
    try:
        doc_ele = b.find_elements('//*[@id="pdf"]')
        if doc_ele:
            download_link = doc_ele[0].get_attribute('src')
            file_name = get_valid_file_path(doi.replace('/', '_') + wget.detect_filename(download_link))
            file_path = os.path.join(dir, file_name)
            try:
                wget.download(download_link, os.path.join(dir, file_name))
            except urllib.error.HTTPError:
                res = session.get(url = download_link, stream=False, timeout=60)
                if res.text.startswith('%PDF'):
                    opts_file(file_path, 'wb', data = result['res'].content)
            result = {'file_name': file_name, 'file_path': os.path.join(dir, file_name)}
            return result
    except:
        return put_err(f'Failed to download paper from {avaliable_scihub_url}/{doi}', None)
        
def download_refs(b: Browser, dir:str, refs:Dict[str, Any], records:Record,
                  available_scihub_url: str):
    for ref_info in refs:
        if 'DOI' in ref_info and ref_info['DOI'] not in records.doi:
            random_sleep(15, 10)
            try:
                ref_result = download_by_scihub(b, dir, ref_info['DOI'], available_scihub_url)
            except requests.exceptions.TooManyRedirects as e:
                ref_result = None
            if ref_result is not None:
                ref_info.update(ref_result)
                # add doi
                records.doi[ref_info['doi']] = ref_result
        # quit
        if statues_que_opts(statuesQue, "quit", "getValue"):
            break
    return refs
        
def download_center_paper_session(b: Browser, dir:str, info: Dict[str, str], records:Record,
                                  available_scihub_url: str, is_download_ref: bool = False):
    # if there has no doi, quikly return
    if not info['doi']:
        return None
    # download center paper
    result = download_by_scihub(b, dir, info['doi'], available_scihub_url)
    if result is not None and 'doi' in result:
        time.sleep(5 + random.randint(5, 10))
        records.doi[result['doi']] = result
        if is_download_ref:
            # get refs
            refs = get_reference_by_doi(result['doi'])
            if refs is not None:
                refs = download_refs(b, dir, refs, records, available_scihub_url)
            # put refs into center paper session
            info['refs'] = refs
        # record center paper session
        info.update(result)
        records.center_paper[result['doi']] = info

if __name__ == "__main__":
    # process args
    args_paser = argparse.ArgumentParser()
    args_paser.add_argument("-i", "--ris", type=str, help="ris file path")
    args_paser.add_argument("-o", "--out", type=str, help="out files directory")
    args_paser.add_argument("-r", "--ref", action="store_true", help="FLAG, enable ref mode to download refrences")
    args_paser.add_argument("--head", action="store_true", help="FLAG, enable browser GUI")
    args_paser.add_argument("-u", "--undetected", action="store_true", help="FLAG, enable to use undetected_chromedriver")
    args = args_paser.parse_args()
    
    # command line path process
    args.ris = args.ris.replace('"', '').replace('\'', '')
    args.out = args.out.replace('"', '').replace('\'', '')
    
    # get available_scihub_urls
    try:
        available_scihub_url = _update_available_scihub_urls()[0]
    except:
        available_scihub_url = 'https://sci-hub.ren'
        put_err(f"can not get available_scihub_urls, try with {available_scihub_url}")
    
    # parse ris and set records
    infos = parse_ris(args.ris, '')
    records_path = os.path.join(args.out, '_records.json')
    records = Record()
    if check_parameters_path(records_path):
        records.from_json(records_path)
        
    # check output directory
    if not check_parameters_path(args.out):
        os.makedirs(args.out)
        
    # show args
    put_log(f'get args: ris: {args.ris}')
    put_log(f'get args: out: {args.out}')
    put_log(f'get args: ref: {args.ref}')
    put_log(f'get args: head: {args.head}')
    put_log(f'get args: undetected: {args.undetected}')
    put_log('downloading papers from SCIHUB, Enter e to stop and save session.')
    
    # setup web server
    launch_sub_thread()
    if args.head:
        b = Browser('Chrome', options=['--no-sandbox'], use_undetected=args.undetected)
    else:
        b = Browser('Chrome', options=['--headless', '--no-sandbox'], use_undetected=args.undetected)

    # download
    prog_bar = tqdm.tqdm(desc="downloading papers", total= len(infos))
    for info in infos:
        if info['doi'] not in records.center_paper:
            download_center_paper_session(b, args.out, info, records,
                                          available_scihub_url, args.ref)
        else:
            refs = get_reference_by_doi(info['doi'])
            if refs is not None and 'refs' in records.center_paper[info['doi']]:
                # NOTE: 有可能一个doi在一个RIS文件中出现了两次，所以此处为了防止第二次出现时报错，先判断是否有key
                if len(records.center_paper[info['doi']]['refs']) != len(refs):
                    download_refs(b, records_path, records.center_paper[info['doi']]['refs'], records)
                
        prog_bar.update(1)
        
        if statues_que_opts(statuesQue, "quit", "getValue"):
            break
    
    records.to_json(records_path)