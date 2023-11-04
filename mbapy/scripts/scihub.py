'''
Date: 2023-07-14 19:29:19
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-11-04 20:06:25
Description: try to download all papers in the RIS file
'''
import argparse
import os
import random
import time
from typing import Any, Dict

import requests
import tqdm

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'True'
from mbapy import base, file, game, paper, web


class Record(game.BaseInfo):
    def __init__(self) -> None:
        super().__init__()
        self.doi: Dict[str, Dict[str, str]] = {} # doi(str) - download_info(Dict[str, str])
        self.center_paper: Dict[str, Dict[str, Any]] = {} # doi(str) - center_paper_info(Dict[str, str|Dict[str, str]])
        
def download_refs(dir:str, refs:Dict[str, Any], records:Record):
    for ref_info in refs:
        if 'DOI' in ref_info and ref_info['DOI'] not in records.doi:
            time.sleep(5 + random.randint(5, 10))
            try:
                ref_result = paper.download_by_scihub(dir, ref_info['DOI'])
            except requests.exceptions.TooManyRedirects as e:
                ref_result = None
            if ref_result is not None:
                # put ref-result into ref-info to record ref-result automatically
                ref_result.__delitem__('res')
                ref_info.update(ref_result)
                # add doi
                records.doi[ref_info['doi']] = ref_result
        # quit
        if web.statues_que_opts(web.statuesQue, "quit", "getValue"):
            break
    return refs
        
# @base.parameter_checker(raise_err=False, dir = base.check_parameters_path)
def download_center_paper_session(dir:str, info: Dict[str, str], records:Record,
                                  is_download_ref: bool = False):
    result = paper.download_by_scihub(dir, info['doi'], info['title'])
    if result is not None and 'doi' in result:
        time.sleep(5 + random.randint(5, 10))
        records.doi[result['doi']] = result
        if is_download_ref:
            # get refs
            refs = paper.get_reference_by_doi(result['doi'])
            if refs is not None:
                refs = download_refs(dir, refs, records)
            # put refs into center paper session
            info['refs'] = refs
        # record center paper session
        result.__delitem__('res')
        info.update(result)
        records.center_paper[result['doi']] = info

if __name__ == "__main__":
    args_paser = argparse.ArgumentParser()
    args_paser.add_argument("-i", "--ris", type=str, help="ris file path")
    args_paser.add_argument("-o", "--out", type=str, help="out files directory")
    args_paser.add_argument("-r", "--ref", action="store_false", help="enable ref mode to download refrences")
    args = args_paser.parse_args()
    
    args.ris = args.ris.replace('"', '').replace('\'', '')
    args.out = args.out.replace('"', '').replace('\'', '')
    
    infos = paper.parse_ris(args.ris, '')
    records_path = os.path.join(args.out, '_records.json')
    records = Record()
    if base.check_parameters_path(records_path):
        records.from_json(records_path)
    
    web.launch_sub_thread()
    base.put_log(f'get args: ris: {args.ris}')
    base.put_log(f'get args: out: {args.out}')
    base.put_log(f'get args: ref: {args.ref}')
    base.put_log('downloading papers from SCIHUB, Enter e to stop and save session.')
    prog_bar = tqdm.tqdm(desc="downloading papers", total= len(infos))
    for info in infos:
        if info['doi'] not in records.center_paper:
            download_center_paper_session(args.out, info, records, args.ref)
        else:
            refs = paper.get_reference_by_doi(info['doi'])
            if refs is not None and 'refs' in records.center_paper[info['doi']]:
                # NOTE: 有可能一个doi在一个RIS文件中出现了两次，所以此处为了防止第二次出现时报错，先判断是否有key
                if len(records.center_paper[info['doi']]['refs']) != len(refs):
                    download_refs(records_path, records.center_paper[info['doi']]['refs'], records)
                
        prog_bar.update(1)
        
        if web.statues_que_opts(web.statuesQue, "quit", "getValue"):
            break
    
    records.to_json(records_path)