'''
Date: 2023-08-03 19:54:12
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-01-06 21:35:10
Description: 
'''
import argparse
import glob
import os

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'True'
from tqdm import tqdm

import mbapy.web as web
from mbapy.base import *
from mbapy.file import *
from mbapy.paper import *


def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser()
    args_paser.add_argument("-i", "--input", type=str, help="paper(pdf) file directory")
    args_paser.add_argument("-o", "--output", type=str, default='_mbapy_extract_paper.json', help="output file name")
    args_paser.add_argument("-b", "--backend", type=str, default='pdfminer', help="paper(pdf) file directory")
    args_paser.add_argument("-l", "--log", action = 'store_true', help="FLAGS, enable log")
    args = args_paser.parse_args(sys_args)
    
    args.input = args.input.replace('"', '').replace('\'', '')
    pdf_paths = glob.glob(os.path.join(args.input, '*.pdf'))
    
    put_log(f'get args: input: {args.input}')
    put_log(f'get args: output: {args.output}')
    put_log(f'get args: backend: {args.backend}')
    put_log(f'get args: log: {args.log}')
    put_log('extracting papers content, Enter e to simply stop.')
        
    if not args.log:
        Configs.err_warning_level == 999
        
    bar =tqdm(total=len(pdf_paths))
    web.launch_sub_thread()
    sum_has_bookmarks = 0
    data = {}
    for pdf_path in pdf_paths:
        try:
            bookmarks = get_english_part_of_bookmarks(get_section_bookmarks(pdf_path))
            pdf_text = convert_pdf_to_txt(pdf_path, backend = args.backend)\
                .replace('\u00a0', ' ').replace('-\n', '').replace('  ', ' ')
            pdf_data = format_paper_from_txt(pdf_text, bookmarks)
            if pdf_data is not None:
                data[pdf_path.split('/')[-1]] = pdf_data
                sum_has_bookmarks += 1
        except:
            try:
                pdf_text = convert_pdf_to_txt(pdf_path, backend = args.backend)\
                    .replace('\u00a0', ' ').replace('-\n', '').replace('  ', ' ')
                data[pdf_path.split('/')[-1]] = pdf_text
            except:
                put_err(f'can not parse pdf: {pdf_path}, skip.')
        bar.update(1)
        if web.statues_que_opts(web.statuesQue, 'quit', 'getValue'):
            break
    save_json(os.path.join(args.input, args.output), data)
    put_log(f'sum has bookmarks:{sum_has_bookmarks}')

if __name__ == '__main__':
    main()