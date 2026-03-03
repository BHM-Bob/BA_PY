'''
Date: 2026-03-03
LastEditors: BHM-Bob, MiniMax-M2.5
Description: extract pdf table of contents to markdown
'''
import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import PyPDF2
from tqdm import tqdm

from mbapy.base import put_err
from mbapy.file import opts_file
from mbapy.scripts._script_utils_ import clean_path, show_args


def extract_bookmarks(outline, level: int = 0) -> List[Tuple[str, int, any]]:
    result = []
    for item in outline:
        if isinstance(item, list):
            result.extend(extract_bookmarks(item, level + 1))
        else:
            result.append((item.title, level, item))
    return result


def get_toc_from_pdf(pdf_path: str) -> List[Tuple[str, int, int]]:
    toc_data = []
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            if hasattr(reader, 'outline') and reader.outline:
                bookmarks = extract_bookmarks(reader.outline)
                for title, level, dest in bookmarks:
                    page_num = 0
                    if hasattr(dest, 'get') and dest is not None:
                        try:
                            if isinstance(dest, PyPDF2.generic.Destination):
                                page_num = reader.get_destination_page_number(dest)
                            elif isinstance(dest, dict):
                                if '/D' in dest:
                                    page_num = dest['/D'].get_object_number()
                        except:
                            pass
                    elif hasattr(dest, 'page'):
                        try:
                            page_num = dest.page.get_object_number()
                            if hasattr(reader, 'pages'):
                                for i, page in enumerate(reader.pages):
                                    if page.get_object() == page_num:
                                        page_num = i
                                        break
                        except:
                            pass
                    toc_data.append((title, level, page_num))
            else:
                put_err('No bookmarks found in PDF')
                return []
    except Exception as e:
        put_err(f'Error reading PDF: {e}')
        return []
    return toc_data


def extract_first_last_paragraphs(page_text: str, max_length: int = 500) -> Tuple[str, str]:
    if not page_text or not page_text.strip():
        return '', ''
    
    paragraphs = re.split(r'\n\s*\n|\n{2,}', page_text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if not paragraphs:
        return '', ''
    
    first_para = paragraphs[0]
    last_para = paragraphs[-1] if len(paragraphs) > 1 else first_para
    
    if len(first_para) > max_length:
        first_para = first_para[:max_length] + '...'
    if len(last_para) > max_length:
        last_para = last_para[:max_length] + '...'
    
    return first_para, last_para


def get_toc_with_content(pdf_path: str, toc_data: List[Tuple[str, int, int]], 
                         max_para_len: int = 500) -> List[Tuple[str, int, int, str, str]]:
    result = []
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            total_pages = len(reader.pages)
            
            for i, (title, level, page_num) in enumerate(tqdm(toc_data, desc='Extracting content')):
                first_para, last_para = '', ''
                
                if 0 <= page_num < total_pages:
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    first_para, last_para = extract_first_last_paragraphs(text, max_para_len)
                
                result.append((title, level, page_num, first_para, last_para))
    except Exception as e:
        put_err(f'Error extracting content: {e}')
    
    return result


def format_toc_markdown(toc_data: List[Tuple[str, int, int]]) -> str:
    lines = []
    lines.append('# Table of Contents\n')
    for title, level, page_num in toc_data:
        indent = '  ' * level
        if page_num > 0:
            line = f'{indent}- [{title}](#page-{page_num})'
        else:
            line = f'{indent}- {title}'
        lines.append(line)
    return '\n'.join(lines)


def format_toc_with_content_markdown(toc_data: List[Tuple[str, int, int, str, str]]) -> str:
    lines = []
    lines.append('# Table of Contents with Content\n')
    for title, level, page_num, first_para, last_para in toc_data:
        indent = '  ' * level
        if page_num > 0:
            lines.append(f'{indent}- [{title}](#page-{page_num})')
        else:
            lines.append(f'{indent}- {title}')
        
        if first_para or last_para:
            lines.append(f'{indent}  - First: {first_para}')
            if last_para != first_para:
                lines.append(f'{indent}  - Last: {last_para}')
            lines.append('')
    return '\n'.join(lines)


def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser()
    args_paser.add_argument('-i', '--input', type=str, required=True,
                            help='input pdf file path')
    args_paser.add_argument('-o', '--output', type=str, default=None,
                            help='output markdown file path, default is <input>_toc.md')
    args_paser.add_argument('-c', '--content', action='store_true', default=False,
                            help='FLAG, extract first and last paragraph from each TOC page')
    args_paser.add_argument('-l', '--max-length', type=int, default=500,
                            help='max length of extracted paragraphs, default is %(default)s')
    args = args_paser.parse_args(sys_args)

    args.input = clean_path(args.input)
    if args.output is None:
        suffix = '_toc_content.md' if args.content else '_toc.md'
        args.output = str(Path(args.input).with_suffix('')) + suffix
    else:
        args.output = clean_path(args.output)

    show_args(args, ['input', 'output', 'content', 'max_length'])

    if not os.path.isfile(args.input):
        put_err(f'PDF file not found: {args.input}')
        return

    print(f'Extracting TOC from: {args.input}')
    toc_data = get_toc_from_pdf(args.input)

    if not toc_data:
        put_err('No TOC extracted, exit.')
        return

    print(f'Found {len(toc_data)} TOC entries')

    if args.content:
        print('Extracting content from each page...')
        toc_with_content = get_toc_with_content(args.input, toc_data, args.max_length)
        markdown_content = format_toc_with_content_markdown(toc_with_content)
    else:
        markdown_content = format_toc_markdown(toc_data)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f'TOC saved to: {args.output}')


if __name__ == "__main__":
    main()
