'''
Date: 2023-07-17 20:41:42
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-07-17 20:42:36
FilePath: \BA_PY\mbapy\sci_utils\paper_pdf.py
Description: 
'''

import os
import re
from typing import Dict, List

import PyPDF2
import rispy

if __name__ == '__main__':
    # dev mode
    import mbapy.web as web
    from mbapy.base import *
    from mbapy.file import (convert_pdf_to_txt, opts_file, read_text,
                            replace_invalid_path_chr)
else:
    # release mode
    from .. import web
    from ..base import *
    from ..file import convert_pdf_to_txt, opts_file, replace_invalid_path_chr


def _flatten_pdf_bookmarks(*bookmarks):
    """
        Parse a list of bookmarks and return a flattened list of all bookmarks.

        Args:
            *bookmarks (List[Any]): A variable number of bookmark lists.

        Returns:
            List[Any]: A flattened list of all bookmarks.
    """
    ret = []
    for bookmark in bookmarks:
        if isinstance(bookmark, list):
            ret = ret + _flatten_pdf_bookmarks(*bookmark)
        else:
            ret.append(bookmark)
    return ret

def has_sci_bookmarks(pdf_path:str = None, pdf_obj = None, section_names:List[str]=[]):
    """
    Checks if a PDF document has bookmarks for scientific sections.

    Parameters:
        pdf_obj: The PDF object(Being opened!). Defaults to None.
        pdf_path (str): The path to the PDF document. Defaults to None.
        section_names (list[str]): A list of section names to check for bookmarks. Defaults to an empty list.

    Returns:
        list[str] or bool: list of section names if the PDF has bookmarks, False otherwise.
    """
    def _get_outlines(pdf_obj):
        if hasattr(pdf_obj, 'outline') and pdf_obj.outline:
            return pdf_obj.outline
        else:
            return []
    # check parameters
    if pdf_path is None and pdf_obj is None:
        return put_err('pdf_path or pdf_obj is None', None)
    # get outlines
    if pdf_obj is not None:
        outlines = _get_outlines(pdf_obj)
    elif pdf_path is not None and os.path.isfile(pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                pdf_obj = PyPDF2.PdfReader(file)
                outlines = _get_outlines(pdf_obj)
        except:
            return put_err(f'Something goes wrong with pdf path:{pdf_path}, return ""', "")
    # check for valid bookmarks, get flat section list
    if len(outlines) == 0:
        return False
    else:
        outlines = _flatten_pdf_bookmarks(*outlines)
    # set default section names
    if not section_names:
        section_names = ['Abstract', 'Introduction', 'Materials', 'Methods',
                         'Results', 'Discussion', 'References']
    # check whether any of the section names is in the outlines
    for outline in outlines:
        for section_name in section_names:
            pattern = r'\b{}\b'.format(re.escape(section_name))
            if re.search(pattern, outline.title, re.IGNORECASE):
                return outlines
    return False

def get_sci_bookmarks_from_pdf(pdf_path:str = None, pdf_obj = None, section_names:List[str]=[]):
    """
    Returns a list of section names from a scientific PDF.

    Parameters:
        pdf_path (str): The path to the PDF file. Default is None.
        pdf_obj: The PDF object. Default is None.
        section_names (List[str]): A list of section names to search for.
            If None, all sections include 'Abstract', 'Introduction', 'Materials', 'Methods',
            'Results', 'Conclusions, 'Discussion', 'References' will be searched.

    Returns:
        List[str]: A list of section names found in the PDF.
    """
    # check parameters
    if pdf_path is None and pdf_obj is None:
        return put_err('pdf_path or pdf_obj is None', None)
    # set default section names
    if not section_names:
        section_names = ['Abstract', 'Introduction', 'Materials', 'Methods',
                         'Results', 'Conclusions', 'Discussion', 'References']
    # get pdf full txt
    if pdf_obj is not None:
        # extract text from pdf obj
        content = '\n'.join([page.extract_text() for page in pdf_obj.pages])
    elif pdf_path is not None and os.path.isfile(pdf_path):
        # get text from pdf file
        content = convert_pdf_to_txt(pdf_path)
    # get section titles
    ret = []
    for section in section_names:
        if content.find(section+'\n') != -1:
            ret.append(section)
    return ret
    
def get_section_bookmarks(pdf_path:str = None, pdf_obj = None):
    """
    Returns a list of titles of bookmark sections in a PDF.

    Parameters:
    - pdf_path (str): The path to the PDF file. Defaults to None.
    - pdf_obj: The PDF object(Being opened!). Defaults to None.

    Returns:
    - list: A list of titles of bookmark sections in the PDF.
    Returns None if there are no bookmark sections or if the PDF file does not exist.
    """
    def worker(pdf_obj):
        sections = has_sci_bookmarks(None, pdf_obj)
        if not sections:
            # do not has inner bookmarks, just parse from text
            return get_sci_bookmarks_from_pdf(None, pdf_obj)
        # has inner bookmarks, parse from outline
        return [section.title for section in sections]
    # check path
    if not os.path.isfile(pdf_path):
        return put_err(f'{pdf_path:s} does not exist', None)
    # get section titles
    if pdf_obj is None:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_obj = PyPDF2.PdfReader(file)
                return worker(pdf_obj)
        except:
            return put_err(f'Something goes wrong with pdf path:{pdf_path}, return ""', "")
    else:
        return worker(pdf_obj)
    
def get_english_part_of_bookmarks(bookmarks:List[str]):
    """
    Retrieves the English part of the given list of bookmarks.

    Parameters:
        bookmarks (list[str]): A list of bookmarks.

    Returns:
        list[str]: A list containing only the English part of the bookmarks.
    """
    if bookmarks is None:
        return put_err('bookmarks is None', None)
    english_bookmarks = []
    for bookmark in bookmarks:
        match = re.search(r'[a-zA-Z]+[a-zA-Z\s\S]+', bookmark)
        english_bookmarks.append(match.group(0).strip() if match else bookmark)
    return english_bookmarks

def get_section_from_paper(paper:str, key:str,
                           keys:List[str] = ['Title', 'Authors', 'Abstract', 'Keywords',
                                             'Introduction', 'Materials & Methods',
                                             'Results', 'Discussion', 'References']):
    """
    extract section of a science paper by key
    
    Parameters:
        paper (str): a science paper.
        key (str): one of the sections in the paper.
            can be 'Title', 'Authors', 'Abstract', 'Keywords', 'Introduction',
            'Materials & Methods', 'Results', 'Discussion', 'References'
        keys (list[str], optional): a list of keys to extract.
            Defaults to ['Title', 'Authors', 'Abstract', 'Keywords', 'Introduction',
            'Materials & Methods', 'Results', 'Discussion', 'References'].
    """
    def _get_valid_key(key:str):
        return key.replace('(', '\(').replace(')', '\)')
    def _has_key(key:str, flags = re.DOTALL):
        return re.findall(r'\b{}(?i:{})\b'.format(key[0], key[1:]), paper, flags)
    def _get_longest(results:List[str]):
        length = [len(i) for i in results]
        return results[length.index(max(length))]
    def _get_match_by_key(key1:str, keys:List[str], key2 = None, flags = re.DOTALL):
        """
        有的文献虽然有Abstract（或其他第一个书签）章节书签，但是在文本中不写，此时取文献开头作为匹配
        有的文献书签首字母大写，但在全文中全字母大写，为了尽可能精确匹配，要求首字母大小写匹配而忽略剩余字符
        """
        key1_s =  _get_valid_key(key1)
        # 得到key2，如果未指定key2，则用key1的下一个key
        key2 = get_default_for_None(key2, keys[keys.index(key1)+1] if key1 != keys[-1] else None)
        if key2 is None:# 单层if会让VSCode认为下方代码为死代码。。。
            matchs = re.findall(r'\b{}(?i:{})\b[ \.\n].+?$'.format(key1_s[0], key1_s[1:]), paper, flags)
            if matchs:
                # 这时如果key1是最后一个key，那么key2就是$。如果key1能找到，直接返回
                return matchs[0]
            else:
                return put_err(f'key1 "{key}" not found in paper and it is the last one', '')
        # 得到合法的用于检索的key2
        key2_s = _get_valid_key(key2)
        has_key1, has_key2 = _has_key(key1_s), _has_key(key2_s)
        # 错误前处理。此时keyx都是原书签，而keyx_s都是合法的检索字符串
        if not has_key1 and not has_key2:
            if flags == (re.DOTALL | re.IGNORECASE):
                return put_err(f'key1 "{key1}" and key2 "{key2}" not found, return ""', "")
            else:
                return _get_match_by_key(key1, keys, key2, re.DOTALL | re.IGNORECASE)
        elif not has_key1 and has_key2 and keys.index(key1) == 0:
            # 如果只有key1没找到，且key1是第一个key，就是用全文第一个字符替代。
            pattern = r'{}.+?\b{}(?i:{})\b'.format(paper[0], key2_s[0], key2_s[1:])
        elif not has_key1 and has_key2 and keys.index(key1) > 0:
            # 如果只有key1没找到，且key1不是是第一个key，将key1提前一个
            return _get_match_by_key(keys[keys.index(key1)-1], keys, key2)
        elif has_key1 and not has_key2 and keys.index(key2) == len(keys) - 1:
            # 如果只有key2没找到，且key2是最后一个key，就用全文最后一个字符替代。
            pattern = r'\b{}(?i:{})\b.+?{}'.format(key1_s[0], key1_s[1:], paper[-1])
        elif has_key1 and not has_key2 and keys.index(key2) < len(keys) - 1:
            # 如果只有key2没找到，且key2不是最后一个key，将key2推后一个
            return _get_match_by_key(key1, keys, keys[keys.index(key2)+1])
        else:
            # 两个key都找到，正常构建pattern
            pattern = r'\b{}(?i:{})\b[ \.\n].+?\b{}(?i:{})\b[ \.\n]'.format(key1_s[0], key1_s[1:], key2_s[0], key2_s[1:])
        matchs = re.findall(pattern, paper, flags)
        # 错误后处理
        if not matchs and not flags == (re.DOTALL | re.IGNORECASE):
            # 如果还没找到，就忽略大小写再找一遍
            ignore_case_result = _get_match_by_key(key1, keys, key2, re.DOTALL | re.IGNORECASE)
            if ignore_case_result:
                return ignore_case_result
            else:# TODO：目前没办法
                return ''
        # 返回match
        return matchs
    
    if paper is None or key is None:
        return put_err('paper or key is None', None)
    # 给最后一个字符后加一个空格以便下面的检索
    paper = paper + ' '
    matchs = _get_match_by_key(key, keys)
    if matchs:
        return _get_longest(matchs) # 有的文献一个章节名存在多次，第一次集中出现，第二次为真正引导章节，取第二次（最长的），若存在更多次，忽略该情况
    else:
        return put_err(f'key "{key}" not found in paper', '')

def format_paper_from_txt(content:str,
                          struct:List[str] = ['Title', 'Authors', 'Abstract', 'Keywords',
                                             'Introduction', 'Materials & Methods',
                                             'Results', 'Discussion', 'References']):
    struction = {}
    for key in struct:
        struction[key] = get_section_from_paper(content, key, struct)
    return struction

if __name__ == '__main__':
    # dev code
    pdf_path = r'data_tmp\papers\Contrasting effects of linaclotide and lubiprostone on restitution of epithelial cell barrier properties and cellular homeostasis after exposure to cell stressors.pdf'
    pdf_text = convert_pdf_to_txt(pdf_path, backend = 'pdfminer')\
        .replace('\u00a0', ' ').replace('-\n', '').replace('  ', ' ')
    opts_file('data_tmp/text.txt', 'w', data = pdf_text)
    print(pdf_path)
    # pdf_text = convert_pdf_to_txt(pdf_path).replace('-\n', '').replace('  ', ' ')
    bookmarks = get_english_part_of_bookmarks(get_section_bookmarks(pdf_path))
    pdf_data = format_paper_from_txt(pdf_text, bookmarks)
    pass