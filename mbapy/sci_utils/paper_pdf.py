'''
Date: 2023-07-17 20:41:42
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-07-17 20:42:36
FilePath: \BA_PY\mbapy\sci_utils\paper_pdf.py
Description: 
'''

import os, re
from typing import List, Dict

import rispy
import PyPDF2

if __name__ == '__main__':
    # dev mode
    from mbapy.base import *
    from mbapy.file import replace_invalid_path_chr, convert_pdf_to_txt, read_text, opts_file
    import mbapy.web as web
else:
    # release mode
    from ..base import *
    from ..file import replace_invalid_path_chr, convert_pdf_to_txt, opts_file
    from .. import web


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
        with open(pdf_path, 'rb') as file:
            pdf_obj = PyPDF2.PdfReader(file)
            outlines = _get_outlines(pdf_obj)
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
            'Results', 'Discussion', 'References' will be searched.

    Returns:
        List[str]: A list of section names found in the PDF.
    """
    # check parameters
    if pdf_path is None and pdf_obj is None:
        return put_err('pdf_path or pdf_obj is None', None)
    # set default section names
    if not section_names:
        section_names = ['Abstract', 'Introduction', 'Materials', 'Methods',
                         'Results', 'Discussion', 'References']
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
        if content.find(section) != -1:
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
        with open(pdf_path, 'rb') as file:
            pdf_obj = PyPDF2.PdfReader(file)
            return worker(pdf_obj)
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
    # 构建正则表达式模式，使用re.IGNORECASE标志进行不区分大小写的匹配
    if paper is None or key is None:
        return put_err('paper or key is None', None)
    # TODO : sometimes may fail
    pattern = r'\b{}\b.*?(?=\b{})'.format(key, keys[keys.index(key)+1] if key != keys[-1] else '$')
    # 使用正则表达式匹配内容
    match = re.search(pattern, paper, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0)
    else:
        return put_err(f'key "{key}" not found in paper', '')

def format_paper_from_txt(content:str,
                          struct:List[str] = ['Title', 'Authors', 'Abstract', 'Keywords',
                                              'Introduction', 'Materials & Methods',
                                              'Results', 'Discussion', 'References']):
    content = content.replace('\n', '')
    struction = {}
    for key in struct:
        struction[key] = get_section_from_paper(content, key, struct)
    return struction