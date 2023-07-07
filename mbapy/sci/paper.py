'''
Date: 2023-07-07 20:51:46
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-07-07 23:57:49
FilePath: \BA_PY\mbapy\sci\paper.py
Description: 
'''
import os, re

import PyPDF2
from scihub_cn.scihub import SciHub

if __name__ == '__main__':
    # dev mode
    from mbapy.base import put_err
else:
    # release mode
    from ..base import put_err

scihub = SciHub()

def download_by_scihub(doi: str, dir: str, use_title_as_name: bool = True):
    """
    Download a paper from Sci-Hub using its DOI.

    Parameters:
        doi (str): The DOI of the paper.
        dir (str): The directory where the downloaded paper will be saved.
        use_title_as_name (bool, optional): Whether to use the paper's title as the file name. Defaults to True.

    Returns:
        dict or None: If successful, returns a dictionary containing information about the paper. 
                      If there is an error, returns an error message. If the download fails, returns None.
    """
    res, paper_info = scihub.fetch({'doi':doi})
    file_name = (paper_info.title if use_title_as_name else doi) + '.pdf'

    if type(res) == dict and 'err' in res:        
        return put_err(res['err'])
    if not res:
        return None
    scihub._save(res.content, os.path.join(dir, file_name))
    return paper_info

def has_section_bookmarks(pdf_path:str = None, pdf_obj = None, section_names:list[str]=[]):
    """
    Checks if a PDF document has bookmarks for specified sections.

    Parameters:
        pdf_path (str): The path to the PDF document. Defaults to None.
        pdf_obj: The PDF object(Being opened!). Defaults to None.
        section_names (list[str]): A list of section names to check for bookmarks. Defaults to an empty list.

    Returns:
        bool: True if the PDF document has bookmarks for any of the specified section names, False otherwise.
    """
    if not os.path.isfile(pdf_path):
        put_err(f'{pdf_path:s} does not exist', None)
    if pdf_obj is None:
        with open(pdf_path, 'rb') as file:
            pdf_obj = PyPDF2.PdfReader(file)
            outlines = pdf_obj.outline
    else:
        outlines = pdf_obj.outline
            
    if not section_names:
        section_names = ['Abstract', 'Introduction', 'Materials', 'Methods',
                            'Results', 'Discussion', 'References']
    for outline in outlines:
        for section_name in section_names:
            pattern = r'\b{}\b'.format(re.escape(section_name))
            if re.search(pattern, outline.title, re.IGNORECASE):
                return True
    return False
    
def _parse_section_bookmarks(*bookmarks):
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
            ret = ret + _parse_section_bookmarks(*bookmark)
        else:
            ret.append(bookmark)
    return ret
    
def get_section_bookmarks(pdf_path:str = None, pdf_obj = None):
    """
    Returns a list of titles of bookmark sections in a PDF.

    Parameters:
    - pdf_path (str): The path to the PDF file. Defaults to None.
    - pdf_obj: The PDF object(Being opened!). Defaults to None.

    Returns:
    - list: A list of titles of bookmark sections in the PDF. Returns None if there are no bookmark sections or if the PDF file does not exist.
    """
    def worker(pdf_obj):
        if not has_section_bookmarks(pdf_path, pdf_obj):
            return None
        sections = _parse_section_bookmarks(pdf_obj.outline)
        return [section.title for section in sections]
    if not os.path.isfile(pdf_path):
        return put_err(f'{pdf_path:s} does not exist', None)
    if pdf_obj is None:
        with open(pdf_path, 'rb') as file:
            pdf_obj = PyPDF2.PdfReader(file)
            return worker(pdf_obj)
    else:
        return worker(pdf_obj)
    
def get_english_part_of_bookmarks(bookmarks:list[str]):
    english_bookmarks = []
    for bookmark in bookmarks:
        match = re.search(r'[a-zA-Z]+[a-zA-Z\s\S]+', bookmark)
        english_bookmarks.append(match.group(0).strip() if match else bookmark)
    return english_bookmarks

def get_section_from_paper(paper:str, key:str,
                           keys:list[str] = ['Title', 'Authors', 'Abstract', 'Keywords',
                                             'Introduction', 'Materials & Methods',
                                             'Results', 'Discussion', 'References']):
    """
    extract section of a science paper by key
    
    Parameters:
        paper (str): a science paper.
        key (str): one of the sections in the paper.
            can be 'Title', 'Authors', 'Abstract', 'Keywords', 'Introduction', 'Materials & Methods', 'Results', 'Discussion', 'References'
        keys (list[str], optional): a list of keys to extract. Defaults to ['Title', 'Authors', 'Abstract', 'Keywords', 'Introduction', 'Materials & Methods', 'Results', 'Discussion', 'References'].
    """
    # 构建正则表达式模式，使用re.IGNORECASE标志进行不区分大小写的匹配
    pattern = r'\b{}\b.*?(?=\b{})'.format(key, keys[keys.index(key)+1] if key != keys[-1] else '$')
    # 使用正则表达式匹配内容
    match = re.search(pattern, paper, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0)
    else:
        return put_err(f'key "{key}" not found in paper', '')

def format_paper_from_txt(content:str,
                          struct:list[str] = ['Title', 'Authors', 'Abstract', 'Keywords',
                                              'Introduction', 'Materials & Methods',
                                              'Results', 'Discussion', 'References']):
    content = content.replace('\n', '')
    struction = {}
    for key in struct:
        struction[key] = ''

if __name__ == '__main__':
    # dev code
    from mbapy.base import rand_choose
    from mbapy.file import convert_pdf_to_txt
    
    # download
    # download_by_scihub('10.3389/fpls.2018.00696', 'E:/')
    
    # extract section
    pdf_path = r"./data_tmp/DiffBP Generative Diffusion of 3D Molecules for Target Protein Binding.pdf"
    sections = get_english_part_of_bookmarks(get_section_bookmarks(pdf_path))
    print(sections)
    paper = convert_pdf_to_txt(pdf_path)
    print(get_section_from_paper(paper, rand_choose(sections, 0), keys=sections))
    
