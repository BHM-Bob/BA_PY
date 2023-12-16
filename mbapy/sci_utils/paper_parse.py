'''
Date: 2023-07-17 20:41:42
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-11-24 10:06:09
FilePath: \BA_PY\mbapy\sci_utils\paper_pdf.py
Description: 
'''

import os
import re
from typing import Dict, List, Tuple

import PyPDF2
import rispy
from bs4 import BeautifulSoup
from bs4 import element as bsel

if __name__ == '__main__':
    # dev mode
    from mbapy.base import (check_parameters_path, get_default_for_None,
                            parameter_checker, put_err)
    from mbapy.file import (convert_pdf_to_txt, opts_file,
                            replace_invalid_path_chr)
else:
    # release mode
    from ..base import (check_parameters_path, get_default_for_None,
                        parameter_checker, put_err)
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


def get_citation_position(pdf_path, refs: List[Dict[str, str]] = None):
    # refs是从corssref获取的refs，作为参考文献数目的参考
    import pdfplumber
    def _encode_bytes(data: bytes):
        try:
            return data.decode('utf-8')
        except:
            try:
                return data.decode('utf-16')
            except:
                return str(data).replace("'", "")[1:] # maybe "b'bib1'", transfer to "bib1"
    def _extract_text_by_rect(page, annot):
        if annot['top'] < 0 and annot['bottom'] < 0:
            annot['top'], annot['bottom'] = -annot['top'], -annot['bottom']
        if annot['top'] > annot['bottom']:
            annot['top'], annot['bottom'] = annot['bottom'], annot['top']
        sub_page = page.crop((annot['x0']-15, annot['top'],
                              annot['x1']+15, annot['bottom']))
        return sub_page.extract_text(y_tolerance = 0, x_tolerance = 3, layout=True)
    def _parse_annot_text(text: str, full_text: str):
        if text.count('\n') > 0:
            # 多行文本，此时可能是上标，取倒数第二行为数字上标，最后一行为被插入文本
            # 在被插入文本的约中间位置的第一个空格前插入数字上标
            annot_lines = text.split('\n')
            # get insert pos
            if ' ' in annot_lines[-1]:
                st_pos = int(len(annot_lines[-1]) * 0.2)
                insert_pos = annot_lines[-1][st_pos:].find(' ') + st_pos
            else:
                insert_pos = len(annot_lines[-1])
            # get ref number str
            results = re.findall('\d+', annot_lines[-2])
            ref_num = results[0] if results else '' # 如果倒二行没有数字, 说明上标被识别到倒一行了, 此时将跳过ref_num
            text = annot_lines[-1][:insert_pos] + ref_num + annot_lines[-1][insert_pos:]
        result =  re.findall('\S.+', text)[0] # 非空格开头
        result = re.sub('\s+', ' ', result) # 将连续空格替换为单个空格
        result = result[:-1] if result[-1] == ' ' else result # 去除结尾空格
        success = result in full_text
        if not success:
            # 如果此时不能匹配，则忽略空格做最大匹配
            non_sp_result = result.replace(' ', '')
            first_chr = non_sp_result[0].replace('[', '\[').replace(']', '\]').replace('(', '\(').replace(')', '\)')
            for matched in re.finditer(first_chr, full_text):
                if non_sp_result in full_text[matched.regs[0][0]:matched.regs[0][0]+2*len(result)+1].replace(' ', ''):
                    return full_text[matched.regs[0][0]:matched.regs[0][0]+2*len(result)+1], True
            # TODO: fix more
        return result, success
    def _parse_annots(pdf: pdfplumber.PDF):
        annots = []
        for page in pdf.pages:
            try:
                page_annots = page.annots
            except:
                continue
            for annot in page_annots:
                if 'Dest' in annot['data'] and isinstance(annot['data']['Dest'], bytes):
                    # dest start with bib or R or bb, if not, just assign to ' '
                    dest = (re.findall(r'(?:bib\d+$|R\d+|bb\d+$)', _encode_bytes(annot['data']['Dest'])) or ' ')[0]
                    if dest.startswith('bib'):
                        # Example: bib1 for Refs No.1
                        # Example: lbib3 for Refs No.3
                        annot_idx = int(dest[3:])
                    elif dest.startswith('R'):
                        annot_idx = int(dest[1:])
                    elif dest.startswith('bb'):
                        annot_idx = -1 # 'bb0005' may means ref1, do not assgin ref_idx
                    else:
                        continue
                    annot_str, annot_success = _parse_annot_text(_extract_text_by_rect(page, annot), pdf_text)
                    annots.append([annot_idx, annot_str, annot_success])
                elif 'Dest' in annot['data'] and isinstance(annot['data']['Dest'], list):
                    pass
                elif 'A' in annot['data'] and annot['data']['A']['S'].name == 'GoTo':
                    # Example: 4e8697fe-fcf7-4002-8235-59e0e1d0f61f.indd:R6:1811 for Refs No.6
                    # Example: 4e8697fe-fcf7-4002-8235-59e0e1d0f61f.indd:BLK_F1:2010 for Figure No.1
                    ref = re.findall('R\d+', _encode_bytes(annot['data']['A']['D']))
                    if not ref:
                        continue
                    annot_idx = int(ref[0][1:])
                    annot_str, annot_success = _parse_annot_text(_extract_text_by_rect(page, annot), pdf_text)
                    annots.append([annot_idx, annot_str, annot_success])
                else:
                    pass
            pass
        return annots    
    
    try:
        pdf_text = convert_pdf_to_txt(pdf_path, backend = 'pdfminer')
    except:
        return put_err(f'Maybe not a valid pdf file: {pdf_path}, return None', None)
    pdf_text = pdf_text.replace('ﬁ', 'fi')
    # opts_file('./data_tmp/pdf.txt', mode = 'w', data=pdf_text)
    with pdfplumber.open(pdf_path) as pdf:
        annots = _parse_annots(pdf)
        if not annots:
            # find with []
            bracket_patten = re.findall('\[ ?\d+ ?(?:[-–][ \d]+)?(?: *,[ \d]+(?:[-–][ \d]+)?)*\]', pdf_text)
            if len(bracket_patten) < 10:
                # find with ()
                bracket_patten = re.findall('\( ?\d+ ?(?:[-–][ \d]+)?(?: *,[ \d]+(?:[-–][ \d]+)?)*\)', pdf_text)
            if len(bracket_patten) < 10:
                pass

@parameter_checker(check_parameters_path, raise_err=False)
def parse_grobid(xml_path: str, encoding = 'utf-8'):
    def _T(element: bsel.Tag):
        if isinstance(element, bsel.Tag):
            return element.text.strip()
        elif isinstance(element, dict):
            return {_T(k_i):_T(v_i) for k_i, v_i in element.items()}
        elif isinstance(element, list) or isinstance(element, tuple):
            return [_T(v_i) for v_i in element]
        else:
            return element
    def _search_ref(content: str, types: List[str] = ['bibr', 'figure', 'table']):
        return re.search('|'.join([f'(?:<ref[^>]*?type="{ty}"[^>]*?>.+?</ref>)' for ty in types]),
                         content, re.DOTALL)
    soup = BeautifulSoup(open(xml_path, encoding=encoding), 'xml')
    article_title = soup.find('titleStmt')
    date = soup.find('publicationStmt').find('date')
    article_publication_date = date['when'] if date and 'when' in date else _T(date)
    article_authors, authors = soup.find('sourceDesc').findAll('author'), {}
    for author in article_authors:
        author_name = author.find('persName')
        if author_name:
            author_name = author_name
            authors[author_name] = {'email': author.find('email')}
            for i, aff in enumerate(author.findAll('affiliation')):
                has_org, has_add =  aff.find('orgName'), aff.find('address')
                authors[author_name][f'aff_{i}'] = {
                    'department': aff.find('orgName', type="department") if has_org else None,
                    'institution': aff.find('orgName', type="institution") if has_org else None,
                    'settlement': aff.find('address').find('settlement') if has_add else None,
                    'region': aff.find('address').find('region') if has_add else None,
                    'country': aff.find('address').find('country') if has_add else None,
                }
    article_doi = soup.find('idno', type="DOI")
    article_submission = soup.find('note', type="submission")
    article_abs = soup.find('abstract').find('p')
    article_sections = []
    for section in soup.find('body').findAll('div', xmlns="http://www.tei-c.org/ns/1.0"):
        content, ref_pos = '\n'.join([str(sec)[3:-4] for sec in section.findAll('p')]), []
        # 转化figure, ref的XML格式
        while _search_ref(content):
            # figure的XML会有变化:<ref target="#fig_0" type="figure">1</ref>和<ref type="figure">2</ref>
            # ref的XML为<ref target="#b5" type="bibr">( 6 )</ref>
            ref = _search_ref(content)
            if 'figure' in ref.group(0) or 'table' in ref.group(0):
                ref_idx = re.search(r'>[^<]+?<', ref.group(0)).group(0)[1:-1]
            elif 'bibr' in ref.group(0):
                ref_idx = re.search(r'\d+', ref.group(0))
                if ref_idx:
                    ref_idx= ref_idx.group(0)
                else:
                    content = content.replace(ref.group(0), ' ')
                    continue
            ref_type = re.search(r'type="\w+?"', ref.group(0)).group(0)[6:-1]
            ref_pos.append({'ref_type': ref_type, 'ref_idx':ref_idx, 'ref_pos': ref.regs[0][0]})
            content = content.replace(ref.group(0), ref_idx+',')
        article_sections.append({'title':section.find('head'), 'content': content,
                                 'ref_pos': ref_pos})
    refs = []
    for ref in soup.find('back').findAll('biblStruct'):
        idx = re.findall('\d+', ref['xml:id'])[0] # start from '0'
        refs.append({
            'title': ref.find('title', level="a", type="main"),
            'authors': [au for au in ref.findAll('author')],
            'monogr': {ref.find('monogr').find('title'): ref.find('monogr').find('date')},
        })
    return _T({
        'title': article_title,
        'submission': article_submission,
        'pub_date': article_publication_date,
        'authors': authors,
        'doi': article_doi,
        'abs': article_abs,
        'sections': article_sections,
        'refs': refs,
        })

__all__ = [
    'has_sci_bookmarks',
    'get_sci_bookmarks_from_pdf',
    'get_section_bookmarks',
    'get_english_part_of_bookmarks',
    'get_section_from_paper',
    'format_paper_from_txt',
    'parse_grobid',
]

if __name__ == '__main__':
    # dev code
    # convert pdf to text
    pdf_path = r'data_tmp\papers\Contrasting effects of linaclotide and lubiprostone on restitution of epithelial cell barrier properties and cellular homeostasis after exposure to cell stressors.pdf'
    pdf_text = convert_pdf_to_txt(pdf_path, backend = 'pdfminer')\
        .replace('\u00a0', ' ').replace('-\n', '').replace('  ', ' ')
    opts_file('data_tmp/text.txt', 'w', data = pdf_text)
    print(pdf_path)
    bookmarks = get_english_part_of_bookmarks(get_section_bookmarks(pdf_path))
    pdf_data = format_paper_from_txt(pdf_text, bookmarks)
    pass