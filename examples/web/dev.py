import base
from web import *

base.__NO_ERR__ = True

new_lines = []
with open(r"C:\Users\Administrator\Downloads\1668356676268.md", 'r', encoding='utf-8') as f_r:
    with open(r"C:\Users\Administrator\Downloads\1668356676268_new.md", 'w', encoding='utf-8') as f_w:
        new_lines = ['\n#### ' + get_between_re(line, '', r'\[\\\[source\\\]\].+?\n', ret_head=True) \
            if line.count("source") == 1 else line \
                for line in f_r.readlines()]
        f_w.writelines(new_lines)