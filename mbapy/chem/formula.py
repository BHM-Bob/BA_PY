import os
import re
from functools import lru_cache
from typing import Dict, List

import pandas as pd

if __name__ == '__main__':
    from mbapy.file import opts_file
else:
    from ..file import opts_file
    
    
formula_existence_cache = {}
FORMULA_EXISTENCE_CACHE_PATH = os.path.expanduser(f'~/.mbapy/cache/formula_existence.pkl')

@lru_cache(maxsize=1024, typed=False)
def check_formula_existence(formula: str, link: int = 0):
    """
    check if a given formula can be formed by a given number of links
    
    Parameters:
        - formula (str): the chemical formula to check, such as 'C3H4O1N'
        - link (int): the number of links of the group defined by the formula
        
    Returns:
        - links (np.ndarray): the matrix of links, shape (len(formula), len(formula))
        - indexs (List[str]): the index of the matrix, such as ['LINK', 'C1', 'C2', 'C3', 'H1', 'O1', 'N1']
    
    Kown issues:
        - atoms may split into several unlinked groups.
    Notes:
    
    |  |L1|N1|H1|H2|  
    |L1|x |  |  |  |  
    |N1|1 |x |  |  |  
    |H1|0 |1 |x |  |  
    |H2|0 |1 |0 |x |  
    """
    # check cache
    global formula_existence_cache
    if not formula_existence_cache:
        os.makedirs(os.path.dirname(FORMULA_EXISTENCE_CACHE_PATH), exist_ok=True)
        if os.path.exists(FORMULA_EXISTENCE_CACHE_PATH):
            formula_existence_cache = opts_file(FORMULA_EXISTENCE_CACHE_PATH, 'rb', way='pkl') or {'flag': 'loaded'}
        else:
            formula_existence_cache['flag'] = 'loaded'
    if (formula, link) in formula_existence_cache:
        return formula_existence_cache[(formula, link)]
    # init link info
    link_num = {'LINK': link, 'C': 4, 'H': 1, 'O': 2, 'N': 3, 'S': 2}
    atom_dict = {atom: (int(count) if count else 1) for atom, count in re.findall(r'([A-Z])(\d*)', formula.upper())}
    indexs = ['LINK'] + [f'{n}{i}' for n, c in atom_dict.items() for i in range(c)]
    var = [[None] * len(indexs) for _ in range(len(indexs))]
    # check unsaturation
    if 2*atom_dict.get('C', 0) + 2 + atom_dict.get('N', 0) - atom_dict.get('H', 0) - link < 0:
        return None, None
    # check flag: only one non-H atom
    more_one_non_H_atom = any(atom_dict[n[0]] > 1 for n in indexs[1:] if n[0] != 'H')
    # init solver
    from ortools.linear_solver import pywraplp
    solver = pywraplp.Solver.CreateSolver("SAT")
    for i in range(1, len(indexs)):
        for j in range(0, i):
            # add condition: atom-pair link can not exceed max link num
            max_v = min(link if i == 0 else link_num[indexs[i][0]],
                        link if j == 0 else link_num[indexs[j][0]])
            var[i][j] = solver.IntVar(0, max_v, f'{indexs[i]}-{indexs[j]}')
            # add condition: same atom can not make max link
            if indexs[i][0] == indexs[j][0]:
                solver.Add(var[i][j] <= link_num[indexs[i][0]] - 1)
            # add condition: H can not make link to LINK
            if j == 0 and indexs[i][0] == 'H':
                solver.Add(var[i][j] == 0)
    # add conditions for atom-pair link
    for i in range(0, len(indexs)):
        if i == 0: # LINK
            solver.Add(sum([var[j][0] for j in range(1, len(indexs))]) == link)
        else: # other atom link
            v = link_num[indexs[i][0]]
            solver.Add(sum([var[i][j] for j in range(0, i)] + [var[i_tmp][i] for i_tmp in range(i+1, len(indexs))]) == v)
            # add condition: all atom can not donate all links to H if non-H atom num is bigger than 1
            if more_one_non_H_atom and indexs[i][0] != 'H':
                solver.Add(sum([var[i][j] for j in range(0, i) if indexs[j][0] == 'H'] + [var[i_tmp][i] for i_tmp in range(i+1, len(indexs)) if indexs[i_tmp][0] == 'H']) <= v-1)
    # add objective
    solver.Maximize(var[-1][-2])
    # solve
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        return None, None
    # retrive solution
    for i in range(1, len(indexs)):
        for j in range(0, i):
            var[i][j] = var[i][j].solution_value()
    # update cache
    formula_existence_cache[(formula, link)] = (var, indexs)
    return var, indexs
    
    
if __name__ == '__main__':
    # dev code
    from mbapy.base import TimeCosts
    @TimeCosts(10)
    def test_fn(idx):
        formula = 'CO2H'
        link = 1
        links, indexs = check_formula_existence(formula, link)
        if links is not None:
            df = pd.DataFrame(links, columns=indexs, index=indexs)
            print(df)
        
    test_fn()
    opts_file(FORMULA_EXISTENCE_CACHE_PATH, 'wb', way='pkl', data=formula_existence_cache)
    