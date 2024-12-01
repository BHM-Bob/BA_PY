import re
from typing import Dict, List

import pandas as pd


def check_formula_existence(formula: str, link: int = 0):
    """
    check if a given formula can be formed by a given number of links
    
    Parameters:
        - formula (str): the chemical formula to check, such as 'C3H4O1N'
        - link (int): the number of links of the group defined by the formula
        
    Returns:
        - links (np.ndarray): the matrix of links, shape (len(formula), len(formula))
        - indexs (List[str]): the index of the matrix, such as ['LINK', 'C1', 'C2', 'C3', 'H1', 'O1', 'N1']
    
    Notes:
    
    |  |L1|N1|H1|H2|  
    |L1|x |  |  |  |  
    |N1|1 |x |  |  |  
    |H1|0 |1 |x |  |  
    |H2|0 |1 |0 |x |  
    """
    from ortools.linear_solver import pywraplp

    # helper function
    def get_max_link_num(i: int, link: int, link_num: Dict[str, int], indexs: List[str]) -> int:
        return link if i == 0 else link_num[indexs[i][0]]
    # init link info
    link_num = {'LINK': link, 'C': 4, 'H': 1, 'O': 2, 'N': 3, 'S': 2}
    atom_dict = {atom: (int(count) if count else 1) for atom, count in re.findall(r'([A-Z])(\d*)', formula.upper())}
    indexs = ['LINK'] + [f'{n}{i}' for n, c in atom_dict.items() for i in range(c)]
    var = [[None] * len(indexs) for _ in range(len(indexs))]
    # init solver
    solver = pywraplp.Solver.CreateSolver("SAT")
    for i in range(1, len(indexs)):
        for j in range(0, i):
            max_v = min(get_max_link_num(i, link, link_num, indexs),
                        get_max_link_num(j, link, link_num, indexs))
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
    # add objective
    solver.Maximize(var[-1][-2])
    # solve
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print("Solution: Objective value =", solver.Objective().Value())
    else:
        return None, None
    # retrive solution
    for i in range(1, len(indexs)):
        for j in range(0, i):
            var[i][j] = var[i][j].solution_value()
    return var, indexs
    
    
if __name__ == '__main__':
    # dev code
    formula = 'C3H4O1N1'
    link = 1
    links, indexs = check_formula_existence(formula, link)
    if links is not None:
        df = pd.DataFrame(links, columns=indexs, index=indexs)
        print(df)
    