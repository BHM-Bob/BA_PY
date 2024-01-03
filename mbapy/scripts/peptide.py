import argparse
import os
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
from mbapy import base, plot


class AnimoAcid:
    aa_mwd = { # amino acid molecular weight dict
        "Ala": 89.09,
        "Arg": 174.20,
        "Asn": 132.12,
        "Asp": 133.10,
        "Cys": 121.16,
        "Gln": 146.15,
        "Glu": 147.13,
        "Gly": 75.07,
        "His": 155.16,
        "Ile": 131.17,
        "Leu": 131.17,
        "Lys": 146.19,
        "Met": 149.21,
        "Phe": 165.19,
        "Pro": 115.13,
        "Ser": 105.09,
        "Thr": 119.12,
        "Trp": 204.23,
        "Tyr": 181.19,
        "Val": 117.15
    }
    pg_mwd = { # protect group molcular weight dict
        'H': 0, # do not calcu mw
        'OH': 0, # do not calcu mw
        'Acm': 72.10, # = 73.10 - 1.0(H)
        'Boc': 101.13, # = 102.13 - 1.0(H)
        'Fmoc': 223.26, # = 224.26 - 1.0(H)
        'OtBu': 57.12, # = 58.12 - 1.0(H)
        'tBu': 57.12, # = 58.12 - 1.0(H)
        'Trt': 243.34, # = 244.34 - 1.0(H)
    }
    all_mwd = deepcopy(aa_mwd)
    all_mwd.update(pg_mwd)
    def __init__(self, repr: str) -> None:
        parts = repr.split('-')
        if len(parts) == 1:
            assert parts[0][:3] in self.aa_mwd.keys(), f'{repr} is not a valid animo acid, it has noly one part and should in {self.aa_mwd.keys()}'
            parts = ['H'] + parts + ['OH']
        elif len(parts) == 2:
            if parts[0][:3] in self.aa_mwd.keys():
                parts = ['H'] + parts
            elif parts[1][:3] in self.aa_mwd.keys():
                parts = parts + ['OH']
            else:
                raise ValueError(f'{repr} is not a valid animo acid, it has two parts and none is in {self.aa_mwd.keys()} with it\'s previous 3 chars')
        elif len(parts) > 3:
            raise ValueError(f'{repr} is not a valid animo acid, it has more than 3 parts splited by dash \'-\'')
        self.N_protect = parts[0]
        self.animo_acid = parts[1]
        self.C_protect = parts[2]
        if len(parts[1]) > 3:
            self.animo_acid = parts[1][0:3]
            self.R_protect = parts[1][4:-1]
        else:
            self.R_protect = 'H'
    def make_pep_repr(self, is_N_terminal: bool = False, is_C_terminal: bool = False):
        parts = []
        parts += ([f'{self.N_protect}-'] if (self.N_protect != 'H' or is_N_terminal) else [])
        parts += ([self.animo_acid] if self.R_protect == 'H' else [f'{self.animo_acid}({self.R_protect})'])
        parts += ([f'-{self.C_protect}'] if (self.C_protect != 'OH' or is_C_terminal) else [])
        return ''.join(parts)
    def __repr__(self) -> str:
        return self.make_pep_repr(True, True)
    def calcu_mw(self, expand_mw_dict: Dict[str, float] = None):
        if expand_mw_dict is not None:
            assert isinstance(expand_mw_dict, dict), 'expand_mw_dict should be a dict contains protect group molecular weight'
            self.all_mwd.update(expand_mw_dict)
        parts = [self.N_protect, self.animo_acid, self.R_protect, self.C_protect]
        mw = sum([self.all_mwd[part] for part in parts])
        if self.N_protect != 'H':
            mw -= 1.0 # because N-terminal residue has no H but AA has H, so we need to minus 1.0 for AA
        if self.C_protect != 'OH':
            mw -= 1.0 # because C-terminal residue has no OH but AA has OH, so we need to minus 1.0 for AA
        if self.R_protect != 'H':
            mw -= 1.0 # because R-terminal residue has no H but AA has H, so we need to minus 1.0 for AA
        return mw
    
class Peptide:
    def __init__(self, repr: str) -> None:
        parts = repr.split('-')
        if parts[0] in AnimoAcid.pg_mwd.keys():
            parts[1] = '-'.join(parts[0:2])
            del parts[0]
        if parts[-1] in AnimoAcid.pg_mwd.keys():
            parts[-2] = '-'.join(parts[-2:])
            del parts[-1]
            
        self.AAs = [AnimoAcid(part) for part in parts]
        
    def __repr__(self) -> str:
        seq = []
        for aa in self.AAs:
            if isinstance(aa, list):
                seq.extend(aa)
            else:
                seq.append(aa)
        return '-'.join([aa.make_pep_repr(is_N_terminal=(i==0),
                                          is_C_terminal=(i==len(seq)-1)) \
                                              for i, aa in enumerate(seq)])
    
    def calcu_mw(self, expand_mw_dict: Dict[str, float] = None):
        if expand_mw_dict is not None:
            assert isinstance(expand_mw_dict, dict), 'expand_mw_dict should be a dict contains protect group molecular weight'
            AnimoAcid.all_mwd.update(expand_mw_dict)
        mw = sum([aa.calcu_mw(expand_mw_dict) for aa in self.AAs])
        mw -= (len(self.AAs) - 1) * 18.02 # because single AA do not has peptide bond, so we need to minus 18.02 for each bond
        return mw
    

def calcu_substitution_value(args):
    """
    Calculates the substitution value and plots a scatter plot with a linear 
    regression model. The function first processes the input arguments to 
    convert the strings to float values and stores them in arrays. It then 
    calculates the average substitution value and prints it on the console. 
    Next, the function fits a linear regression model to the data using the 
    LinearRegression class from scikit-learn and calculates the equation 
    parameters and R-squared value for the fit. Finally, it plots the linear 
    regression line on the scatter plot using matplotlib and displays the 
    equation and R-squared value using text annotations.
    
    该函数使用给定的参数计算取代值，并绘制线性回归模型的散点图。函数首先将参数中的字符串转换为
    浮点数，并将其存储在数组中。然后，计算取代值的平均值，并在控制台上打印。接下来，函数使用线
    性回归模型拟合数据，并计算拟合方程的参数和R平方值。最后，函数在散点图上绘制线性回归线，并
    显示方程和R平方值。
    """
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    a = np.array([float(i) for i in args.absorbance.split(',') if len(i)])
    m = np.array([float(i) for i in args.weight.split(',') if len(i)])
    mean_subval = np.mean(args.coff*a/m)
    print(f'\nAvg Substitution Value: {mean_subval}')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, max(20, m.max()*1.2))
    ax.set_ylim(0, max(0.4, a.max()*1.2))

    regressor = LinearRegression()
    regressor = regressor.fit(m.reshape(-1, 1), a.reshape(-1, 1))
    equation_a, equation_b = regressor.coef_.item(), regressor.intercept_.item()
    equation_r2 = '{:4.3f}'.format(regressor.score(m.reshape(-1, 1), a.reshape(-1, 1)))
    sns.regplot(x = m, y = a, color = 'black', marker = 'o', truncate = False, ax = ax)

    equationStr = f'OD = {equation_a:5.4f} * m {" " if equation_b<0 else "+ ":}{equation_b:5.4f}'
    print(equationStr, '\n', 'R^2 = ', equation_r2)
    plt.text(0.1, 0.1, '$'+equationStr+'$', fontsize=20)
    plt.text(0.1, 0.3, '$R^2 = $'+equation_r2, fontsize=20)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    ax.set_title(f'Avg Substitution Value: {mean_subval:.5f}', fontsize=25)
    ax.set_xlabel('Weight of Resin (mg)', fontsize=25)
    ax.set_ylabel('OD (304 nm)', fontsize=25)
    plt.show()
    
    
@dataclass
class MutationOpts:
    AA_deletion: bool = True # whether delete AA can be performed
    AA_repeat: int = 1 # max repeat times of AA
    N_protect_deletion: bool = True # whether delete N-terminal protect group can be performed
    C_protect_deletion: bool = True # whether delete C-terminal protect group can be performed
    R_protect_deletion: bool = True # whether delete R-terminal protect group can be performed
    
@dataclass
class MutationTree:
    peptide: Peptide # this dot's peptide seqeunce to perform mutate
    pos: int # current AA pos to opt in self.peptide sequence, from N terminal to C terminal.
    # [current repeat pos, sum repeat in this AA in this branch], if second number is 1, means no repeat in this branch
    repeat_pos: List[int] = field(default_factory = lambda: [0, 1])
     # mutate branchs for AA repeat mutation, can be null with AA_repeat set to 0
    repeats: List['MutationTree'] = field(default_factory = list)
    del_AA: 'MutationTree' = None # mutate branch for AA delete mutation
    del_N: 'MutationTree' = None # mutate branch for N protect delete mutation
    del_C: 'MutationTree' = None # mutate branch for C protect delete mutation
    del_R: 'MutationTree' = None # mutate branch for R protect delete mutation
    father: 'MutationTree' = None # father dot
    
def extract_mutations(mutations: MutationTree):
    """extract all sub dots from mutations(Tree)"""
    def flatten_peptide(peptide: Peptide):
        seq = []
        for aa in peptide.AAs:
            if isinstance(aa, list):
                seq.extend(aa)
            else:
                seq.append(aa)
        peptide.AAs = seq
        return peptide
    if mutations.del_AA is None: # because repeat can be set to 0, so there has no repeat mutation
        if mutations.father is not None:
            muts = [flatten_peptide(mut.peptide) for mut in mutations.repeats]
            return muts + [flatten_peptide(getattr(mutations, n).peptide) \
                for n in ['del_AA', 'del_N', 'del_C', 'del_R'] \
                    if getattr(mutations, n) is not None]
        return []
    final_seq = []
    for repeat in mutations.repeats:
        final_seq.extend(extract_mutations(repeat))
    for n in ['del_AA', 'del_N', 'del_C', 'del_R']:
        final_seq.extend(extract_mutations(getattr(mutations, n)))
    return final_seq

def delete_NCR(_seq: Peptide, mutations: MutationTree, _del_X: str, null_pg: str, max_repeat):
    """
    Params:
        - _seq: Peptide, mother peptide.
    """
    pos = mutations.pos
    del_X = getattr(mutations, _del_X)
    # delete N-terminal protect group, move to next NEW AA
    seq = deepcopy(_seq)
    if mutations.repeat_pos[1] == 1 and seq.AAs[pos].N_protect != null_pg:
        seq.AAs[pos].N_protect = 'H'
        del_X = MutationTree(seq, pos+1, [0, 1], father = mutations)
        setattr(mutations, _del_X, del_X)
        mutate_peptide(_seq, del_X, MutationOpts(AA_repeat=max_repeat))
    elif mutations.repeat_pos[1] > 1 and _seq.AAs[pos].N_protect != null_pg:
        seq.AAs[pos][mutations.repeat_pos[0]].N_protect = null_pg
        if mutations.repeat_pos[0] == mutations.repeat_pos[1] - 1:
            # move to next NEW AA
            del_X = MutationTree(seq, pos+1, [0, 1], father = mutations)
            setattr(mutations, _del_X, del_X)
            mutate_peptide(_seq, del_X, MutationOpts(AA_repeat=max_repeat), max_repeat)
        else:
            # move to next repeat AA
            del_X = MutationTree(seq, pos, deepcopy(mutations.repeat_pos), father = mutations)
            del_X.repeat_pos[0] += 1
            setattr(mutations, _del_X, del_X)
            mutate_peptide(_seq, del_X, MutationOpts(AA_repeat=0, AA_deletion=False), max_repeat)

def mutate_peptide(peptide: Peptide, mutations: MutationTree, opts: MutationOpts,
                   max_repeat: int):
    """
    Parameters:
        - peptide: Peptide, mother peptide.
        - mutations: Tree object, store all mutations and there relationship.
        - opts: MutationOpts to perform mutation.
        - max_repeat: int
    """
    pos = mutations.pos
    if pos >= len(peptide.AAs):
        return mutations
    # perform AA repeat mutation AND MAKE A REMAIN ONE BY repeat_idx=0
    for repeat_idx in range(opts.AA_repeat+1): # repeat_idx is in [0, repeat]
        seq = deepcopy(mutations.peptide)
        if repeat_idx == 0:
            # remain, move to next NEW AA
            seq.AAs[pos] = deepcopy(peptide.AAs[pos])
            mutations.repeats.append(MutationTree(seq, pos+1, [0, 1], father = mutations))
            mutate_peptide(peptide, mutations.repeats[-1], MutationOpts(AA_repeat=max_repeat), max_repeat)
        else:
            # repeat, continue in this AA start from first repeat
            seq.AAs[pos] = [deepcopy(peptide.AAs[pos])] * (repeat_idx + 1)
            if pos == 0 and peptide.AAs[pos].N_protect != 'H':
                for aa in seq.AAs[pos][1:]:
                    aa.N_protect = 'H'
            elif pos == len(peptide.AAs)-1 and peptide.AAs[pos].C_protect != 'OH':
                for aa in seq.AAs[pos][:-1]:
                    aa.C_protect = 'OH'
            # mutate each from repeat_idx=0
            for i in range(repeat_idx+1):
                mutations.repeats.append(MutationTree(seq, pos, [0, repeat_idx + 1], father = mutations))
                mutate_peptide(peptide, mutations.repeats[-1], MutationOpts(AA_deletion=False, AA_repeat=0), max_repeat)
    # perform AA delete mutation, this AA CAN NOT be repeated
    if opts.AA_deletion:
        # delete AA(s) in pos via replacing AA by [], move to next NEW AA
        seq = deepcopy(mutations.peptide)
        seq.AAs[pos] = []
        mutations.del_AA = MutationTree(seq, pos+1, [0, 1], father = mutations)
        # NEW AA, NEW MutationOpts
        mutate_peptide(peptide, mutations.del_AA, MutationOpts(AA_repeat=max_repeat), max_repeat)
    # perform N protect deletion, the AA can be repeated
    if opts.N_protect_deletion:
        # delete N-terminal protect group, move to next NEW AA
        delete_NCR(peptide, mutations, 'del_N', 'OH', max_repeat)
    # perform C protect deletion, the AA can be repeated
    if opts.C_protect_deletion:
        # delete C-terminal protect group, move to next NEW AA
        delete_NCR(peptide, mutations, 'del_C', 'OH', max_repeat)
    # perform R protect deletion, the AA can be repeated
    if opts.R_protect_deletion:
        # delete R-terminal protect group, move to next NEW AA
        delete_NCR(peptide, mutations, 'del_R', 'H', max_repeat)

def calcu_mw_of_mutations(args):
    expand_mw_dict = [i.split('-') for i in args.weight.split(',') if len(i) > 2]
    expand_mw_dict = {i[0]:i[1] for i in expand_mw_dict}
    peptide = Peptide(args.seq)
    print(f'\npeptide: {peptide}')
    print(f'MW: {peptide.calcu_mw(expand_mw_dict)}')
    # calcu mutations
    all_mutations = MutationTree(pos = 0, peptide = peptide)
    all_mutations = mutate_peptide(peptide, all_mutations,
                                   MutationOpts(AA_repeat=args.max_repeat),
                                   args.max_repeat)

def calcu_mw(args):
    expand_mw_dict = [i.split('-') for i in args.weight.split(',') if len(i) > 2]
    expand_mw_dict = {i[0]:i[1] for i in expand_mw_dict}
    peptide = Peptide(args.seq)
    print(f'\npeptide: {peptide}')
    print(f'MW: {peptide.calcu_mw(expand_mw_dict)}')


calcu_mw_of_mutations(namedtuple('Args', ['seq','max_repeat', 'weight'],
                                 defaults = ['Fmoc-Cys(Trt)-Leu-Asn(Trt)', 1, ''])())


_str2func = {
    'sb': calcu_substitution_value,
    'subval': calcu_substitution_value,
    'mw': calcu_mw,
    'mmw': calcu_mw_of_mutations,
    'mutationweight': calcu_mw_of_mutations,
}


if __name__ == "__main__":
    args_paser = argparse.ArgumentParser()
    subparsers = args_paser.add_subparsers(title='subcommands', dest='sub_command')
    
    sub_val_args = subparsers.add_parser('subval', aliases = ['sb'], description='calcu SPPS substitution value for a release test of resin.')
    sub_val_args.add_argument('-a', '-A', '--absorbance', '--Absorbance', type = str, help='Absorbance (OD value), input as 0.503,0.533')
    sub_val_args.add_argument('-m', '-w', '--weight', type = str, help='resin wight (mg), input as 0.165,0.155')
    sub_val_args.add_argument('-c', '--coff', default = 16.4, type = float, help='coff, default is 16.4')
    
    molecularnweight = subparsers.add_parser('molecularnweight', aliases = ['mw'], description='calcu MW of peptide.')
    molecularnweight.add_argument('-s', '--seq', '--seqeunce', '--pep', '--peptide', type = str, help='peptide seqeunce, input as Fmoc-Cys(Acm)-Leu-OH or H-Cys(Trt)-Leu-OH')
    molecularnweight.add_argument('-w', '--weight', type = str, default = '', help='MW of peptide AAs and protect group, input as Trt-243.34,Boc-101.13 and do not include weight of -H')
    
    mutationweight = subparsers.add_parser('mutationweight', aliases = ['mmw'], description='calcu MW of each peptide mutations syn by SPPS.')
    mutationweight.add_argument('-s', '--seq', '--seqeunce', '--pep', '--peptide', type = str, help='peptide seqeunce, input as Fmoc-Cys(Acm)-Leu-OH or H-Cys(Trt)-Leu-OH')
    mutationweight.add_argument('-w', '--weight', type = str, default = '', help='MW of peptide AAs and protect group, input as Trt-243.34,Boc-101.13 and do not include weight of -H')
    mutationweight.add_argument('--max-repeat', type = int, default = 1, help='max times for repeat a AA in sequence')
    
    args = args_paser.parse_args()
    
    if args.sub_command in _str2func:
        print(f'excuting command: {args.sub_command}')
        _str2func[args.sub_command](args)
    else:
        base.put_err(f'no such sub commmand: {args.sub_command}')