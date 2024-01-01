import argparse
import os
from copy import deepcopy
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
        'Boc': 101.13, # = 102.13 - 1.0(H)
        'Fmoc': 223.26, # = 224.26 - 1.0(H)
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
        return '-'.join([aa.make_pep_repr(is_N_terminal=(i==0),
                                          is_C_terminal=(i==len(self.AAs)-1)) \
                                              for i, aa in enumerate(self.AAs)])
    
    def calcu_mw(self, expand_mw_dict: Dict[str, float] = None):
        if expand_mw_dict is not None:
            assert isinstance(expand_mw_dict, dict), 'expand_mw_dict should be a dict contains protect group molecular weight'
            AnimoAcid.all_mwd.update(expand_mw_dict)
        mw = sum([aa.calcu_mw(expand_mw_dict) for aa in self.AAs])
        mw -= (len(self.AAs) - 1) * 18.0 # because single AA do not has peptide bond, so we need to minus 18.0 for each bond
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
    
    
def caucu_mw_of_mutations(args):
    peptide = Peptide(args.seq)
    print(f'\npeptide: {peptide}')
    print(f'mw: {peptide.calcu_mw()}')


_str2func = {
    'sb': calcu_substitution_value,
    'subval': calcu_substitution_value,
    'mw': caucu_mw_of_mutations,
    'mmw': caucu_mw_of_mutations,
    'mutationweight': caucu_mw_of_mutations,
}


if __name__ == "__main__":
    args_paser = argparse.ArgumentParser()
    subparsers = args_paser.add_subparsers(title='subcommands', dest='sub_command')
    
    sub_val_args = subparsers.add_parser('subval', aliases = ['sb'], description='calcu SPPS substitution value for a release test of resin.')
    sub_val_args.add_argument('-a', '-A', '--absorbance', '--Absorbance', type = str, help='Absorbance (OD value), input as 0.503,0.533')
    sub_val_args.add_argument('-m', '-w', '--weight', type = str, help='resin wight (mg), input as 0.165,0.155')
    sub_val_args.add_argument('-c', '--coff', default = 16.4, type = float, help='coff, default is 16.4')
    
    sub_val_args = subparsers.add_parser('mutationweight', aliases = ['mw', 'mmw'], description='calcu MW of each peptide mutaitions syn by SPPS.')
    sub_val_args.add_argument('-s', '--seq', '--seqeunce', '--pep', '--peptide', type = str, help='peptide seqeunce, input as Fmoc-Cys(Acm)-Leu-OH or H-Cys(Trt)-Leu-OH')
    sub_val_args.add_argument('-w', '--weight', type = str, default = '', help='MW of peptide AAs and protect group, input as Trt-243.34,Boc-101.13 and do not include weight of -H')
    
    args = args_paser.parse_args()
    
    if args.sub_command in _str2func:
        _str2func[args.sub_command](args)
    else:
        base.put_err(f'no such sub commmand: {args.sub_command}')