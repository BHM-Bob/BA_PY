import argparse
import os
import sys

from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
from mbapy import base, plot, file


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
        
    def flatten(self, inplace: bool = False):
        """
        Params:
            - inplace: bool(False), if True, make inplace change and return self, else return changed seq only
        """
        seq = []
        for aa in self.AAs:
            if isinstance(aa, list):
                seq.extend(aa)
            else:
                seq.append(aa)
        if inplace:
            self.AAs = seq
            return self
        else:
            return seq
        
    def __repr__(self) -> str:
        seq = self.flatten()
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


def calcu_mw(args, _print = print):
    expand_mw_dict = [i.split('-') for i in args.weight.split(',') if len(i) > 2]
    expand_mw_dict = {i[0]:i[1] for i in expand_mw_dict}
    peptide = Peptide(args.seq)
    _print(f'\npeptide: {peptide}')
    _print(f'MW: {peptide.calcu_mw(expand_mw_dict)}')
    return peptide, expand_mw_dict
    
    
@dataclass
class MutationOpts:
    AA_deletion: bool = True # whether delete AA can be performed
    AA_repeat: int = 1 # AA repeat times of AA
    N_protect_deletion: bool = True # whether delete N-terminal protect group can be performed
    C_protect_deletion: bool = True # whether delete C-terminal protect group can be performed
    R_protect_deletion: bool = True # whether delete R-terminal protect group can be performed
    
    def check_empty(self, _pos: List[int], seq: Peptide):
        """
        return list of signals which is able to opt, if empty, the lis is also empty.
        """
        pos, repeat_pos, sum_repeat = _pos
        able = []
        if pos >= len(seq.AAs):
            return []
        if sum_repeat == 1:
            if self.AA_deletion:
                able.append('AA_deletion')
            if self.AA_repeat > 0:
                able.append('AA_repeat')
            if seq.AAs[pos].N_protect != 'H' and self.N_protect_deletion:
                able.append('N_protect_deletion')
            if seq.AAs[pos].C_protect != 'OH' and self.C_protect_deletion:
                able.append('C_protect_deletion')
            if seq.AAs[pos].R_protect != 'H' and self.R_protect_deletion:
                able.append('R_protect_deletion')
        else:
            if seq.AAs[pos][repeat_pos].N_protect != 'H' and self.N_protect_deletion:
                able.append('N_protect_deletion')
            if seq.AAs[pos][repeat_pos].C_protect != 'OH' and self.C_protect_deletion:
                able.append('C_protect_deletion')
            if seq.AAs[pos][repeat_pos].R_protect != 'H' and self.R_protect_deletion:
                able.append('R_protect_deletion')
        return able
                
    def delete_AA(self, tree: 'MutationTree', max_repeat: int):
        """
        perform delete_AA mutation in tree.mutate branch, trun off the tree.branches.opt.AA_deletion.
            - THE AA CAN NOT BE REPEATED.
            
        Params:
            - tree: MutationTree, tree to opt.
        """
        pos, repeat_pos, sum_repeat = tree.pos
        # perform delete_AA mutation in tree.mutate branch
        tree.mutate.seq.AAs[pos] = []
        #  mutate branch MOVE TO NEXT NEW AA
        tree.mutate.pos[0] += 1
        tree.mutate.opts = MutationOpts(AA_repeat=max_repeat)
        # trun off the delete AA opt in remain branch
        tree.remain.opts.AA_deletion = False
        return tree
    
    def repeat_AA(self, tree: 'MutationTree'):
        """
        perform delete_AA mutation in tree.mutate branch, trun off the tree.branches.opt.AA_deletion.
            - THE AA CAN NOT BE REPEATED.
            
        Params:
            - tree: MutationTree, tree to opt.
        """
        pos, repeat_pos, sum_repeat = tree.pos
        # perform repeat_AA mutation in tree.mutate branch
        tree.mutate.seq.AAs[pos] = [deepcopy(tree.mutate.seq.AAs[pos]) \
            for _ in range(tree.opts.AA_repeat + 1)]
        # change repeated AAs' N/C protect group if needed
        if pos == 0 and tree.peptide.AAs[pos].N_protect != 'H':
            for aa in tree.mutate.seq.AAs[pos][1:]:
                aa.N_protect = 'H'
        elif pos == len(tree.peptide.AAs)-1 and tree.peptide.AAs[pos].C_protect != 'OH':
            for aa in tree.mutate.seq.AAs[pos][:-1]:
                aa.C_protect = 'OH'
        # change mutate branch 's pos 's sum_repeat to tree.opts.repeat_AA + 1
        tree.mutate.pos[2] = tree.opts.AA_repeat + 1
        # trun off the repeat AA opts in mutate branches
        tree.mutate.opts.AA_repeat = 0
        # decrease the repeat AA opts in remain branches
        tree.remain.opts.AA_repeat -= 1
        return tree
        

    def delete_NCR(self, tree: 'MutationTree', NCR: str):
        """
        perform delete_NCR mutation in tree.mutate branch, trun off the tree.branches.opt.X_protect_deletion
        Params:
            - tree: MutationTree, tree to opt.
            - NCR: str, N or C or R.
        """
        pos, repeat_pos, sum_repeat = tree.pos
        null_pg = 'H' if NCR in ['N', 'R'] else 'OH'
        # delete X-terminal protect group
        if sum_repeat == 1 and getattr(tree.seq.AAs[pos], f'{NCR}_protect') != null_pg:
            setattr(tree.mutate.seq.AAs[pos], f'{NCR}_protect', null_pg)
        elif sum_repeat > 1 and getattr(tree.seq.AAs[pos][repeat_pos], f'{NCR}_protect') != null_pg:
            setattr(tree.mutate.seq.AAs[pos][repeat_pos], f'{NCR}_protect', null_pg)
        # trun off the opts in two branches
        setattr(tree.mutate.opts, f'{NCR}_protect_deletion', False)
        setattr(tree.remain.opts, f'{NCR}_protect_deletion', False)
        return tree
                
    def perform_one(self, tree: 'MutationTree', max_repeat: int):
        """
        Perform ONE mutation opt left in tree.opts, return this tree. Also check if it is a repeated AA.
        If it is a repeated AA depend on tree.pos[2], skip AA deletion and AA repeat.
            - If no opts left to do:
                - let two brance still be None.
                - return the tree.
            - IF HAS:
                - generate two branch, change branches' father
                - perform mutation in mutate branch
                - trun off opts in two branches
                - move pos ONLY in mutate branch.
                - DO NOT CHECK IF MEETS END in both this dot and two branches.
                - return the tree.
        """
        able = tree.opts.check_empty(tree.pos, tree.seq)
        if able:
            # generate two branch
            tree.generate_two_branch()
            # perform mutation
            if 'AA_deletion' in able:
                tree = self.delete_AA(tree, max_repeat)
            elif 'AA_repeat' in able:
                tree = self.repeat_AA(tree)
            elif 'N_protect_deletion' in able:
                tree = self.delete_NCR(tree, 'N')
            elif 'C_protect_deletion' in able:
                tree = self.delete_NCR(tree, 'C')
            elif 'R_protect_deletion' in able:
                tree = self.delete_NCR(tree, 'R')
            else:
                raise ValueError('error when check empty with MutationOpts')
        # return tree
        return tree
    
@dataclass
class MutationTree:
    peptide: Peptide # mother peptide and remians unchanged
    seq: Peptide # this dot's peptide seqeunce to perform mutate
    opts: MutationOpts # opts left to perform
    # [current AA pos, current repeat pos, sum repeat in this AA in seq], if last number is 1, means no repeat in this AA
    pos: List[int] = field(default_factory = lambda: [0, 0, 1])
    father: 'MutationTree' = None # father dot
    remain: 'MutationTree' = None # father dot
    mutate: 'MutationTree' = None # father dot
    
    def extract_mutations(self):
        """
        extract all terminal dots from mutations(Tree)
            - will CHANGE it's peptide.AAs
        """
        if self.mutate is None and self.remain is None:
            return [self.seq.flatten(inplace=True)]
        else:
            final_seq = []
            final_seq.extend(self.remain.extract_mutations())
            final_seq.extend(self.mutate.extract_mutations())
            return final_seq
    
    def check_is_end_pos(self):
        """check if current AA is the last AA whether in repeat or mother peptide"""
        if self.pos[0] >= len(self.peptide.AAs) - 1 and self.pos[1] >= self.pos[2] - 1:
            return True
        return False
        
    def generate_two_branch(self):
        """Generate two branch with all None remian and mutate branch, return itself."""
        self.remain = deepcopy(self)
        self.remain.father = self
        self.remain.remain = self.remain.mutate = None
        self.mutate = deepcopy(self)
        self.mutate.father = self
        self.mutate.remain = self.mutate.mutate = None
        return self
        
    def move_to_next(self, max_repeat: int):
        """
        move current AA pos to next repeat AA or next NEW AA
            - return True is moved, else False when is end."""
        if not self.check_is_end_pos():
            if self.pos[1] == self.pos[2] - 1:
                # repeat idx meets end or do not have a repeat, move to next NEW AA
                self.pos[0] += 1
                self.pos[1] = 0
                self.pos[2] = 1
            else:
                # move to next repeat AA
                self.pos[1] += 1
            # reset opts to full
            self.opts = MutationOpts(AA_repeat = max_repeat)
            return True
        return False
        
def mutate_peptide(tree: MutationTree, max_repeat: int):
    """
    Parameters:
        - mutations: Tree object, store all mutations and there relationship.
        - max_repeat: int
    """
    # perofrm ONE mutation
    tree = tree.opts.perform_one(tree, max_repeat)
    # if NO mutaion can be done, 
    if tree.mutate is None and tree.remain is None:
        # try move current AA in this tree to next AA
        if tree.move_to_next(max_repeat):
            # move success, go on
            mutate_peptide(tree, max_repeat)
        else:
            # it is the end, return tree
            return tree
    else: # go on with two branches
        mutate_peptide(tree.mutate, max_repeat)
        mutate_peptide(tree.remain, max_repeat)
    return tree

def calcu_mw_of_mutations(args):
    # set _print
    def _print(content: str, f):
        if f is not None:
            f.write(content+'\n')
        print(content)
    if args.out is not None:
        args.out = args.out.replace('"', '').replace('\'', '')
        if os.path.isdir(args.out):
            file_name = file.get_valid_file_path(" ".join(sys.argv[1:]))+'.txt'
            args.out = os.path.join(args.out, file_name)
        f = open(args.out, 'w')
    else:
        f = None
    # show mother peptide info
    peptide, expand_mw_dict = calcu_mw(args, _print = lambda x : _print(x, f))
    # calcu mutations
    all_mutations = MutationTree(peptide=peptide, seq=deepcopy(peptide),
                                 opts=MutationOpts(AA_repeat=args.max_repeat),
                                 pos=[0, 0, 1])
    all_mutations = mutate_peptide(all_mutations, args.max_repeat)
    all_mutations = all_mutations.extract_mutations()
    mw2pep = {}
    for mutation in all_mutations:
        if len(mutation.AAs):
            mw = mutation.calcu_mw()
            if mw in mw2pep:
                mw2pep[mw].append(mutation)
            else:
                mw2pep[mw] = [mutation]
    # output info
    _print(f'\n{len(all_mutations)-1} mutations found, followings include one original peptide seqeunce:\n', f)
    idx = 0
    for i, mw in enumerate(sorted(mw2pep)):
        _print(f'\nMW: {mw:10.5}', f)
        for j, pep in enumerate(mw2pep[mw]):
            _print(f'    pep-{i:>4}-{j:<4}({idx:8d}): {pep}', f)
            idx += 1
    # handle f-print
    if f is not None:
        f.close()


_str2func = {
    'sb': calcu_substitution_value,
    'subval': calcu_substitution_value,
    'mw': calcu_mw,
    'mmw': calcu_mw_of_mutations,
    'mutationweight': calcu_mw_of_mutations,
}


# if __name__ == '__main__':
#     # dev code
#     from mbapy.game import BaseInfo
#     calcu_mw_of_mutations(BaseInfo(seq = 'Boc-Asn(Trt)-Asp(OtBu)-Glu(OtBu)',
#                                    max_repeat = 0, weight = ''))


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
    mutationweight.add_argument('-o', '--out', type = str, default = None, help='save results to output file/dir. Defaults None, do not save.')
    
    args = args_paser.parse_args()
    
    if args.sub_command in _str2func:
        print(f'excuting command: {args.sub_command}')
        _str2func[args.sub_command](args)
    else:
        base.put_err(f'no such sub commmand: {args.sub_command}')