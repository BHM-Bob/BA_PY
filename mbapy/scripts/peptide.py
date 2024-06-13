import argparse
import itertools
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['MBAPY_FAST_LOAD'] = 'True'

from mbapy.base import put_err, put_log, split_list
from mbapy.bio.peptide import AnimoAcid, Peptide
from mbapy.file import get_paths_with_extension, get_valid_file_path, opts_file
from mbapy.sci_instrument.mass import MassData, SciexOriData, SciexPeakListData
from mbapy.scripts._script_utils_ import (Command, _print, clean_path,
                                          excute_command, show_args)
from mbapy.scripts.mass import load_single_mass_data_file, plot_mass
from mbapy.web import TaskPool


def calcu_substitution_value(args: argparse.Namespace):
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
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    
    a = np.array([float(i) for i in args.absorbance.split(',') if len(i)])
    m = np.array([float(i) for i in args.weight.split(',') if len(i)])
    mean_subval = np.mean(args.coff*a/m)
    print(f'\nSubstitution Value: {args.coff*a/m}')
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


def calcu_mw(args: argparse.Namespace, _print = print):
    """
    Calculates the molecular weight (MW) of a peptide based on its amino acid sequence and a dictionary of weights for each amino acid.

    Args:
        args (Namespace): An object containing the command line arguments.
        _print (function, optional): A function used for printing. Defaults to the built-in print function.

    Returns:
        tuple: A tuple containing the peptide object and the expanded MW dictionary.

    Example:
        >>> args = Namespace(seq='A-C-D-E', weight='A-71.04,C-103.01,D-115.03,E-129.04', mass=True)
        >>> calcu_mw(args)
        peptide: ACDE
        MW: 418.12
        Chemical Formular: C5H10N2O3
        Exact Mass: 118.07
        (<peptide object>, {'A': '71.04', 'C': '103.01', 'D': '115.03', 'E': '129.04'})
    """
    expand_mw_dict = [i.split('-') for i in args.weight.split(',') if len(i) > 2]
    expand_mw_dict = {i[0]:i[1] for i in expand_mw_dict}
    peptide = Peptide(args.seq)
    _print(f'\npeptide: {peptide}')
    _print(f'MW: {peptide.calcu_mw(expand_mw_dict)}')
    if args.mass:
        _print(f'Chemical Formular: {peptide.get_molecular_formula()}, Exact Mass: {peptide.calcu_mass()}')
    return peptide, expand_mw_dict
    
    
@dataclass
class MutationOpts:
    AA_deletion: bool = True # whether delete AA can be performed
    AA_repeat: int = 1 # AA repeat times of AA
    N_protect_deletion: bool = True # whether delete N-terminal protect group can be performed
    C_protect_deletion: bool = True # whether delete C-terminal protect group can be performed
    R_protect_deletion: bool = True # whether delete R-terminal protect group can be performed
    
    def copy(self):
        return MutationOpts(self.AA_deletion, self.AA_repeat, self.N_protect_deletion,
                            self.C_protect_deletion, self.R_protect_deletion)
    
    def check_empty(self, _pos: List[int], seq: Peptide, args: argparse.Namespace):
        """
        return list of signals which is able to opt, if empty, the lis is also empty.
        """
        pos, repeat_pos, sum_repeat = _pos
        able = []
        if pos >= len(seq.AAs):
            return []
        if sum_repeat == 1:
            if self.AA_deletion and not args.disable_aa_deletion:
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
        tree.mutate.seq.AAs[pos] = [tree.mutate.seq.AAs[pos].copy() \
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
        tree.mutate.opts = MutationOpts(AA_repeat=0)
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
        if sum_repeat == 1 and getattr(tree.mutate.seq.AAs[pos], f'{NCR}_protect') != null_pg:
            setattr(tree.mutate.seq.AAs[pos], f'{NCR}_protect', null_pg)
        elif sum_repeat > 1 and getattr(tree.mutate.seq.AAs[pos][repeat_pos], f'{NCR}_protect') != null_pg:
            setattr(tree.mutate.seq.AAs[pos][repeat_pos], f'{NCR}_protect', null_pg)
        # trun off the opts in two branches
        setattr(tree.mutate.opts, f'{NCR}_protect_deletion', False)
        setattr(tree.remain.opts, f'{NCR}_protect_deletion', False)
        return tree
                
    def perform_one(self, tree: 'MutationTree', args: argparse.Namespace):
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
        able = tree.opts.check_empty(tree.pos, tree.seq, args)
        if able:
            # generate two branch and set seq to None to free memory
            tree.generate_two_branch()
            # perform mutation
            if 'AA_deletion' in able:
                tree = self.delete_AA(tree, args.max_repeat)
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
    peptide: Peptide # mother peptide, READ ONLY, remians unchanged
    seq: Peptide # this dot's peptide seqeunce to perform mutate
    opts: MutationOpts # opts left to perform
    # [current AA pos, current repeat pos, sum repeat in this AA in seq], if last number is 1, means no repeat in this AA
    pos: List[int] = field(default_factory = lambda: [0, 0, 1])
    father: 'MutationTree' = None # father dot
    remain: 'MutationTree' = None # father dot
    mutate: 'MutationTree' = None # father dot
    
    def copy(self, copy_peptide: bool = False, copy_branch: bool = False,
             father: 'MutationTree' = None, remain: 'MutationTree' = None,
             mutate: 'MutationTree' = None):
        """
        Params:
            - copy_peptide: bool, whether to copy mother peptide.
            - copy_branch: bool, whether to copy father, remain, mutate branch via deepcopy. If False, leave it None.
        """
        if copy_peptide:
            cp = MutationTree(self.peptide.copy(), self.seq.copy(), self.opts.copy(), [i for i in self.pos])
        else:
            cp = MutationTree(self.peptide, self.seq.copy(), self.opts.copy(), [i for i in self.pos])
        if copy_branch:
            cp.father = deepcopy(self.father)
            cp.remain = deepcopy(self.remain)
            cp.mutate = deepcopy(self.mutate)
        else:
            cp.father = father
            cp.remain = remain
            cp.mutate = mutate
        return cp
    
    def extract_mutations(self, flatten: bool = True):
        """
        extract all terminal dots from mutations(Tree)
            - flatten==True:  will CHANGE it's peptide.AAs, return the flattened peptide.
            - flatten==False: will simply return all leaves of MutationTree.
        """
        if self.mutate is None and self.remain is None:
            if flatten:
                self.seq.flatten(inplace=True)
                return [self.seq]
            else:
                return [self]
        else:
            final_seq = []
            final_seq.extend(self.remain.extract_mutations(flatten))
            final_seq.extend(self.mutate.extract_mutations(flatten))
            return final_seq
    
    def check_is_end_pos(self):
        """check if current AA is the last AA whether in repeat or mother peptide"""
        if self.pos[0] >= len(self.peptide.AAs) - 1 and self.pos[1] >= self.pos[2] - 1:
            return True
        return False
        
    def generate_two_branch(self):
        """Generate two branch with all None remian and mutate branch, return itself."""
        self.remain = self.copy(father=self)
        self.mutate = self.copy(father=self)
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

def calcu_mutations_mw_batch(seqs: List[Peptide], mass: bool = False, verbose: bool = True):
    peps, mw2pep = {}, {}
    for pep in tqdm(seqs,
                    desc='Gathering mutations and Calculating molecular weight',
                    disable=not verbose):
        full_pep = Peptide(None)
        full_pep.AAs.extend(aa.AAs for aa in pep)
        full_pep.flatten(inplace = True)
        if len(full_pep.AAs):
            pep_repr = str(full_pep)
            if pep_repr not in peps:
                peps[pep_repr] = len(peps)
                if mass:
                    mw = full_pep.calcu_mass()
                else:
                    mw = full_pep.calcu_mw()
                if mw in mw2pep:
                    mw2pep[mw].append(full_pep)
                else:
                    mw2pep[mw] = [full_pep]
    return peps, mw2pep


class mutation_weight(Command):
    """
    Calculates the molecular weight of mutations based on the given arguments.

    Args:
        args (object): An object that contains the following attributes:
            - seq (str): The input sequence.
            - weight (bool): If True, the weight of the peptide will be calculated instead of the mass.
            - max_repeat (int): The maximum number of repeat amino acids allowed in the peptide.
            - out (str): The output file path. If None, the output will be printed to the console.
            - mass (bool): If True, the mass of the peptide will be calculated instead of the weight.

    Prints the molecular weight of the mutations and the corresponding peptide sequences.

    The function first sets up a helper function, `_print`, for printing information to the console and/or a file.
    It then processes the `args.out` attribute to obtain a valid file path if it is a directory.
    Next, it opens the output file for writing if `args.out` is not None, otherwise it sets `f` to None.
    The function then prints the values of the input arguments using `_print`.
    After that, it calls the `calcu_mw` function to calculate the molecular weight of the peptide and obtain a dictionary of expanded molecular weights.
    Following that, it creates a `MutationTree` object to hold the peptide and its mutations.
    It then mutates the peptide according to the maximum repeat allowed.
    Next, it extracts the individual mutations from the `all_mutations` object.
    The function then initializes dictionaries to store the molecular weight to peptide mapping and the unique peptide sequences.
    It iterates over each individual mutation and calculates its molecular weight.
    If the molecular weight is already present in `mw2pep`, the mutation is appended to the list of peptides with the same molecular weight.
    Otherwise, a new entry is created in `mw2pep` with the molecular weight as the key and the mutation as the value.
    Finally, the function prints the number of mutations found and the details of each mutation, along with their respective indices.
    If an output file was specified, it is closed at the end.
    """
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        self.verbose = True
        
    @staticmethod
    def make_args(args: argparse.ArgumentParser):
        args.add_argument('-s', '--seq', '--seqeunce', '--pep', '--peptide', type = str,
                        help='peptide seqeunce, input as Fmoc-Cys(Acm)-Leu-OH or H-Cys(Trt)-Leu-OH')
        args.add_argument('-w', '--weight', type = str, default = '',
                        help='MW of peptide AAs and protect group, input as Trt-243.34,Boc-101.13 and do not include weight of -H')
        args.add_argument('--max-repeat', type = int, default = 1,
                        help='max times for repeat a AA in sequence')
        args.add_argument('--disable-aa-deletion', action='store_true', default=False,
                        help='disable AA deletion in mutations.')
        args.add_argument('-o', '--out', type = str, default = None,
                        help='save results to output file/dir. Defaults None, do not save.')
        args.add_argument('-m', '--mass', action='store_true', default=False,
                        help='calcu Exact Mass instead of Molecular Weight.')
        args.add_argument('--disable-verbose', action='store_true', default=False,
                        help='disable verbose output to console.')
        args.add_argument('--multi-process', type = int, default = 1,
                        help='number of multi-process to use. Defaults 1, no multi-process.')
        args.add_argument('--batch-size', type = int, default = 500000,
                        help='number of peptides to process in each batch. Defaults %(default)s in a batch.')
        return args
    
    def process_args(self):
        if self.args.out is not None:
            self.args.out = clean_path(self.args.out)
            if os.path.isdir(self.args.out):
                file_name = get_valid_file_path(" ".join(sys.argv[1:]))+'.txt'
                self.args.out = os.path.join(self.args.out, file_name)
            self.f = open(self.args.out, 'w')
        else:
            self.f = None
        self.printf = partial(_print, f = self.f)
        self.verbose = not self.args.disable_verbose
        
    @staticmethod
    def mutate_peptide(tree: MutationTree, args: argparse.Namespace):
        """
        Parameters:
            - mutations: Tree object, store all mutations and there relationship.
            - max_repeat: int
        """
        # perofrm ONE mutation
        tree = tree.opts.perform_one(tree, args)
        # if NO mutaion can be done, 
        if tree.mutate is None and tree.remain is None:
            # try move current AA in this tree to next AA
            if tree.move_to_next(args.max_repeat):
                # move success, go on
                mutation_weight.mutate_peptide(tree, args)
            else:
                # it is the end, return tree
                return tree
        else: # go on with two branches
            mutation_weight.mutate_peptide(tree.mutate, args)
            mutation_weight.mutate_peptide(tree.remain, args)
        return tree
    
    @staticmethod
    def generate_mutate_peps(peptide: Peptide, args: argparse.Namespace):
        """
        gernerate all possible mutations of a peptide.
        
        Parameters:
            - peptide: Peptide object, the mother peptide.
            - args: argparse.Namespace, the input arguments, must contain 'max_repeat', 'disable_aa_deletion', 
        
        Returns:
            - all_mutations: list of Peptide objects, all possible mutations of the mother peptide.
        """
        seq = []
        for aa in tqdm(peptide.AAs, desc='Mutating peptide'):
            pep = Peptide(None)
            pep.AAs = [aa.copy()]
            aa_mutations = MutationTree(peptide=pep, seq=pep.copy(),
                                        opts=MutationOpts(AA_repeat=args.max_repeat),
                                        pos=[0, 0, 1])
            aa_mutations = mutation_weight.mutate_peptide(aa_mutations, args)
            seq.append(aa_mutations.extract_mutations())
        return list(itertools.product(*seq))
    
    @staticmethod
    def calcu_mutations_mw(seqs: List[Peptide], mass: bool = False,
                           multi_process: int = 1, batch_size: int = 500000):
        if multi_process == 1:
            return calcu_mutations_mw_batch(seqs, mass=mass, verbose=True)
        else:
            print('Gathering mutations and Calculating molecular weight...')
            peps, mw2pep = {}, {}
            pool = TaskPool('process', multi_process)
            for i, batch in enumerate(split_list(seqs, batch_size)):
                pool.add_task(f'{i}', calcu_mutations_mw_batch, batch, mass, False)
            pool.run()
            pool.wait_till(lambda : pool.count_done_tasks() == len(pool.tasks), verbose=True)
            for (_, (peps_i, mw2pep_i), _) in pool.tasks.values():
                peps.update(peps_i)
                for i in mw2pep_i:
                    if i in mw2pep:
                        mw2pep[i].extend(mw2pep_i[i])
                    else:
                        mw2pep[i] = mw2pep_i[i]
            return peps, mw2pep
        
    def main_process(self):
        # show mother peptide info
        peptide, expand_mw_dict = calcu_mw(self.args, _print = self.printf)
        # calcu mutations
        seqs = mutation_weight.generate_mutate_peps(peptide, self.args)
        # gather mutations, calcu mw and store in dict
        peps, mw2pep = self.calcu_mutations_mw(seqs, self.args.mass, self.args.multi_process, self.args.batch_size)
        # output info
        self.printf(f'\n{len(peps)-1} mutations found, followings include one original peptide seqeunce:\n')
        if self.verbose:
            idx, weigth_type = 0, 'Exact Mass' if self.args.mass else 'MW'
            for i, mw in enumerate(sorted(mw2pep)):
                self.printf(f'\n{weigth_type}: {mw:10.5f}', verbose = self.verbose)
                for j, pep in enumerate(mw2pep[mw]):
                    mf = f'({pep.get_molecular_formula()})' if self.args.mass else ''
                    self.printf(f'    pep-{i:>4}-{j:<4}({idx:8d})({len(pep.AAs)} AA){mf}: {pep}', verbose = self.verbose)
                    idx += 1
        # handle f-print
        if self.f is not None:
            self.f.close()
            # save mw2pep and peps
            opts_file(str(self.args.out)+'.mbapy.mmw.pkl', 'wb', data = {'mw2pep':mw2pep, 'peps':peps}, way = 'pkl')
        

class cycmmw(Command):
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        

class fit_mass(Command):
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        self.task_pool: TaskPool = None
        self.mw2pep: Dict[int, List[Peptide]] = {}
        self.mass_dfs: Dict[str, MassData] = None
        
    @staticmethod
    def make_args(args: argparse.ArgumentParser):
        args.add_argument('-s', '--seq', type = str, default='.',
                          help='input file/dir or peptide 3 letter seqeunce, default is %(default)s')
        args.add_argument('-m', '--mass-file', type=str, default='.',
                          help='input file/dir or peptide mass file, default is %(default)s')
        args.add_argument('-sys', '--mass-system', type=str, choices=list(plot_mass.SUPPORT_SYS.keys())+['ALL'], default='ALL',
                          help='mass system, default is %(default)s')
        args.add_argument('-r', '--recursive', action='store_true', default=False,
                          help='recursive search for seq file and mass files in dir, default is %(default)s')
        args.add_argument('-o', '--output', type=str, default='.',
                          help='output dir, default is %(default)s')
        args.add_argument('--max-repeat', type = int, default = 1,
                          help='max times for repeat a AA in sequence')
        args.add_argument('--disable-aa-deletion', action='store_true', default=False,
                          help='disable AA deletion in mutations.')
        args.add_argument('--multi-process', type = int, default = 4,
                          help='number of multi-process to use. Defaults 1, no multi-process.')
        args.add_argument('--batch-size', type = int, default = 500000,
                          help='number of peptides to process in each batch. Defaults %(default)s in a batch.')
        args.add_argument('-eps', '--error-tolerance', type = float, default = 0.1,
                          help='error tolerance for fitting mass data, default is %(default)s.')
        args.add_argument('-min', '--min-height', type = int, default=0,
                          help='filter data with min height in peak list plot, default is %(default)s')
        args.add_argument('-minp', '--min-height-percent', type = float, default=1,
                          help='filter data with min height percent to hightest in mass/charge plot, default is %(default)s')
        args.add_argument('--min-peak-width', type = float, default=4,
                          help='filter peaks with min width in Mass/Charge plot, default is %(default)s')
        args.add_argument('-xlim', type = str, default=None,
                          help='x-axis data limit for pre-filter of mass data, input as "200,2000", default is %(default)s')
        args.add_argument('--ms-lim', type = str, default=None,
                          help='mass limit for second-filter of transfered mass data, input as "200,2000", default is %(default)s')
        return args
        
    def process_args(self):
        show_args(self.args, list(self.args.__dict__.keys()), self.printf)
        self.printf('processing arguments...\n')
        self.args.xlim = eval(f'({self.args.xlim})') if self.args.xlim is not None else None
        self.args.ms_lim = eval(f'({self.args.ms_lim})') if self.args.ms_lim is not None else None
        # set task pool
        if self.args.multi_process > 1:
            self.task_pool = TaskPool('process', self.args.multi_process).run()
            self.printf(f'task pool created with {self.args.multi_process} processes')
        # process argument: seq
        if os.path.isfile(clean_path(self.args.seq)) and self.args.seq.endswith('mbapy.mmw.pkl'):
            self.mw2pep = opts_file(self.args.seq, mode = 'rb', way='pkl')['mw2pep']
            self.printf(f'load mw2pep from {self.args.seq}')
        elif os.path.isdir(clean_path(self.args.seq)):
            for path in get_paths_with_extension(self.args.seq, ['mbapy.mmw.pkl'], self.args.recursive):
                mw2pep = opts_file(path, mode = 'rb', way='pkl')['mw2pep']
                self.printf(f'load mw2pep from {path}')
                for i in mw2pep:
                    if i in self.mw2pep:
                        self.mw2pep[i].extend(mw2pep[i])
                    else:
                        self.mw2pep[i] = mw2pep[i]
            if not self.mw2pep:
                return put_err(f'no valid peptide found in {self.args.seq}, return None')
        else:
            seq = Peptide(self.args.seq)
            # calcu mutations
            seqs = mutation_weight.generate_mutate_peps(seq, self.args)
            # gather mutations, calcu mw and store in dict
            _, self.mw2pep = mutation_weight.calcu_mutations_mw(seqs, self.args.mass, self.args.multi_process, self.args.batch_size)
        self.printf(f'{len(self.mw2pep)} peptides loaded')
        # process argument: mass_file
        mass_manager = plot_mass(self.args)
        if os.path.isfile(self.args.mass_file):
            mass_data = load_single_mass_data_file(self.args.mass_file, set(), plot_mass.SUPPORT_SYS)
            if not mass_data:
                put_err(f'failed to load mass data from {self.args.mass_file}, support systems: {list(plot_mass.SUPPORT_SYS.keys())}')
                exit()
            self.mass_dfs = {mass_data.get_tag(): mass_data}
        elif os.path.isdir(self.args.mass_file):
            self.mass_dfs = mass_manager.load_data(self.args.mass_file, self.args.recursive)
        else:
            return put_err(f'error: mass_file {self.args.mass_file} not found')
        # set output file
        self.args.output = clean_path(self.args.output)

    def main_process(self):
        candidates = np.array(list(self.mw2pep.keys()))
        for n, mass_df in self.mass_dfs.items():
            print(f'fitting {n} now...')
            # make peaks df
            if mass_df.peak_df is None or mass_df.check_processed_data_empty(mass_df.peak_df):
                mass_df.search_peaks(self.args.xlim, self.args.min_peak_width, self.task_pool, self.args.multi_process)
            mass_df.filter_peaks(self.args.xlim, self.args.min_height, self.args.min_height_percent)
            # set charge column
            if mass_df.CHARGE_HEADER is None:
                put_log(f'{n} has no charge header, assuming charge 1')
                charges = [1]*len(mass_df.peak_df)
            else:
                charges = mass_df.peak_df[mass_df.CHARGE_HEADER].values
            # match and set match column
            for i, (ms, charge) in enumerate(zip(mass_df.peak_df[mass_df.X_HEADER], charges)):
                for mode, iron in MassData.ESI_IRON_MODE.items():
                    transfered_ms = (ms*charge-iron['im'])/iron['m']
                    if self.args.ms_lim is None or (transfered_ms > self.args.ms_lim[0] and transfered_ms < self.args.ms_lim[1]):
                        matched = np.where(np.abs(candidates - transfered_ms) < self.args.error_tolerance)[0]
                        if matched.size > 0:
                            all_matched_peps = [(pep, candidates[match_i]) for match_i in matched for pep in self.mw2pep[candidates[match_i]]]
                            mass_df.peak_df.loc[i, f'match {mode}'] = ' | '.join(pep[0].repr() for pep in all_matched_peps)
                            self.printf(f'matched {len(all_matched_peps)} peptide(s) with {mode} at {ms:.4f} (transfered: {transfered_ms:.4f})')
                            for i, (pep, pep_mass) in enumerate(all_matched_peps):
                                self.printf(f'{i}: ({len(pep.AAs)} AA)[{pep_mass:.4f}]{pep.get_molecular_formula()}: {pep}')
                            self.printf('\n\n')
            # save result
            mass_df.peak_df.to_excel(os.path.join(self.args.output, f'{n}.mbapy.fit-mass.xlsx'))

        
def transfer_letters(args):
    # show args
    show_args(args, ['seq', 'src', 'trg', 'dpg', 'ddash', 'input', 'out'])
    # get input
    if args.input is not None:
        path = clean_path(args.input)
        peps = []
        for line in opts_file(path, way='lines'):
            try:
                peps.append(Peptide(line, args.src))
            except:
                put_err(f'error when parsing: {line}, skip')
    else:
        peps = [Peptide(args.seq, args.src)]
    # make output
    reprs = [pep.repr(args.trg, not args.dpg, not args.ddash) for pep in peps]
    if args.out is not None:
        from mbapy.file import opts_file
        path = clean_path(args.output)
        opts_file(path, 'w', data = '\n'.join(reprs))
    [print(r) for r in reprs]
    return reprs


_str2func = {
    'sb': calcu_substitution_value,
    'subval': calcu_substitution_value,
    
    'mw': calcu_mw,
    'molecularweight': calcu_mw,
    'molecular-weight': calcu_mw,
    
    'mmw': mutation_weight,
    'mutationweight': mutation_weight,
    'mutation-weight': mutation_weight,
    
    # 'cycmmw': None,
    # 'cyclic-mutation-weight': None,
    
    'fit-mass': fit_mass,
    
    'letters': transfer_letters,
    'transfer-letters': transfer_letters,
}


def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser()
    subparsers = args_paser.add_subparsers(title='subcommands', dest='sub_command')
    
    sub_val_args = subparsers.add_parser('subval', aliases = ['sb'], description='calcu SPPS substitution value for a release test of resin.')
    sub_val_args.add_argument('-a', '-A', '--absorbance', '--Absorbance', type = str,
                              help='Absorbance (OD value), input as 0.503,0.533')
    sub_val_args.add_argument('-m', '-w', '--weight', type = str,
                              help='resin wight (mg), input as 0.165,0.155')
    sub_val_args.add_argument('-c', '--coff', default = 16.4, type = float,
                              help='coff, default is 16.4')
    
    molecularweight = subparsers.add_parser('molecularweight', aliases = ['molecular-weight', 'mw'], description='calcu MW of peptide.')
    molecularweight.add_argument('-s', '--seq', '--seqeunce', '--pep', '--peptide', type = str,
                                 help='peptide seqeunce, input as Fmoc-Cys(Acm)-Leu-OH or H-Cys(Trt)-Leu-OH')
    molecularweight.add_argument('-w', '--weight', type = str, default = '',
                                 help='MW of peptide AAs and protect group, input as Trt-243.34,Boc-101.13 and do not include weight of -H')
    molecularweight.add_argument('-m', '--mass', action='store_true', default=False,
                                 help='calcu Exact Mass instead of Molecular Weight.')
    
    mutationweight_args = mutation_weight.make_args(subparsers.add_parser('mutationweight', aliases = ['mutation-weight', 'mmw'], description='calcu MW of each peptide mutations syn by SPPS.'))
    
    fit_mass_args = fit_mass.make_args(subparsers.add_parser('fit-mass', description='fit peptide mutation and mass iron exact mass for mass data'))
    
    letters = subparsers.add_parser('letters', aliases = ['transfer-letters'], description='transfer AnimoAcid repr letters width.')
    letters.add_argument('-s', '--seq', '--seqeunce', '--pep', '--peptide', type = str, default='',
                                help='peptide seqeunce, input as Fmoc-Cys(Acm)-Leu-OH or ABC(Trt)DE')
    letters.add_argument('--src', '--source-width', type = int, choices=[1, 3], default = 3,
                                help='source repr width of AnimoAcid, only accept 1 and 3.')
    letters.add_argument('--trg', '--target-width', type = int, choices=[1, 3], default = 1,
                                help='traget repr width of AnimoAcid, only accept 1 and 3.')
    letters.add_argument('--dpg', '--disable-pg', action='store_true', default = False,
                                help='whether to include protect groups in target repr.')
    letters.add_argument('--ddash', '--disable-dash', action='store_true', default = False,
                                help='whether to include dash line in target repr.')
    letters.add_argument('-i', '--input', type = str, default = None,
                                help='input file where peptide seq exists in each line. Defaults None, do not save.')
    letters.add_argument('-o', '--out', type = str, default = None,
                                help='save results to output file/dir. Defaults None, do not save.')
    
    if __name__ in ['__main__', 'mbapy.scripts.peptide']:
        excute_command(args_paser, sys_args, _str2func)

if __name__ in {"__main__", "__mp_main__"}:
    # dev code. MUST BE COMMENTED OUT WHEN PUBLISHING
    # main('fit-mass -s data_tmp/scripts/peptide/C.mbapy.mmw.pkl -m data_tmp/scripts/mass/pl.txt'.split())
    
    # release code
    main()