import argparse
import itertools
import math
import os
import sys
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['MBAPY_FAST_LOAD'] = 'True'

from mbapy.base import put_err, put_log, split_list
from mbapy.bio.high_level import calcu_peptide_mutations
from mbapy.bio.peptide import AnimoAcid, MutationOpts, Peptide
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
    
    a = np.array(args.absorbance)
    m = np.array(args.weight)
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
    expand_mw_dict = [i.split('-') for i in args.weight if len(i) > 2]
    expand_mw_dict = {i[0]:i[1] for i in expand_mw_dict}
    peptide = Peptide(args.seq)
    _print(f'\npeptide: {peptide}')
    _print(f'MW: {peptide.calcu_mw(expand_mw_dict)}')
    if args.mass:
        _print(f'Chemical Formular: {peptide.get_molecular_formula()}, Exact Mass: {peptide.calcu_mass()}')
    return peptide, expand_mw_dict


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
        self.seq: Peptide = None
        self.verbose = True
        
    @staticmethod
    def make_mutation_args(args: argparse.ArgumentParser):
        args.add_argument('--max-repeat', type = int, default = 0,
                          help='max times for repeat any AA in sequence at all, default is %(default)s.')
        args.add_argument('--each-repeat', type=int, nargs='+', default=[],
                          help='each repeat times for each AA in sequence, input as "0 1" for Cys(Trt)-Leu-OH, default is follow --max-repeat.')

        args.add_argument('--replace-aa', type=str, default='',
                          help='AAs to replace, input as "Cys(Acm),Trt", default is no AA to replace.')
        args.add_argument('--max-replace', type=int, default=0,
                          help='max times for any AA replacement in sequence at all, 0 for no replacement, default is %(default)s.')
        args.add_argument('--each-replace', type=int, nargs='+', default=[],
                          help='each replacement times for each AA in sequence, input as "0 1" for Cys(Trt)-Leu-OH, max is 1, default is follow --max-replace.')
        
        args.add_argument('--max-deletion', type = int, default=None,
                          help='max times for any AA deletion in sequence at all, 0 for no deletion, None means 1, default is %(default)s.')
        args.add_argument('--each-deletion', type=int, nargs='+', default=[],
                          help='each deletion times for each AA in sequence, input as "0 1" for Cys(Trt)-Leu-OH, max is 1, default is follow --max-deletion.')
        
        args.add_argument('--max-deprotection', type=int, default=None,
                          help='max times for deprotect any AA in sequence at all, 0 for no deprotection, None means 1, default is %(default)s.')
        args.add_argument('--each-deprotection', type=str, default='',
                          help='each deprotection times for each AA in sequence, input as "0,1" for Cys(Trt)-Leu-OH, max is 1, default is follow --max-deprotection.')
        return args
        
    @staticmethod
    def make_args(args: argparse.ArgumentParser):
        args.add_argument('-s', '--seq', '--seqeunce', '--pep', '--peptide', type = str,
                          help='peptide seqeunce, input as Fmoc-Cys(Acm)-Leu-OH or H-Cys(Trt)-Leu-OH')
        args.add_argument('-w', '--weight', type = str, default = '',
                          help='MW of peptide AAs and protect group, input as Trt-243.34,Boc-101.13 and do not include weight of -H')
        args = mutation_weight.make_mutation_args(args) # add mutation args to args parser, such as --max-repeate, --replace-aa, etc.
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
    
    def process_mutation_args(self):
        self.seq = Peptide(self.args.seq.strip('"').strip('\''))
        def _argvec2intvec(args: argparse.Namespace, vec_name: List[int]):
            each_name, max_name = f'each_{vec_name}', f'max_{vec_name}'
            if getattr(args, each_name):
                intvec = getattr(args, each_name)
                if len(intvec) != len(self.seq.AAs):
                    raise ValueError(f'--{each_name} must have the same length as the peptide sequence.')
                return intvec
            else:
                max_value = getattr(args, max_name)
                return [max_value if max_value is not None else 1] * len(self.seq.AAs) # None means seq len, that is 1 for each aa
        self.args.each_repeat = _argvec2intvec(self.args, 'repeat')
        self.args.each_replace = _argvec2intvec(self.args, 'replace')
        self.args.each_deletion = _argvec2intvec(self.args, 'deletion')
        self.args.each_deprotection = _argvec2intvec(self.args, 'deprotection')
        self.args.max_deletion = self.args.max_deletion or len(self.seq.AAs)
        self.args.max_deprotection = self.args.max_deprotection or len(self.seq.AAs)
        if self.args.each_replace or self.args.max_replace:
            if self.args.replace_aa:
                self.args.replace_aa = [AnimoAcid(aa) for aa in self.args.replace_aa.split(',')]
            else:
                put_log(f'--repeat-aa not set, use all AAs in sequence to repeat.')
                self.args.replace_aa = list(set(self.seq.AAs))
    
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
        # task pool
        if self.args.multi_process > 1:
            self.task_pool = TaskPool('process', self.args.multi_process).start()
        else:
            self.task_pool = None
        # mutation controls
        self.process_mutation_args()
        
    def generate_mutation_opts(self):
        opts = []
        for i in range(len(self.args.each_repeat)):
            opts.append(MutationOpts(self.args.each_deletion[i], self.args.each_repeat[i], self.args.each_replace[i],
                                     self.args.replace_aa, self.args.each_deprotection[i],
                                     self.args.each_deprotection[i], self.args.each_deprotection[i]))
        return opts        
        
    def main_process(self):
        # show mother peptide info
        peptide, expand_mw_dict = calcu_mw(self.args, _print = self.printf)
        # make mutation opts
        opts = self.generate_mutation_opts()
        # calcu mutations, gather mutations, calcu mw and store in dict
        peps, mw2pep = calcu_peptide_mutations(self.seq, opts, self.args.mass,
                                               self.task_pool, self.args.batch_size,
                                               self.args.out if self.args.out is None else str(self.args.out)+'.mbapy.mmw.pkl')
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
        return peps, mw2pep


class cycmmw(Command):
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        

class fit_mass(mutation_weight):
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        self.task_pool: TaskPool = None
        self.mw2pep: Dict[int, List[Peptide]] = {}
        self.mass_dfs: Dict[str, MassData] = None
        self.ion_mode: Dict[str, Dict[str, float]] = None
        
    @staticmethod
    def make_args(args: argparse.ArgumentParser):
        args.add_argument('-s', '--seq', type = str, default='.',
                          help='input file/dir or peptide 3 letter seqeunce, default is %(default)s')
        args.add_argument('-m', '--mass-file', type=str, default='.',
                          help='input file/dir or peptide mass file, default is %(default)s')
        args.add_argument('--mass', type=bool, default=True,
                          help='calcu Exact Mass instead of Molecular Weight, default is %(default)s')
        args.add_argument('-sys', '--mass-system', type=str, choices=list(plot_mass.SUPPORT_SYS.keys())+['ALL'], default='ALL',
                          help='mass system, default is %(default)s')
        args.add_argument('-r', '--recursive', action='store_true', default=False,
                          help='recursive search for seq file and mass files in dir, default is %(default)s')
        args.add_argument('-o', '--output', type=str, default='.',
                          help='output dir, default is %(default)s')
        args.add_argument('--multi-process', type = int, default = 4,
                          help='number of multi-process to use. Defaults 1, no multi-process.')
        args = mutation_weight.make_mutation_args(args) # add mutation args to args parser, such as --max-repeate, --replace-aa, etc.
        args.add_argument('--ion-mode', type = str, nargs='+', default = 'all',
                          help=f'ion mode, default is %(default)s. Avaliable mode are: {list(MassData.ESI_IRON_MODE.keys())}')
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
        args.add_argument('--repr-w', type = int, default=3,
                          help = 'width of peptide animo acid representation in output, default is %(default)s')
        args.add_argument('--disable-repr-dash', default=False, action='store_true',
                          help = 'dash option of peptide animo acid representation in output, default is %(default)s')
        args.add_argument('--remain-old-match', default=False, action='store_true',
                          help='keep old match record in mass data, default is %(default)s')
        return args
        
    def process_args(self):
        show_args(self.args, ['seq'], self.printf)
        self.printf('processing arguments...\n')
        self.args.xlim = eval(f'({self.args.xlim})') if self.args.xlim is not None else None
        self.args.ms_lim = eval(f'({self.args.ms_lim})') if self.args.ms_lim is not None else None
        # set task pool
        if self.args.multi_process > 1:
            self.task_pool = TaskPool('process', self.args.multi_process).start()
            self.printf(f'task pool created with {self.args.multi_process} processes')
        # process argument: seq
        if os.path.isfile(clean_path(self.args.seq)) and self.args.seq.endswith('mbapy.mmw.pkl'):
            self.args.seq = str(clean_path(self.args.seq))
            self.mw2pep = opts_file(self.args.seq, mode = 'rb', way='pkl')['mw2pep']
            self.printf(f'load mw2pep from {self.args.seq}')
        elif os.path.isdir(clean_path(self.args.seq)):
            self.args.seq = str(clean_path(self.args.seq))
            for path in get_paths_with_extension(self.args.seq, ['mbapy.mmw.pkl'], self.args.recursive):
                mw2pep = opts_file(path, mode = 'rb', way='pkl')['mw2pep']
                load_count = 0
                for i in mw2pep:
                    if i in self.mw2pep:
                        extend = [pep for pep in mw2pep[i] if pep not in self.mw2pep[i]]
                        self.mw2pep[i].extend(extend)
                        load_count += 1
                    else:
                        self.mw2pep[i] = mw2pep[i]
                        load_count += 1
                self.printf(f'load {load_count} peptide(s) from {path} (contains {len(mw2pep)})')
            if not self.mw2pep:
                return put_err(f'no valid peptide found in {self.args.seq}, return None')
        else:
            self.process_mutation_args()
            # make mutation opts
            opts = self.generate_mutation_opts()
            # calcu mutations, gather mutations, calcu mw and store in dict
            _, self.mw2pep = calcu_peptide_mutations(self.seq, opts, self.args.mass,
                                                     self.task_pool, self.args.batch_size, None)
        self.printf(f'{len(self.mw2pep)} peptides loaded')
        # set ion mode
        if self.args.ion_mode == 'all':
            self.ion_mode = MassData.ESI_IRON_MODE
        else:
            self.ion_mode = {k: v for k, v in MassData.ESI_IRON_MODE.items() if k in self.args.ion_mode}
        self.printf(f'ion mode: {list(self.ion_mode.keys())}')
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
        
    def match_single_mass_data(self, candidates: np.ndarray, data_i: MassData, i: int, ms: float, h: float, charge: int, mono: bool):
        for mode, iron in self.ion_mode.items():
            if iron['c'] != charge:
                continue
            transfered_ms = (ms*charge-iron['im'])/iron['m']
            if self.args.ms_lim is None or (transfered_ms > self.args.ms_lim[0] and transfered_ms < self.args.ms_lim[1]):
                matched = np.where(np.abs(candidates - transfered_ms) <= self.args.error_tolerance)[0] # <= if the eps is 0
                if matched.size > 0:
                    all_matched_peps = [(pep, candidates[match_i]) for match_i in matched for pep in self.mw2pep[candidates[match_i]]]
                    data_i.add_match_record(ms, h, charge, mono, mode, ' | '.join(pep[0].repr(self.args.repr_w, True, not self.args.disable_repr_dash) for pep in all_matched_peps))
                    self.printf(f'matched {len(all_matched_peps)} peptide(s) with {mode} at {ms:.4f} (transfered: {transfered_ms:.4f})')
                    for i, (pep, pep_mass) in enumerate(all_matched_peps):
                        self.printf(f'{i}: ({len(pep.AAs)} AA)[{pep_mass:.4f}]{pep.get_molecular_formula()}: {pep.repr(self.args.repr_w, True, not self.args.disable_repr_dash)}')
                    self.printf('\n\n')
        return data_i

    def main_process(self):
        candidates = np.array(list(self.mw2pep.keys()))
        for n, data_i in self.mass_dfs.items():
            print(f'fitting {n} now...')
            # make peaks df
            if data_i.peak_df is None or data_i.check_processed_data_empty(data_i.peak_df):
                data_i.search_peaks(self.args.xlim, self.args.min_peak_width, self.task_pool, self.args.multi_process)
            data_i.filter_peaks(self.args.xlim, self.args.min_height, self.args.min_height_percent)
            # match and set match column
            ## set monoisotopic df
            if 'Monoisotopic' in data_i.peak_df.columns:
                monoisotopic_df = data_i.peak_df[data_i.peak_df['Monoisotopic']]
            else:
                put_log(f'{n} has no "Monoisotopic" column, using all peaks')
                monoisotopic_df = data_i.peak_df.copy(True)
            ## set charge column
            if data_i.CHARGE_HEADER is None:
                put_log(f'{n} has no charge header, assuming charge 1')
                charges = [1] * len(monoisotopic_df)
            else:
                charges = monoisotopic_df[data_i.CHARGE_HEADER].values
            ## get monoisotopic flag
            if 'Monoisotopic' in data_i.peak_df.columns:
                monoisotopic = monoisotopic_df['Monoisotopic']
            else:
                monoisotopic = [True] * len(monoisotopic_df)
            ## match
            if not self.args.remain_old_match:
                data_i.match_df = data_i.match_df.iloc[0:0]
            for i, (ms, h, charge, mono) in enumerate(zip(monoisotopic_df[data_i.X_HEADER], monoisotopic_df[data_i.Y_HEADER], charges, monoisotopic)):
                data_i = self.match_single_mass_data(candidates, data_i, i, ms, h, charge, mono)
            # save result
            data_i.save_processed_data()
        if self.args.multi_process > 1:
            self.task_pool.close(1)
        

class riddle_mass(fit_mass):
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        self.cache_exhaustivity_ms = None
        self.cache_all_combinations = None
        self.cache_all_combinations_ms = None
        from mbapy.chem import formula
        self.formula = formula
        
    @staticmethod
    def make_args(args: argparse.ArgumentParser):
        fit_mass.make_args(args)
        args.add_argument('--riddle-tolerance', type = float, default = 200,
                          help='make riddle if the transfered mass is within the error tolerance, default is %(default)s.')
        args.add_argument('--leaving-groups', type = str, nargs='+', default = ['H', 'OH'],
                          help='leaving groups for riddle, comma separated chemical formula string, default is %(default)s.')
        args.add_argument('--atom-candidates', type=str, nargs='+', default=['C', 'H', 'O', 'N', 'S'],
                          help = 'atom candidates for riddle, default is %(default)s')
        args.add_argument('--method', type=str, default='exhaustivity', choices=['exhaustivity'],
                          help = 'method to generate solution for riddle, default is %(default)s')
        args.add_argument('--atom-num-range', type=str, nargs='+', default=[],
                          help = 'atom candidates number range for riddle, input as "C,0,5 H,0,10", default is %(default)s')
        args.add_argument('--riddle-mode', type=str, nargs='+', default=['[M+H]+'],
                          help = 'iron mode for riddle, input as "[M+H]+ [M+Na]+", default is %(default)s')

    def process_args(self):
        super().process_args()
        self.args.leaving_groups = self.args.leaving_groups
        self.leaving_groups_ms = [AnimoAcid.calc_exact_mass(formula=lg) for lg in self.args.leaving_groups]
        self.args.atom_candidates = self.args.atom_candidates
        self.atom_candidates_msd = {ac: AnimoAcid.atom_msd[ac] for ac in self.args.atom_candidates}
        self.atom_candidates_msa = np.array([self.atom_candidates_msd[ac] for ac in self.args.atom_candidates])
        self.atom_candidates_range = {atom.split(',')[0]:atom.split(',')[1:] for atom in self.args.atom_num_range if atom}
        self.atom_num_min = np.array([int(self.atom_candidates_range.get(ac, (0, 10000))[0]) for ac in self.args.atom_candidates])
        self.atom_num_max = np.array([int(self.atom_candidates_range.get(ac, (0, 10000))[1]) for ac in self.args.atom_candidates])
        self.ESI_IRON_MODE = {mode: iron for mode, iron in MassData.ESI_IRON_MODE.items() if mode in self.args.riddle_mode}
        self.unsaturation = np.array([2, -1, 0, 1, 0])
        
    def riddle_mass_value_by_exhaustivity(self, ms: float) -> Tuple[List[str], List[float]]:
        """ms > 0"""
        if self.cache_exhaustivity_ms is None or ms > self.cache_exhaustivity_ms:
            self.cache_exhaustivity_ms = ms
            ## calcu all possible atoms combinations
            atom_candidates_num = [math.ceil(ms/AnimoAcid.atom_msd[ac]) for ac in self.args.atom_candidates]
            all_combinations = itertools.product(*[range(n+1) for n in atom_candidates_num])
            self.cache_all_combinations = all_combinations = np.array([list(combination) for combination in all_combinations])
            ## calcu all ms mat np.array
            self.cache_all_combinations_ms = all_combinations_ms = all_combinations.dot(self.atom_candidates_msa)
        else:
            all_combinations = self.cache_all_combinations
            all_combinations_ms = self.cache_all_combinations_ms
        # filter candidates by error tolerance and atom number range
        eps_index = np.abs(all_combinations_ms - ms) <= self.args.error_tolerance
        range_index = np.all((all_combinations >= self.atom_num_min) & (all_combinations <= self.atom_num_max), axis=-1)
        # filter by chemical degree of unsaturation
        unsaturation = (all_combinations.dot(self.unsaturation) + 2 - 1) / 2 # unsaturation = (2*C+2+N−H−X−L)/2, X is Cl, L is link num
        unsaturation_index = unsaturation >= 0
        # filter by chemical formula validity
        index = np.where(eps_index & range_index & unsaturation_index)
        valid_combinations, valid_combinations_ms = [], []
        for combination, ms in zip(all_combinations[index], all_combinations_ms[index]):
            formula_str = ''.join([f'{a}{n}' for a, n in zip(self.args.atom_candidates, combination) if n > 0])
            if self.formula.check_formula_existence(formula_str, link=1) != (None, None):
                valid_combinations.append(combination)
                valid_combinations_ms.append(ms)
        return valid_combinations, valid_combinations_ms
        
    def riddle_mass_diff_by_exhaustivity(self, pep: Peptide, pep_mass: float, diff: float) -> Tuple[List[str], List[float]]:
        # if diff > 0, candidates shuold drop the leaving group and add the candidate atom combination to fit the diff be within the error tolerance
        if diff > 0:
            candidates = {}
            max_candidate_ms = diff + max(self.leaving_groups_ms)
            self.riddle_mass_value_by_exhaustivity(max_candidate_ms)
            for name, ms in zip(self.args.leaving_groups, self.leaving_groups_ms):
                candidates[name] = self.riddle_mass_value_by_exhaustivity(diff + ms)
            lgs = list(candidates.keys())
            comb2formula = lambda comb: ''.join([f'{a}{n}' for a, n in zip(self.args.atom_candidates, comb) if n > 0])
            subs = [[comb2formula(comb) for comb in combs] for combs, _ in candidates.values()]
            subs_ms = [ms for _, ms in candidates.values()]
        else: # diff < 0 only
        # if diff < 0, candidates shuold add the leaving group (s)? and ?
            return [], [], []
        return lgs, subs, subs_ms

    def match_single_mass_data(self, candidates: np.ndarray, data_i: MassData, i: int, ms: float, h: float, charge: int, mono: bool):
        # exact match
        super().match_single_mass_data(candidates, data_i, i, ms, h, charge, mono)
        # riddle the left
        for mode, iron in self.ESI_IRON_MODE.items():
            if iron['c'] != charge:
                continue
            transfered_ms = (ms*charge-iron['im'])/iron['m']
            if self.args.ms_lim is None or (transfered_ms > self.args.ms_lim[0] and transfered_ms < self.args.ms_lim[1]):
                diff = transfered_ms - candidates
                matched = np.where((np.abs(diff) > self.args.error_tolerance) & (np.abs(diff) < self.args.riddle_tolerance))[0] # so the abs(diff) must > 0
                all_matched_peps = [(pep, candidates[match_i], diff[match_i]) for match_i in matched for pep in self.mw2pep[candidates[match_i]]]
                for pep, pep_mass, diff in all_matched_peps:
                    lgs, subss, subss_ms = getattr(self, f'riddle_mass_diff_by_{self.args.method}')(pep, pep_mass, diff)
                    if all(not subs for subs in subss):
                        continue
                    pep_rper = pep.repr(self.args.repr_w, True, not self.args.disable_repr_dash)
                    for lg, subs in zip(lgs, subss):
                        if subs:
                            data_i.add_match_record(ms, h, charge, mono, mode, f'{pep_rper} -{lg}+[' + '/'.join(subs) + ']')
                    self.printf(f'riddle {sum(map(len, subss))} subs(s) for {pep_rper} with {mode} at {ms:.4f} (transfered: {transfered_ms:.4f})')
                    for i, (lg, subs, sub_ms) in enumerate(zip(lgs, subss, subss_ms)):
                        if subs:
                            self.printf(f'{i}: ({len(pep.AAs)} AA)[{pep_mass+min(sub_ms):.4f} ~ {pep_mass+max(sub_ms):.4f}]{pep.get_molecular_formula()}-{lg}: {pep.repr(self.args.repr_w, True, not self.args.disable_repr_dash)} -{lg}+{subs}')
                    self.printf('\n\n')
        # save formula cache
        opts_file(self.formula.FORMULA_EXISTENCE_CACHE_PATH, 'wb', way='pkl', data = self.formula.formula_existence_cache)
        return data_i

        
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
    
    'riddle-mass': riddle_mass,
    
    'letters': transfer_letters,
    'transfer-letters': transfer_letters,
}


def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser()
    subparsers = args_paser.add_subparsers(title='subcommands', dest='sub_command')
    
    sub_val_args = subparsers.add_parser('subval', aliases = ['sb'], description='calcu SPPS substitution value for a release test of resin.')
    sub_val_args.add_argument('-a', '-A', '--absorbance', '--Absorbance', type = float, nargs='+',
                              help='Absorbance (OD value), input as 0.503 0.533')
    sub_val_args.add_argument('-m', '-w', '--weight', type = float, nargs='+',
                              help='resin wight (mg), input as 0.165 0.155')
    sub_val_args.add_argument('-c', '--coff', default = 16.4, type = float,
                              help='coff, default is 16.4')
    
    molecularweight = subparsers.add_parser('molecularweight', aliases = ['molecular-weight', 'mw'], description='calcu MW of peptide.')
    molecularweight.add_argument('-s', '--seq', '--seqeunce', '--pep', '--peptide', type = str,
                                 help='peptide seqeunce, input as Fmoc-Cys(Acm)-Leu-OH or H-Cys(Trt)-Leu-OH')
    molecularweight.add_argument('-w', '--weight', type = str, nargs='+', default = [],
                                 help='MW of peptide AAs and protect group, input as Trt-243.34 Boc-101.13 and do not include weight of -H')
    molecularweight.add_argument('-m', '--mass', action='store_true', default=False,
                                 help='calcu Exact Mass instead of Molecular Weight.')
    
    mutationweight_args = mutation_weight.make_args(subparsers.add_parser('mutationweight', aliases = ['mutation-weight', 'mmw'], description='calcu MW of each peptide mutations syn by SPPS.'))
    
    fit_mass_args = fit_mass.make_args(subparsers.add_parser('fit-mass', description='fit peptide mutation and mass iron exact mass for mass data'))
    
    riddle_mass_args = riddle_mass.make_args(subparsers.add_parser('riddle-mass', description='fit peptide mutation and mass iron exact mass for mass data with riddle'))
    
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
    # main('riddle-mass -s data_tmp/scripts/peptide/C.mbapy.mmw.pkl -m data_tmp/scripts/mass/pl.txt'.split())
    # main('mmw -s Boc-Asn(Trt)-Asp(OtBu)-Glu(OtBu)-Cys(Trt)-Glu(OtBu)-Leu-OH -m --max-repeat 0 --max-deletion 0 --max-replace 0  --multi-process 4'.split())
    
    # release code
    main()