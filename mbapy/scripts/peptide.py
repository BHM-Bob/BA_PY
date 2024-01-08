import argparse
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['MBAPY_FAST_LOAD'] = 'True'
from mbapy import base, file


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
    pg_mwd = { # protect group molecular weight dict
        'H': 0, # do not calcu mw
        'OH': 0, # do not calcu mw
        'Acm': 72.10, # = 73.10 - 1.0(H)
        'Boc': 101.13, # = 102.13 - 1.0(H)
        'Fmoc': 223.26, # = 224.26 - 1.0(H)
        'OtBu': 57.12, # = 58.12 - 1.0(H)
        'tBu': 57.12, # = 58.12 - 1.0(H)
        'Trt': 243.34, # = 244.34 - 1.0(H)
    }
    mfd = { # AnimoAcid molecular formula dict
        "Ala": {'C': 3, 'H': 7, 'O':2, 'N':1, 'S':0, 'P':0},
        "Arg": {'C': 6, 'H':14, 'O':2, 'N':4, 'S':0, 'P':0},
        "Asn": {'C': 4, 'H': 8, 'O':3, 'N':2, 'S':0, 'P':0},
        "Asp": {'C': 4, 'H': 7, 'O':4, 'N':1, 'S':0, 'P':0},
        "Cys": {'C': 3, 'H': 7, 'O':2, 'N':1, 'S':1, 'P':0},
        "Gln": {'C': 5, 'H':10, 'O':3, 'N':2, 'S':0, 'P':0},
        "Glu": {'C': 5, 'H': 9, 'O':4, 'N':1, 'S':0, 'P':0},
        "Gly": {'C': 2, 'H': 5, 'O':2, 'N':0, 'S':0, 'P':0},
        "His": {'C': 6, 'H': 9, 'O':2, 'N':3, 'S':0, 'P':0},
        "Ile": {'C': 6, 'H':13, 'O':2, 'N':1, 'S':0, 'P':0},
        "Leu": {'C': 6, 'H':13, 'O':2, 'N':1, 'S':0, 'P':0},
        "Lys": {'C': 6, 'H':14, 'O':2, 'N':2, 'S':0, 'P':0},
        "Met": {'C': 5, 'H':11, 'O':2, 'N':1, 'S':1, 'P':0},
        "Phe": {'C': 9, 'H':11, 'O':2, 'N':1, 'S':0, 'P':0},
        "Pro": {'C': 5, 'H': 9, 'O':2, 'N':1, 'S':0, 'P':0},
        "Ser": {'C': 3, 'H': 7, 'O':3, 'N':1, 'S':0, 'P':0},
        "Thr": {'C': 4, 'H': 9, 'O':3, 'N':1, 'S':0, 'P':0},
        "Trp": {'C':11, 'H':12, 'O':2, 'N':2, 'S':0, 'P':0},
        "Tyr": {'C': 9, 'H':11, 'O':3, 'N':1, 'S':0, 'P':0},
        "Val": {'C': 5, 'H':11, 'O':2, 'N':1, 'S':0, 'P':0},
        'H'   :{'C': 0, 'H': 1, 'O':0, 'N':0, 'S':0, 'P':0},
        'OH'  :{'C': 0, 'H': 1, 'O':1, 'N':0, 'S':0, 'P':0},
        'Acm' :{'C': 3, 'H': 6, 'O':1, 'N':1, 'S':0, 'P':0}, # deleted (H)
        'Boc' :{'C': 5, 'H': 9, 'O':2, 'N':0, 'S':0, 'P':0}, # deleted (H)
        'Fmoc':{'C':15, 'H':11, 'O':2, 'N':0, 'S':0, 'P':0}, # deleted (H)
        'OtBu':{'C': 4, 'H': 9, 'O':0, 'N':0, 'S':0, 'P':0}, # deleted (H)
        'tBu' :{'C': 4, 'H': 9, 'O':0, 'N':0, 'S':0, 'P':0}, # deleted (H)
        'Trt' :{'C':19, 'H':15, 'O':0, 'N':0, 'S':0, 'P':0}, # deleted (H)
    }
    all_mwd = deepcopy(aa_mwd)
    all_mwd.update(pg_mwd)
    def __init__(self, repr: str) -> None:
        """
        Initializes an instance of the class with the given representation string.

        Parameters:
            repr (str): The representation string of a peptide to initialize the instance with.
        """
        if repr is not None:
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
        """
        Generate a PEP representation of the amino acid sequence.

        Args:
            is_N_terminal (bool, optional): Whether the sequence is at the N-terminus. Defaults to False.
            is_C_terminal (bool, optional): Whether the sequence is at the C-terminus. Defaults to False.

        Returns:
            str: The PEP representation of the amino acid sequence.
        """
        parts = []
        parts += ([f'{self.N_protect}-'] if (self.N_protect != 'H' or is_N_terminal) else [])
        parts += ([self.animo_acid] if self.R_protect == 'H' else [f'{self.animo_acid}({self.R_protect})'])
        parts += ([f'-{self.C_protect}'] if (self.C_protect != 'OH' or is_C_terminal) else [])
        return ''.join(parts)
    
    def __repr__(self) -> str:
        return self.make_pep_repr(True, True)
    
    def calcu_mw(self, expand_mw_dict: Dict[str, float] = None):
        """
        Calculate the molecular weight of the peptide sequence.

        Args:
            expand_mw_dict (Dict[str, float], optional): A dictionary containing the molecular weights of the protect groups. Defaults to None.

        Returns:
            float: The calculated molecular weight of the peptide sequence.
        """
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
    
    def get_molecular_formula_dict(self):
        """
        Returns a dictionary containing the molecular formula of the protein sequence.
        
        The function creates a deep copy of the molecular formula dictionary for the current amino acid. If the N-terminal protecting group is not 'H', it subtracts one hydrogen atom from the dictionary to account for the absence of a hydrogen atom in the N-terminal residue. It then adds the molecular formula of the N-terminal protecting group to the dictionary. Similarly, if the C-terminal protecting group is not 'OH', it subtracts one hydrogen atom and one oxygen atom from the dictionary to account for the absence of these atoms in the C-terminal residue. It then adds the molecular formula of the C-terminal protecting group to the dictionary. Finally, if the side chain protecting group is not 'H', it subtracts one hydrogen atom from the dictionary to account for the absence of a hydrogen atom in the side chain. It then adds the molecular formula of the side chain protecting group to the dictionary.

        Returns:
            mfd (dict): A dictionary containing the molecular formula of the protein sequence.
        """
        mfd = {k:v for k,v in self.mfd[self.animo_acid].items()} # deepcopy may slow
        if self.N_protect != 'H':
            mfd['H'] -= 1 # because N-terminal residue has no H but AA has H, so we need to minus one H for AA
            for k,v in self.mfd[self.N_protect].items():
                mfd[k] += v
        if self.C_protect != 'OH':
            mfd['H'] -= 1 # because C-terminal residue has no OH but AA has OH, so we need to minus one H for AA
            mfd['O'] -= 1 # because C-terminal residue has no OH but AA has OH, so we need to minus one H for AA
            for k,v in self.mfd[self.C_protect].items():
                mfd[k] += v
        if self.R_protect != 'H':
            mfd['H'] -= 1 # because C-terminal residue has no OH but AA has OH, so we need to minus one H for AA
            for k,v in self.mfd[self.R_protect].items():
                mfd[k] += v
        return mfd
    
    def get_molecular_formula(self, molecular_formula_dict: Dict[str, int] = None):
        """
        Generates the molecular formula from a given dictionary of element symbols and their counts.

        Args:
            molecular_formula_dict (Dict[str, int], optional): A dictionary containing element symbols as keys and their counts as values. Defaults to None.

        Returns:
            str: The molecular formula generated from the dictionary, with element symbols and their counts concatenated.

        Example:
            >>> molecular_formula_dict = {'C': 6, 'H': 12, 'O': 6}
            >>> get_molecular_formula(molecular_formula_dict)
            'C6H12O6'
        """
        if not isinstance(molecular_formula_dict, dict):
            molecular_formula_dict = self.get_molecular_formula_dict()
        return ''.join([f'{k}{v}' for k,v in molecular_formula_dict.items() if v > 0])
    
    def calcu_mass(self, molecular_formula: str = None,
                   molecular_formula_dict: Dict[str, int] = None):
        """
        Calculate the mass of a molecule based on its molecular formula.

        Args:
            molecular_formula (str, optional): The molecular formula of the molecule. Defaults to None.
            molecular_formula_dict (Dict[str, int], optional): A dictionary representing the molecular formula of 
                the molecule, where the keys are element symbols and the values are the number of atoms. 
                Defaults to None.

        Returns:
            float: The calculated mass of the molecule.
        """
        if not isinstance(molecular_formula, str):
            if not isinstance(molecular_formula_dict, dict):
                mfd = self.get_molecular_formula_dict()
            else:
                mfd = molecular_formula_dict
            molecular_formula = ''.join([f'{k}{v}' for k,v in mfd.items() if v > 0])
        from pyteomics import mass
        return mass.calculate_mass(formula=molecular_formula)
    
    def copy(self):
        """
        Creates a copy of the current instance.

        Returns:
            An `AnimoAcid` object that is a copy of the current instance.
        """
        cp = AnimoAcid(None)
        cp.animo_acid = self.animo_acid
        cp.C_protect = self.C_protect
        cp.N_protect = self.N_protect
        cp.R_protect = self.R_protect
        return cp
    
class Peptide:
    """
    This class definition is for a Peptide class.

    Attributes:
        - AAs (List[AnimoAcid]): A list of AnimoAcid objects representing the amino acids in the peptide.
        
    Methods:
        - __init__(self, repr: str): Initializes a Peptide object by splitting the input string and creating a list of AnimoAcid objects.
        - flatten(self, inplace: bool = False): Flattens the list of AnimoAcid objects into a single list. If inplace is True, the method makes the change in place and returns self, otherwise it returns the changed sequence only.
        - __repr__(self): Returns a string representation of the Peptide object by joining the representations of each AnimoAcid object in the sequence.
        - get_molecular_formula_dict(self): Returns a dictionary representing the molecular formula of the Peptide object by summing the molecular formulas of each AnimoAcid object in the sequence.
        - get_molecular_formula(self, molecular_formula_dict: Dict[str, int] = None): Returns a string representation of the molecular formula of the Peptide object by joining the elements of the molecular_formula_dict dictionary.
        - calcu_mw(self, expand_mw_dict: Dict[str, float] = None): Calculates the molecular weight of the Peptide object by summing the molecular weights of each AnimoAcid object in the sequence.
        - calcu_mass(self, molecular_formula: str = None, molecular_formula_dict: Dict[str, int] = None): Calculates the mass of the Peptide object by calling the calcu_mass method of the first AnimoAcid object in the sequence.
        - copy(self): Creates a copy of the Peptide object by creating a new Peptide object and copying the list of AnimoAcid objects.
    """
    def __init__(self, repr: str) -> None:
        if repr is not None:
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
    
    def get_molecular_formula_dict(self):
        """
        Calculates the molecular formula of the protein by summing the molecular formula of each amino acid.

        Returns:
            dict: A dictionary representing the molecular formula of the protein, where the keys are the elements and the values are the corresponding counts.
        """
        mfd = self.AAs[0].get_molecular_formula_dict()
        for aa in self.AAs[1:]:
            for k,v in aa.get_molecular_formula_dict().items():
                mfd[k] += v
            mfd['H'] -= 2
            mfd['O'] -= 1
        return mfd
    
    def get_molecular_formula(self, molecular_formula_dict: Dict[str, int] = None):
        """
        Generate the molecular formula from the given molecular formula dictionary.

        Args:
            molecular_formula_dict (Dict[str, int], optional): A dictionary representing the molecular formula, 
                where the keys are the element symbols and the values are the corresponding counts. 
                Defaults to None.

        Returns:
            str: The molecular formula as a string.

        """
        if not isinstance(molecular_formula_dict, dict):
            molecular_formula_dict = self.get_molecular_formula_dict()
        return ''.join([f'{k}{v}' for k,v in molecular_formula_dict.items() if v > 0])
    
    def calcu_mw(self, expand_mw_dict: Dict[str, float] = None):
        """
        Calculate the molecular weight (mw) of the peptide.

        Args:
            expand_mw_dict (Optional[Dict[str, float]]): A dictionary containing the molecular weights of the protected group. Defaults to None.

        Returns:
            float: The calculated molecular weight of the peptide.
        """
        if expand_mw_dict is not None:
            assert isinstance(expand_mw_dict, dict), 'expand_mw_dict should be a dict contains protect group molecular weight'
            AnimoAcid.all_mwd.update(expand_mw_dict)
        mw = sum([aa.calcu_mw(expand_mw_dict) for aa in self.AAs])
        mw -= (len(self.AAs) - 1) * 18.02 # because single AA do not has peptide bond, so we need to minus 18.02 for each bond
        return mw
    
    def calcu_mass(self, molecular_formula: str = None,
                   molecular_formula_dict: Dict[str, int] = None):
        """
        Calculates the mass of a molecule based on its molecular formula.

        Args:
            molecular_formula (str, optional): The molecular formula of the molecule. Defaults to None.
            molecular_formula_dict (Dict[str, int], optional): The dictionary representation of the molecular formula. Defaults to None.

        Returns:
            float: The calculated mass of the molecule.

        Example:
            >>> calcu_mass('H2O')
            18.01528
        """
        if not isinstance(molecular_formula, str) and\
            not isinstance(molecular_formula_dict, dict):
            mfd = self.get_molecular_formula_dict()
        return self.AAs[0].calcu_mass(molecular_formula = molecular_formula,
                                      molecular_formula_dict=mfd)
    
    def copy(self):
        """
        Create a copy of the Peptide object.

        Returns:
            Peptide: A new Peptide object that is an exact copy of the original.
        """
        cp = Peptide(None)
        cp.AAs = []
        for aa in self.AAs:
            if isinstance(aa, AnimoAcid):
                cp.AAs.append(aa.copy())
            elif isinstance(aa, list) and len(aa) == 0:
                cp.AAs.append([])
            elif isinstance(aa, list) and any([isinstance(aa_i, AnimoAcid) for aa_i in aa]):
                cp.AAs.append([aa_i.copy() for aa_i in aa])
            else:
                raise ValueError(f'unkown type {type(aa)}')
        return cp
    

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
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    
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
    # show args
    _print(f'get arg: seqeunce: {args.seq}', f)
    _print(f'get arg: weight: {args.weight}', f)
    _print(f'get arg: max-repeat: {args.max_repeat}', f)
    _print(f'get arg: out: {args.out}', f)
    _print(f'get arg: mass: {args.mass}', f)
    # show mother peptide info
    peptide, expand_mw_dict = calcu_mw(args, _print = lambda x : _print(x, f))
    # calcu mutations
    all_mutations = MutationTree(peptide=peptide, seq=peptide.copy(),
                                 opts=MutationOpts(AA_repeat=args.max_repeat),
                                 pos=[0, 0, 1])
    all_mutations = mutate_peptide(all_mutations, args.max_repeat)
    all_mutations = all_mutations.extract_mutations()
    mw2pep, peps = {}, {}
    for pep in all_mutations:
        if len(pep.AAs):
            pep_repr = str(pep)
            if pep_repr not in peps:
                peps[pep_repr] = len(peps)
                if args.mass:
                    mw = pep.calcu_mass()
                else:
                    mw = pep.calcu_mw()
                if mw in mw2pep:
                    mw2pep[mw].append(pep)
                else:
                    mw2pep[mw] = [pep]
    # output info
    _print(f'\n{len(peps)-1} mutations found, followings include one original peptide seqeunce:\n', f)
    idx, weigth_type = 0, 'Exact Mass' if args.mass else 'MW'
    for i, mw in enumerate(sorted(mw2pep)):
        _print(f'\n{weigth_type}: {mw:10.5f}', f)
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
#     calcu_mw_of_mutations(BaseInfo(seq = 'Fmoc-Cys(Acm)-Val-Asn(Trt)', out = '.',
#                                    max_repeat = 1, weight = '', mass = False))
#     calcu_mw_of_mutations(BaseInfo(seq = 'Fmoc-Cys(Acm)-Val-Asn(Trt)', out = '.',
#                                    max_repeat = 1, weight = '', mass = True))

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
    
    molecularnweight = subparsers.add_parser('molecularnweight', aliases = ['mw'], description='calcu MW of peptide.')
    molecularnweight.add_argument('-s', '--seq', '--seqeunce', '--pep', '--peptide', type = str,
                                  help='peptide seqeunce, input as Fmoc-Cys(Acm)-Leu-OH or H-Cys(Trt)-Leu-OH')
    molecularnweight.add_argument('-w', '--weight', type = str, default = '',
                                  help='MW of peptide AAs and protect group, input as Trt-243.34,Boc-101.13 and do not include weight of -H')
    molecularnweight.add_argument('-m', '--mass', action='store_true', default=False,
                                  help='calcu Exact Mass instead of Molecular Weight.')
    
    mutationweight = subparsers.add_parser('mutationweight', aliases = ['mmw'], description='calcu MW of each peptide mutations syn by SPPS.')
    mutationweight.add_argument('-s', '--seq', '--seqeunce', '--pep', '--peptide', type = str,
                                help='peptide seqeunce, input as Fmoc-Cys(Acm)-Leu-OH or H-Cys(Trt)-Leu-OH')
    mutationweight.add_argument('-w', '--weight', type = str, default = '',
                                help='MW of peptide AAs and protect group, input as Trt-243.34,Boc-101.13 and do not include weight of -H')
    mutationweight.add_argument('--max-repeat', type = int, default = 1,
                                help='max times for repeat a AA in sequence')
    mutationweight.add_argument('-o', '--out', type = str, default = None,
                                help='save results to output file/dir. Defaults None, do not save.')
    mutationweight.add_argument('-m', '--mass', action='store_true', default=False,
                                help='calcu Exact Mass instead of Molecular Weight.')
    
    args = args_paser.parse_args(sys_args)
    
    if args.sub_command in _str2func:
        print(f'excuting command: {args.sub_command}')
        _str2func[args.sub_command](args)
    else:
        base.put_err(f'no such sub commmand: {args.sub_command}')

if __name__ == "__main__":
    main()