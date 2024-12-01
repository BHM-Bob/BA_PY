import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List


class AnimoAcid:
    atom_msd = { # atom exact mass dict, from pyteomics.mass.calculate_mass
        'H': 1.00782503207,
        'C': 12.0,
        'N': 14.0030740048,
        'O': 15.99491461956,
        'S': 31.972071,
    }
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
    aa_3to1 = {
        'Ala': 'A',
        'Arg': 'R',
        'Asn': 'N',
        'Asp': 'D',
        'Cys': 'C',
        'Gln': 'Q',
        'Glu': 'E',
        'Gly': 'G',
        'His': 'H',
        'Ile': 'I',
        'Leu': 'L',
        'Lys': 'K',
        'Met': 'M',
        'Phe': 'F',
        'Pro': 'P',
        'Ser': 'S',
        'Thr': 'T',
        'Trp': 'W',
        'Tyr': 'Y',
        'Val': 'V',
    }
    aa_1to3 = {v:k for k,v in aa_3to1.items()}
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
        "Gly": {'C': 2, 'H': 5, 'O':2, 'N':1, 'S':0, 'P':0},
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
        'OtBu':{'C': 4, 'H': 9, 'O':0, 'N':0, 'S':0, 'P':0}, # has ZERO Oxygen atom because lose H2O when condensed with AAs, deleted (H)
        'tBu' :{'C': 4, 'H': 9, 'O':0, 'N':0, 'S':0, 'P':0}, # deleted (H)
        'Trt' :{'C':19, 'H':15, 'O':0, 'N':0, 'S':0, 'P':0}, # deleted (H)
    }
    all_mwd = deepcopy(aa_mwd)
    all_mwd.update(pg_mwd)
    def __init__(self, repr: str, aa_repr_w: int = 3) -> None:
        """
        Initializes an instance of the class with the given representation string.

        Parameters:
            repr (str): The representation string of a peptide to initialize the instance with.
        """
        if repr is not None:
            parts = repr.split('-')
            if len(parts) == 1:
                assert self.check_is_aa(parts[0][:3]), f'{repr} is not a valid animo acid, it has noly one part and should in {self.aa_mwd.keys()}'
                parts = ['H'] + parts + ['OH']
            elif len(parts) == 2:
                if self.check_is_aa(parts[0][:3]) == aa_repr_w:
                    parts = ['H'] + parts
                elif self.check_is_aa(parts[1][:3]) == aa_repr_w:
                    parts = parts + ['OH']
                else:
                    raise ValueError(f'{repr} is not a valid animo acid, it has two parts and none is in {self.aa_mwd.keys()} with it\'s previous 3 chars')
            elif len(parts) > 3:
                raise ValueError(f'{repr} is not a valid animo acid, it has more than 3 parts splited by dash \'-\'')
            self.N_protect = parts[0]
            self.animo_acid = parts[1]
            self.C_protect = parts[2]
            if '(' in parts[1] or len(parts[1]) == 1:
                if self.check_is_aa(parts[1]) == 1:
                    self.animo_acid = self.aa_1to3[parts[1][0]]
                    self.R_protect = parts[1][2:-1] if parts[1][2:-1] else 'H'
                else:
                    self.animo_acid = parts[1][0:3]
                    self.R_protect = parts[1][4:-1]
            else:
                self.R_protect = 'H'
        else:
            self.N_protect, self.animo_acid, self.R_protect, self.C_protect = None, None, None, None
            
    @staticmethod
    def calc_exact_mass(formula: str = None, atom_dict: Dict[str, int] = None) -> float:
        """using AminoAcid.atom_msd to calculate the exact mass of the given formula."""
        if formula is not None:
            assert atom_dict is None, 'atom_dict should be None when formula is not None'
            atom_dict = {atom: count for atom, count in re.findall(r'([A-Z])(\d*)', formula.upper())}
        return sum([AnimoAcid.atom_msd[atom] * (int(count) if count else 1) for atom, count in atom_dict.items()])
              
    @staticmethod  
    def check_is_aa(aa: str):
        if aa[0] in AnimoAcid.aa_1to3 and (len(aa) == 1 or aa[1] == '('):
            return 1
        elif aa[:3] in AnimoAcid.aa_3to1:
            return 3
        return 0
                
    def make_pep_repr(self, is_N_terminal: bool = False, is_C_terminal: bool = False,
                      repr_w: int = 3, include_pg: bool = True):
        """
        Generate a PEP representation of the amino acid sequence.

        Args:
            is_N_terminal (bool, optional): Whether the sequence is at the N-terminus. Defaults to False.
            is_C_terminal (bool, optional): Whether the sequence is at the C-terminus. Defaults to False.

        Returns:
            str: The PEP representation of the amino acid sequence.
        """
        assert repr_w in [1, 3], "repr_w must be 1 or 3"
        aa = self.animo_acid if repr_w == 3 else self.aa_3to1[self.animo_acid]
        if include_pg:
            parts = [f'{self.N_protect}-'] if (self.N_protect != 'H' or is_N_terminal) else []
            parts += ([aa] if self.R_protect == 'H' else [f'{aa}({self.R_protect})'])
            parts += ([f'-{self.C_protect}'] if (self.C_protect != 'OH' or is_C_terminal) else [])
        else:
            parts = [aa]
        return ''.join(parts)
    
    def __repr__(self) -> str:
        return self.make_pep_repr(True, True)
    
    def __eq__(self, other: 'AnimoAcid') -> bool:
        return self.N_protect == other.N_protect and self.animo_acid == other.animo_acid and self.C_protect == other.C_protect and self.R_protect == other.R_protect
    
    def __hash__(self) -> int:
        return hash(self.N_protect + self.animo_acid + self.C_protect + self.R_protect)
    
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
            mfd['H'] -= 1 # normally lose one H atom, so we need to minus one H for AA
            for k,v in self.mfd[self.N_protect].items():
                mfd[k] += v
        if self.C_protect != 'OH':
            mfd['H'] -= 1 # normally lose one H atom, so we need to minus one H for AA
            for k,v in self.mfd[self.C_protect].items():
                mfd[k] += v
        if self.R_protect != 'H':
            mfd['H'] -= 1 # normally lose one H atom, so we need to minus one H for AA
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
        try:
            from pyteomics import mass
            return mass.calculate_mass(formula=molecular_formula)
        except ImportError:
            return self.calc_exact_mass(formula=molecular_formula)
    
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
    
class AminoAcid(AnimoAcid):
    """fix the name of AnimoAcid to AminoAcid"""
    
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
    def __init__(self, repr: str, aa_repr_w: int = 3) -> None:
        assert aa_repr_w in [1, 3], "repr_w must be 1 or 3"
        if repr is not None:
            if aa_repr_w == 3:
                # 3 letters repr
                parts = repr.split('-')
                if parts[0] in AnimoAcid.pg_mwd.keys():
                    parts[1] = parts[0] + '-' + parts[1]
                    del parts[0]
                if parts[-1] in AnimoAcid.pg_mwd.keys():
                    parts[-2] = parts[-2] + '-' + parts[-1]
                    del parts[-1]
            else:
                # 1 letter repr
                pattern = f"[{'|'.join(AnimoAcid.aa_3to1.values())}]" + r'(?:\([A-Za-z]+\))?'
                N_pg, C_pg = "", ""
                if '-' in repr:
                    parts = repr.split('-')
                    # has pg in N terminal
                    if not AnimoAcid.check_is_aa(parts[0]):
                        N_pg = parts[0] + '-'
                        repr = repr.replace(N_pg,'')
                    # has pg in C terminal
                    if not AnimoAcid.check_is_aa(parts[-1]):
                        C_pg = parts[-1] + '-'
                        repr = repr.replace(C_pg,'')
                parts = re.findall(pattern, repr)
                parts[0] = N_pg + parts[0]
                parts[-1] = parts[-1] + C_pg
            # generate AAs
            self.AAs = [AnimoAcid(part, aa_repr_w) for part in parts]
        else:
            self.AAs = []
        
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
        return seq
        
    def repr(self, repr_w: int = 3, include_pg: bool = True,
             include_dash: bool = True):
        """
        NOTE: if do not include dash and include pg, the N and C ternimal pg will STILL get a dash.
        """
        assert repr_w in [1, 3], "repr_w must be 1 or 3"
        seq = self.flatten(inplace=False)
        if not seq:
            return ''
        dash = "-" if include_dash else ""
        repr_str = dash.join([aa.make_pep_repr(is_N_terminal=(i==0),
                                                is_C_terminal=(i==len(seq)-1),
                                                repr_w = repr_w,
                                                include_pg = include_pg) \
                                                    for i, aa in enumerate(seq)])
        # handle terminal Acid with no pg
        if not include_dash:
            if seq[0].N_protect == 'H' and repr_str[:2] == 'H-':                
                repr_str = repr_str[2:]
            if seq[-1].C_protect == 'OH' and repr_str[-3:] == '-OH':
                repr_str = repr_str[:-3]
        return repr_str
        
    def __repr__(self) -> str:
        return self.repr(3, True, True)
    
    def __eq__(self, other: 'Peptide') -> bool:
        return all([aa1 == aa2 for aa1, aa2 in zip(self.AAs, other.AAs)])
    
    def __hash__(self) -> int:
        return hash(self.repr(1, True, False))
    
    def __len__(self) -> int:
        return len(self.AAs)
    
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
    

@dataclass
class MutationOpts:
    AA_deletion: bool = True # whether delete AA can be performed
    AA_repeat: int = 1 # AA repeat times of AA
    AA_replace: bool = False # whether replace AA can be performed
    AA_replace_AAs: List[AnimoAcid] = field(default_factory=list) # AA replace AAs of AA
    N_protect_deletion: bool = True # whether delete N-terminal protect group can be performed
    C_protect_deletion: bool = True # whether delete C-terminal protect group can be performed
    R_protect_deletion: bool = True # whether delete R-terminal protect group can be performed
    
    def copy(self):
        return MutationOpts(self.AA_deletion, self.AA_repeat,
                            self.AA_replace, [aa.copy() for aa in self.AA_replace_AAs],
                            self.N_protect_deletion, self.C_protect_deletion, self.R_protect_deletion)
    
    def check_empty(self, _pos: List[int], seq: Peptide):
        """
        return list of signals which is able to opt, if empty, the lis is also empty.
        """
        pos, repeat_pos, sum_repeat = _pos
        able = []
        if pos >= len(seq.AAs):
            return []
        if sum_repeat == 1:
            # this AA has not been repeated yet, make all options
            if self.AA_deletion:
                able.append('AA_deletion')
            if self.AA_replace and self.AA_replace_AAs:
                able.append('AA_replace')
            if self.AA_repeat > 0:
                able.append('AA_repeat')
            if seq.AAs[pos].N_protect != 'H' and self.N_protect_deletion:
                able.append('N_protect_deletion')
            if seq.AAs[pos].C_protect != 'OH' and self.C_protect_deletion:
                able.append('C_protect_deletion')
            if seq.AAs[pos].R_protect != 'H' and self.R_protect_deletion:
                able.append('R_protect_deletion')
        else:
            # this AA has been repeated, make only deprotection options
            if seq.AAs[pos][repeat_pos].N_protect != 'H' and self.N_protect_deletion:
                able.append('N_protect_deletion')
            if seq.AAs[pos][repeat_pos].C_protect != 'OH' and self.C_protect_deletion:
                able.append('C_protect_deletion')
            if seq.AAs[pos][repeat_pos].R_protect != 'H' and self.R_protect_deletion:
                able.append('R_protect_deletion')
        return able
                
    def delete_AA(self, tree: 'MutationTree'):
        """
        perform delete_AA mutation in tree.mutate branch, trun off the tree.branches.opt.AA_deletion.
            - THE AA CAN NOT BE REPEATED.
            
        Params:
            - tree: MutationTree, tree to opt.
        """
        pos, repeat_pos, sum_repeat = tree.pos
        # perform delete_AA mutation in tree.mutate branch
        tree.mutate.seq.AAs[pos] = []
        # mutate branch MOVE TO NEXT NEW AA, this also means trun off the delete AA opt in mutate branch
        tree.mutate.pos[0] += 1
        if len(tree.OPTS) > tree.mutate.pos[0]:
            tree.mutate.opt = tree.OPTS[tree.mutate.pos[0]].copy()
        # trun off the delete AA opt in remain branch
        tree.remain.opt.AA_deletion = False
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
            for _ in range(tree.opt.AA_repeat + 1)]
        # change repeated AAs' N/C protect group if needed
        if pos == 0 and tree.PEPTIDE.AAs[pos].N_protect != 'H':
            for aa in tree.mutate.seq.AAs[pos][1:]:
                aa.N_protect = 'H'
        elif pos == len(tree.PEPTIDE.AAs)-1 and tree.PEPTIDE.AAs[pos].C_protect != 'OH':
            for aa in tree.mutate.seq.AAs[pos][:-1]:
                aa.C_protect = 'OH'
        # change mutate branch 's pos 's sum_repeat to tree.opt.repeat_AA + 1
        tree.mutate.pos[2] = tree.opt.AA_repeat + 1
        # trun off the repeat AA opt in mutate branches
        tree.mutate.opt.AA_repeat = 0
        # decrease the repeat AA opt in remain branches
        tree.remain.opt.AA_repeat -= 1
        return tree
    
    def replace_AA(self, tree: 'MutationTree'):
        """
        perform replace_AA mutation in tree.mutate branch, trun off the tree.branches.opt.AA_replace.
        
        Params:
            - tree: MutationTree, tree to opt.
        """
        pos, repeat_pos, sum_repeat = tree.pos
        # perform replace_AA mutation in tree.mutate branch
        tree.mutate.seq.AAs[pos] = self.AA_replace_AAs[0]
        # trun off the replace AA opt in two branches
        tree.mutate.opt.AA_replace_AAs.pop(0)
        tree.remain.opt.AA_replace_AAs.pop(0)
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
        # trun off the opt in two branches
        setattr(tree.mutate.opt, f'{NCR}_protect_deletion', False)
        setattr(tree.remain.opt, f'{NCR}_protect_deletion', False)
        return tree
                
    def perform_one(self, tree: 'MutationTree'):
        """
        Perform ONE mutation opt left in tree.opt, return this tree. Also check if it is a repeated AA.
        If it is a repeated AA depend on tree.pos[2], skip AA deletion and AA repeat.
            - If no opt left to do:
                - let two brance still be None.
                - return the tree.
            - IF HAS:
                - generate two branch, change branches' father
                - perform mutation in mutate branch
                - trun off opt in two branches
                - move pos ONLY in mutate branch.
                - DO NOT CHECK IF MEETS END in both this dot and two branches.
                - return the tree.
        """
        able = tree.opt.check_empty(tree.pos, tree.seq)
        if able:
            # generate two branch and set seq to None to free memory
            tree.generate_two_branch()
            # perform mutation
            if 'AA_deletion' in able:
                tree = self.delete_AA(tree)
            elif 'AA_replace' in able:
                tree = self.replace_AA(tree)
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
    PEPTIDE: Peptide # mother peptide, READ ONLY, remians unchanged
    OPTS: List[MutationOpts] # each AAs' opt to perform, READ-ONLY, remains unchanged
    seq: Peptide # current dot's peptide seqeunce to perform mutate
    opt: MutationOpts # current dot's opt to perform
    # [current AA pos, current repeat pos, sum repeat in this AA in seq], if last number is 1, means no repeat in this AA
    pos: List[int] = field(default_factory = lambda: [0, 0, 1])
    father: 'MutationTree' = None # father dot
    remain: 'MutationTree' = None # father dot
    mutate: 'MutationTree' = None # father dot
    
    def copy(self, father: 'MutationTree' = None, remain: 'MutationTree' = None,
             mutate: 'MutationTree' = None):
        """
        """
        cp = MutationTree(self.PEPTIDE, self.OPTS, self.seq.copy(), self.opt.copy(), [i for i in self.pos])
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
        if self.pos[0] >= len(self.PEPTIDE.AAs) - 1 and self.pos[1] >= self.pos[2] - 1:
            return True
        return False
        
    def generate_two_branch(self):
        """Generate two branch with all None remian and mutate branch, return itself."""
        self.remain = self.copy(father=self)
        self.mutate = self.copy(father=self)
        return self
        
    def move_to_next(self):
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
            # copy next original opt to current opt
            self.opt = self.OPTS[self.pos[0]].copy()
            return True
        return False
    

__all__ = [
    'AnimoAcid',
    'Peptide',
    'MutationOpts',
    'MutationTree'
]


if __name__ == "__main__":
    # dev code
    pep = Peptide('Leu-OtBu', 3)
    # dev code
    pep = Peptide('H-Cys(Trt)-G(OtBu)', 3)
    print(pep.repr(3, True, True))
    print(pep.repr(3, False, True))
    print(pep.repr(3, True, False))
    print(pep.repr(3, False, False))
    print(pep.repr(1, True, True))
    print(pep.repr(1, False, True))
    print(pep.repr(1, True, False))
    print(pep.repr(1, False, False))