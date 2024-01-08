
from copy import deepcopy
from typing import Dict, List

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
    

__all__ = [
    'AnimoAcid',
    'Peptide',
]