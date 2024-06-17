'''
Date: 2024-06-17 15:06:17
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-06-17 15:20:03
Description: 
'''
from typing import List

from mbapy.bio.peptide import Peptide
from mbapy.file import opts_file
from mbapy.game import BaseInfo
from mbapy.scripts.peptide import mutation_weight


def calcu_peptide_mutations(peptide: Peptide, max_repeat: int = 0,
                              disable_aa_deletion: bool = True, mass: bool = True,
                              n_worker: int = 1, result_path: str = None):
    """
    Parameters:
        - peptide: Peptide object, mother peptide to generate mutations.
        - max_repeat: int, maximum repeat number of amino acids in a peptide.
        - disable_aa_deletion: bool, whether to disable amino acid deletion.
        - mass: bool, whether to calculate the Exact Mass of each mutation.
        - n_worker: int, number of process to calculate the mutations.
        - result_path: str, path to save the result. If None, the result will not be saved.

    Returns:
        - peps: Dict[str, int], all possible mutations string repr and their corresponding index.
        - mw2pep: Dict[float, List[Peptide]], dictionary of the exact mass of each mutation and its corresponding peptide.
    """
    args = BaseInfo(max_repeat = max_repeat, disable_aa_deletion = disable_aa_deletion)
    seqs = mutation_weight.generate_mutate_peps(peptide, args)
    peps, mw2pep = mutation_weight.calcu_mutations_mw(seqs, mass, n_worker)
    if result_path:
        opts_file(result_path, 'wb', data = {'mw2pep':mw2pep, 'peps':peps}, way = 'pkl')
    return peps, mw2pep


__all__ = [
    'calcu_peptide_mutations'
]