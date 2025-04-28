'''
Date: 2024-06-17 15:06:17
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-03-29 11:32:37
Description: 
'''

import itertools
from typing import Dict, List, Tuple

from tqdm import tqdm

if __name__ == '__main__':
    from mbapy.base import split_list, put_log
    from mbapy.bio.peptide import (AnimoAcid, MutationOpts, MutationTree,
                                   Peptide)
    from mbapy.file import opts_file
    from mbapy.web_utils.task import TaskPool
else:
    from ..base import split_list, put_log
    from ..file import opts_file
    from ..web_utils.task import TaskPool
    from .peptide import AnimoAcid, MutationOpts, MutationTree, Peptide


def calcu_mutations_mw_in_batch(seqs: List[Peptide], mass: bool = False, verbose: bool = True) -> Tuple[Dict[str, int], Dict[float, List[Peptide]]]:
    """
    Calculate the mutations and molecular weight of peptides in batch.

    Parameters
    ----------
    seqs : List[Peptide]
        List of peptides to be analyzed.
    mass : bool, optional
        If True, calculate the mass instead of the molecular weight. Default is False.
    verbose : bool, optional
        If True, display the progress bar during the calculation. Default is True.

    Returns
    -------
    Tuple[Dict[str, int], Dict[float, List[Peptide]]]
        A tuple containing two dictionaries:
        - The first dictionary maps the peptide representation to its index.
        - The second dictionary maps the molecular weight (or mass) to a list of peptides with that molecular weight (or mass).

    """
    peps, mw2pep = {}, {}
    for pep in tqdm(seqs, desc='Gathering mutations and Calculating molecular weight',
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

def mutate_peptide(tree: MutationTree):
    """
    Parameters:
        - mutations: Tree object, store all mutations and there relationship.
        - max_repeat: int
    """
    # perofrm ONE mutation
    tree = tree.opt.perform_one(tree)
    # if NO mutaion can be done, 
    if tree.mutate is None and tree.remain is None:
        # try move current AA in this tree to next AA
        if tree.move_to_next():
            # move success, go on
            mutate_peptide(tree)
        else:
            # it is the end, return tree
            return tree
    else: # go on with two branches
        mutate_peptide(tree.mutate)
        mutate_peptide(tree.remain)
    return tree

def calcu_peptide_mutations(peptide: Peptide, mutate_opts: List[MutationOpts],
                            mass: bool = True,
                            task_pool: TaskPool = None, batch_size: int = 10000,
                            result_path: str = None):
    """
    Parameters:
    -----------
        - peptide: Peptide object, mother peptide to generate mutations.
        - mass: bool, whether to calculate the Exact Mass of each mutation.
        - task_pool: TaskPool object, to perform the calculation in parallel.
        - batch_size: int, number of peptides to be calculated in each batch.
        - result_path: str, path to save the result. If None, the result will not be saved.

    Returns:
    --------
        - peps: Dict[str, int], all possible mutations string repr and their corresponding index.
        - mw2pep: Dict[float, List[Peptide]], dictionary of the exact mass of each mutation and its corresponding peptide.
    """
    # check arguments
    if len(peptide) != len(mutate_opts):
        raise ValueError('The length of peptide and mutate_opts should be equal.')
    # generate all possible mutations
    seqs = []
    for aa, opt in tqdm(zip(peptide.AAs, mutate_opts), desc='Mutating peptide'):
        pep = Peptide(None)
        pep.AAs = [aa.copy()]
        aa_mutations = MutationTree(PEPTIDE=pep, seq=pep.copy(),
                                    OPTS=[opt], opt=opt.copy(),
                                    pos=[0, 0, 1])
        aa_mutations = mutate_peptide(aa_mutations)
        seqs.append(aa_mutations.extract_mutations())
    seqs = list(itertools.product(*seqs))
    put_log(f'Total number of mutated candidates: {len(seqs)}')
    # calcu MW or Exact Mass
    if task_pool is None:
        peps, mw2pep = calcu_mutations_mw_in_batch(seqs, mass=mass, verbose=True)
    else:
        put_log('Gathering mutations and Calculating molecular weight...')
        peps, mw2pep = {}, {}
        for i, batch in enumerate(split_list(seqs, batch_size)):
            task_pool.add_task(f'{i}', calcu_mutations_mw_in_batch, batch, mass, False)
        task_pool.start().wait_till(lambda : task_pool.count_done_tasks() == len(task_pool.tasks), verbose=True)
        for (_, (peps_i, mw2pep_i), _) in task_pool.tasks.values():
            peps.update(peps_i)
            for i in mw2pep_i:
                if i in mw2pep:
                    mw2pep[i].extend(mw2pep_i[i])
                else:
                    mw2pep[i] = mw2pep_i[i]
    # save and return results
    if result_path:
        opts_file(result_path, 'wb', data = {'mw2pep':mw2pep, 'peps':peps}, way = 'pkl')
    return peps, mw2pep


__all__ = [
    'calcu_peptide_mutations'
]


if __name__ == '__main__':
    # dev code
    pep = Peptide('Boc-Ala-Glu(OtBu)-Asp(OtBu)-Ala-Glu(OtBu)-Asp(OtBu)-Ala-Glu(OtBu)-Asp(OtBu)-OtBu')
    replace_aa = [AnimoAcid('Cys(Acm)'), AnimoAcid('Val')]
    opts = [MutationOpts(AA_deletion=True, AA_repeat=1, AA_replace=True, AA_replace_AAs=replace_aa),
            MutationOpts(AA_deletion=False, AA_repeat=0, AA_replace=False, AA_replace_AAs=replace_aa),
            MutationOpts(AA_deletion=False, AA_repeat=0, AA_replace=False, AA_replace_AAs=replace_aa),
            MutationOpts(AA_deletion=False, AA_repeat=0, AA_replace=False, AA_replace_AAs=replace_aa),
            MutationOpts(AA_deletion=False, AA_repeat=0, AA_replace=False, AA_replace_AAs=replace_aa),
            MutationOpts(AA_deletion=False, AA_repeat=0, AA_replace=False, AA_replace_AAs=replace_aa),
            MutationOpts(AA_deletion=False, AA_repeat=0, AA_replace=False, AA_replace_AAs=replace_aa),
            MutationOpts(AA_deletion=False, AA_repeat=0, AA_replace=False, AA_replace_AAs=replace_aa),
            MutationOpts(AA_deletion=True, AA_repeat=2, AA_replace=False, AA_replace_AAs=replace_aa),]
    calcu_peptide_mutations(pep, opts, mass=True, task_pool=None, batch_size=10000, result_path=None)