
This documentation provides a comprehensive overview of the `AnimoAcid` and `Peptide` classes, including their attributes, methods, parameters, return values, and usage examples.
`Kimi` generated.

# Class

## AnimoAcid

### Class Overview
The `AnimoAcid` class is designed to represent an amino acid with its respective properties, such as molecular weight, molecular formula, and protecting groups.

### Members

#### aa_mwd
A dictionary containing the molecular weights of amino acids.

#### aa_3to1
A dictionary mapping three-letter amino acid codes to their one-letter codes.

#### aa_1to3
The inverse of `aa_3to1`, mapping one-letter codes to three-letter codes.

#### pg_mwd
A dictionary containing the molecular weights of protecting groups.

#### mfd
A dictionary containing the molecular formulas of amino acids and protecting groups.

#### all_mwd
A dictionary combining the molecular weights of amino acids and protecting groups.

### Methods

#### __init__(self, repr: str, aa_repr_w: int = 3) -> None
##### Method Overview
Initializes an instance of the class with the given representation string.

##### Parameters
- `repr (str)`: The representation string of a peptide to initialize the instance with.
- `aa_repr_w (int, optional)`: The width of the amino acid representation (1 or 3 letters). Defaults to 3.

##### Exceptions
- Raises `ValueError` if the representation string is not valid according to the amino acid and protecting group definitions.

##### Examples
```python
aa = AnimoAcid('H-Ala-OH', 3)
```

#### check_is_aa(aa: str)
##### Method Overview
Checks if a given string is a valid amino acid or protecting group representation.

##### Parameters
- `aa (str)`: The string to check.

##### Returns
- `int`: Returns 1 if the string is a valid one-letter amino acid code, 3 if it's a valid three-letter code, or 0 otherwise.

##### Examples
```python
result = AnimoAcid.check_is_aa('A')  # Returns 1
result = AnimoAcid.check_is_aa('Ala')  # Returns 3
```

#### make_pep_repr(self, is_N_terminal: bool = False, is_C_terminal: bool = False, repr_w: int = 3, include_pg: bool = True)
##### Method Overview
Generates a PEP representation of the amino acid sequence.

##### Parameters
- `is_N_terminal (bool, optional)`: Whether the sequence is at the N-terminus. Defaults to False.
- `is_C_terminal (bool, optional)`: Whether the sequence is at the C-terminus. Defaults to False.
- `repr_w (int, optional)`: The width of the amino acid representation (1 or 3 letters). Defaults to 3.
- `include_pg (bool, optional)`: Whether to include protecting groups in the representation. Defaults to True.

##### Returns
- `str`: The PEP representation of the amino acid sequence.

##### Examples
```python
pep_repr = aa.make_pep_repr(is_N_terminal=True, is_C_terminal=True, repr_w=3)
```

#### calcu_mw(self, expand_mw_dict: Dict[str, float] = None)
##### Method Overview
Calculates the molecular weight of the peptide sequence.

##### Parameters
- `expand_mw_dict (Dict[str, float], optional)`: A dictionary containing the molecular weights of additional protecting groups. Defaults to None.

##### Returns
- `float`: The calculated molecular weight of the peptide sequence.

#### get_molecular_formula_dict(self)
##### Method Overview
Returns a dictionary containing the molecular formula of the protein sequence.

##### Returns
- `dict`: A dictionary containing the molecular formula.

#### get_molecular_formula(self, molecular_formula_dict: Dict[str, int] = None)
##### Method Overview
Generates the molecular formula from a given dictionary of element symbols and their counts.

##### Parameters
- `molecular_formula_dict (Dict[str, int], optional)`: A dictionary containing element symbols as keys and their counts as values. Defaults to None.

##### Returns
- `str`: The molecular formula generated from the dictionary.

#### calcu_mass(self, molecular_formula: str = None, molecular_formula_dict: Dict[str, int] = None)
##### Method Overview
Calculates the mass of a molecule based on its molecular formula.

##### Parameters
- `molecular_formula (str, optional)`: The molecular formula of the molecule. Defaults to None.
- `molecular_formula_dict (Dict[str, int], optional)`: A dictionary representing the molecular formula of the molecule. Defaults to None.

##### Returns
- `float`: The calculated mass of the molecule.

#### copy(self)
##### Method Overview
Creates a copy of the current instance.

##### Returns
- A copy of the `AnimoAcid` object.

## Peptide

### Class Overview
The `Peptide` class represents a peptide, which is a sequence of `AnimoAcid` objects.

### Methods

#### __init__(self, repr: str, aa_repr_w: int = 3) -> None
##### Method Overview
Initializes a Peptide object by splitting the input string and creating a list of AnimoAcid objects.

##### Parameters
- `repr (str)`: The representation string of the peptide.
- `aa_repr_w (int, optional)`: The width of the amino acid representation (1 or 3 letters). Defaults to 3.

##### Exceptions
- Raises `ValueError` if the representation string is not valid according to the amino acid and protecting group definitions.

#### flatten(self, inplace: bool = False)
##### Method Overview
Flattens the list of AnimoAcid objects into a single list.

##### Parameters
- `inplace (bool, optional)`: If True, makes the change in place and returns self, otherwise returns the changed sequence only. Defaults to False.

##### Returns
- The flattened list of AnimoAcid objects.

#### repr(self, repr_w: int = 3, include_pg: bool = True, include_dash: bool = True)
##### Method Overview
Returns a string representation of the Peptide object by joining the representations of each AnimoAcid object in the sequence.

##### Parameters
- `repr_w (int, optional)`: The width of the amino acid representation (1 or 3 letters). Defaults to 3.
- `include_pg (bool, optional)`: Whether to include protecting groups in the representation. Defaults to True.
- `include_dash (bool, optional)`: Whether to include dashes in the representation. Defaults to True.

##### Returns
- `str`: The string representation of the Peptide object.

#### get_molecular_formula_dict(self)
##### Method Overview
Returns a dictionary representing the molecular formula of the Peptide object by summing the molecular formulas of each AnimoAcid object in the sequence.

##### Returns
- `dict`: A dictionary representing the molecular formula.

#### get_molecular_formula(self, molecular_formula_dict: Dict[str, int] = None)
##### Method Overview
Returns a string representation of the molecular formula of the Peptide object by joining the elements of the molecular_formula_dict dictionary.

##### Parameters
- `molecular_formula_dict (Dict[str, int], optional)`: A dictionary representing the molecular formula, where the keys are the element symbols and the values are the corresponding counts. Defaults to None.

##### Returns
- `str`: The molecular formula as a string.

#### calcu_mw(self, expand_mw_dict: Dict[str, float] = None)
##### Method Overview
Calculates the molecular weight of the Peptide object by summing the molecular weights of each AnimoAcid object in the sequence.

##### Parameters
- `expand_mw_dict (Optional[Dict[str, float]])`: A dictionary containing the molecular weights of the protected group. Defaults to None.

##### Returns
- `float`: The calculated molecular weight of the peptide.

#### calcu_mass(self, molecular_formula: str = None, molecular_formula_dict: Dict[str, int] = None)
##### Method Overview
Calculates the mass of the Peptide object by calling the calcu_mass method of the first AnimoAcid object in the sequence.

##### Parameters
- `molecular_formula (str, optional)`: The molecular formula of the molecule. Defaults to None.
- `molecular_formula_dict (Dict[str, int], optional)`: The dictionary representation of the molecular formula. Defaults to None.

##### Returns
- `float`: The calculated mass of the molecule.

#### copy(self)
##### Method Overview
Creates a copy of the Peptide object by creating a new Peptide object and copying the list of AnimoAcid objects.

##### Returns
- A copy of the `Peptide` object.

## Examples

```python
# Example usage of the Peptide class
pep = Peptide('H-Cys(Trt)-G(OtBu)', 3)
print(pep.repr(3, True, True))
print(pep.get_molecular_formula())
print(pep.calcu_mw())
```