<!--
 * @Date: 2024-04-22 20:00:44
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-04-22 20:06:41
 * @Description: 
-->
*`Kimi` generated*.

## Command: subval
### Introduction
The `subval` command is used to calculate the substitution value for a release test of resin in solid phase peptide synthesis (SPPS). It processes input absorbance and weight data to compute the average substitution value and plots a scatter plot with a linear regression model.
### Parameters
- `-a`, `-A`, `--absorbance`: Absorbance values (OD values) separated by commas, e.g., "0.503,0.533".
- `-m`, `-w`, `--weight`: Resin weights in milligrams separated by commas, e.g., "0.165,0.155".
- `-c`, `--coff`: A coefficient used in the calculation, which defaults to 16.4 if not provided.

### Behavior
The command converts input strings to float values, calculates the average substitution value, fits a linear regression model to the data, and plots the linear regression line on a scatter plot. It also prints the calculated average substitution value and the equation of the linear regression model with its R-squared value.

### Notes
- Ensure that the input data for absorbance and weight are correctly formatted as comma-separated values.
- The command uses the `matplotlib` and `seaborn` libraries for plotting, and `scikit-learn` for linear regression.

### Example
```
mbapy-cli peptide subval --absorbance 0.503,0.533 --weight 0.165,0.155
```

---

## Command: molecularweight | mw
### Introduction
The `molecularweight` command calculates the molecular weight (MW) of a peptide based on its amino acid sequence and a provided dictionary of weights for each amino acid.
### Parameters
- `-s`, `--seq`, `--seqeunce`, `--pep`, `--peptide`: The peptide sequence, e.g., "Fmoc-Cys(Acm)-Leu-OH".
- `-w`, `--weight`: A comma-separated list of amino acid weights, e.g., "Trt-243.34,Boc-101.13".
- `-m`, `--mass`: A flag to calculate the exact mass instead of the molecular weight (default is False).

### Behavior
The command creates a peptide object, calculates its molecular weight using the provided weights, and prints the result. If the `--mass` flag is used, it also prints the chemical formula and the exact mass.

### Notes
- The peptide sequence should be provided in a format that includes amino acid codes and protect groups.
- The weight of the terminal amino group (-H) is not included and should not be provided.

### Example
```
mbapy-cli peptide molecularweight --seq Fmoc-Cys(Acm)-Leu-OH --weight Trt-243.34,Boc-101.13
```

---

## Command: mutationweight | mmw
### Introduction
The `mutationweight` command calculates the molecular weight of each peptide mutation synthesized by SPPS.
### Parameters
- `-s`, `--seq`: The peptide sequence.
- `-w`, `--weight`: Amino acid weights and protect group weights.
- `--max-repeat`: The maximum number of times an amino acid can be repeated in the sequence (default is 1).
- `--disable-aa-deletion`: A flag to disable amino acid deletion in mutations (default is False).
- `-o`, `--out`: The output file path or directory where results will be saved. If not provided, results are printed to the console.
- `-m`, `--mass`: A flag to calculate the exact mass instead of the molecular weight (default is False).

### Behavior
The command performs mutations on the peptide sequence up to the maximum repeat allowed, extracts individual mutations, and calculates their molecular weights. It then prints the number of mutations found and details of each mutation, including their molecular weights and peptide sequences.

### Notes
- The output can be directed to a file or directory for further analysis or documentation.
- The command provides a comprehensive analysis of all possible mutations based on the input parameters.

### Example
```
mbapy-cli peptide mutationweight --seq Fmoc-Cys(Acm)-Leu-OH --weight Trt-243.34,Boc-101.13 --max-repeat 2 --out ./mutations.txt
```

---

## Command: letters
### Introduction
The `letters` command transfers the representation of amino acid letters between different width formats, such as from three-letter to one-letter representation.
### Parameters
- `-s`, `--seq`: The peptide sequence.
- `--src`, `--source-width`: The source representation width of amino acids, which can be either 1 or 3.
- `--trg`, `--target-width`: The target representation width of amino acids, which can be either 1 or 3.
- `--dpg`, `--disable-pg`: A flag to include or exclude protect groups in the target representation.
- `--ddash`, `--disable-dash`: A flag to include or exclude dash lines in the target representation.
- `-i`, `--input`: An input file containing peptide sequences, one per line (default is None).
- `-o`, `--out`: An output file path or directory where the converted representations will be saved (default is None).

### Behavior
The command reads peptide sequences from the provided input or command line argument, converts the amino acid representations according to the specified source and target widths, and prints or saves the converted representations.

### Notes
- This command is useful for converting peptide sequences between different notational conventions.
- The output can be saved to a file or directory for further use.

### Example
```
mbapy-cli peptide letters --seq Fmoc-Cys(Acm)-Leu-OH --src 3 --trg 1 --out ./converted.txt
```