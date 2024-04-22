<!--
 * @Date: 2024-04-22 19:57:19
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-04-22 19:57:31
 * @Description: 
-->
*`Kimi` generated*.

## Command: plot-hplc

### Introduction
The `plot-hplc` command is a Python script designed to process and visualize high-performance liquid chromatography (HPLC) data. It is capable of reading data files, typically in the ARW format exported by Waters systems, and generating plots of the absorbance over time.

### Parameters

- `-i`, `--input`: The directory containing the data files. Defaults to the current directory if not specified.
- `-s`, `--system`: The HPLC system used. Currently, only 'waters' is supported and it's the default.
- `-r`, `--recursive`: If set, the script will search for data files recursively within the input directory.
- `-merge`: If set, the script will merge data from multiple files into a single plot.
- `-o`, `--output`: The directory where the output plots will be saved. Defaults to the same directory as the input if not specified.
- `--min-peak-width`: A float value to filter peaks by their minimum width in the HPLC plot.
- `-xlim`: A string to set the x-axis limit for the plot, in the format "xmin,xmax".
- `-colors`: A string specifying the color of the plot lines.
- `-labels`: A string to label peaks, formatted as "mass,label[,color];...".
- `-flabels`, `--file-labels`: A string to label files, formatted as "label,color;...".
- `-labels-eps`: A float value representing the epsilon for recognizing labels.
- `-expand`: A float value to expand the x and y axes of the plot.
- `-lpos`, `-legend-pos`: The position of the legend on the plot, either as a string like "upper center" or as coordinates as two floats.
- `-lposbbox1`, `-legend-pos-bbox1`: The bbox anchor for the legend position.
- `-lposbbox2`, `-legend-pos-bbox2`: The second bbox anchor for the legend position.

### Behavior
The script will read the specified HPLC data files, process them, and create plots of the absorbance over time. If the `merge` option is used, it will combine data from all files into a single plot. The plots will be saved in the specified output directory.

### Notes
- The script requires the `matplotlib`, `numpy`, `pandas`, and `scipy` libraries to be installed.
- The script assumes that the HPLC data files are in the ARW format if the `--system` is set to 'waters'.
- The script uses environment variables `MBAPY_AUTO_IMPORT_TORCH` and `MBAPY_FAST_LOAD` to control certain behaviors, which are set at the beginning of the script.

### Examples
To plot HPLC data from the current directory and save the output in a directory named 'plots':

```bash
mbapy-cli hplc plot-hplc -i . -o plots
```

To plot HPLC data from a specific directory, recursively search within that directory, and merge the plots:

```bash
mbapy-cli hplc plot-hplc -i /path/to/data -r -merge -o /path/to/output
```

To plot HPLC data with specific labels and colors:

```bash
mbapy-cli hplc plot-hplc -i /path/to/data -labels "1000,Pep1;1050,Pep2" -flabels "228,blue;304,red" -o plots
```