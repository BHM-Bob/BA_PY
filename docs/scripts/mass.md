<!--
 * @Date: 2024-04-23 18:58:04
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-07-15 13:07:24
 * @Description: 
-->
# Command: plot-mass

## Introduction
The `plot-mass` command is designed to plot mass spectrum data from files within a specified directory. It supports various parameters to customize the plot, such as filtering criteria, output directory, and plot appearance.

## Parameters

- `-d`, `--dir`: The directory containing the txt files to be processed. Defaults to the current directory if not specified.
- `-r`, `--recursive`: A flag to indicate whether to search the directory recursively for files. Defaults to `False`.
- `-o`, `--output`: The output directory or file path where the plots will be saved. If not specified, it will default to the input directory.
- `-m`, `-mass`: A flag to plot the mass data instead of the mass/charge ratio. Defaults to `False`.
- `-min`, `--min-height`: The minimum height filter for peak list plots.
- `-minp`, `--min-height-percent`: The minimum height percentage to the highest in mass/charge plots.
- `--min-peak-width`: The minimum width filter for peaks in the Mass/Charge plot.
- `-xlim`: The x-axis limit for the plot.
- `--col`, `--color`: The color of the plot.
- `--marker-size`: The size of the markers on the plot.
- `-labels`, `--labels`: Labels for the peaks, in the format of "mass,label,color;mass,label,...".
- `--labels-eps`: The epsilon value for recognizing labels.
- `-sf`, `--show-fig`: A flag to automatically show the figure after plotting.
- `-lposbbox`, `--legend-bbox`: The bounding box for the legend position.
- `-mp`, `--multi-process`: The number of processes to use for parallel plotting.

## Behavior
The command will load mass data from the specified directory, apply any specified filters, and generate plots for each file. The plots will be saved to the specified output location, and optionally displayed if the `--show-fig` flag is set.

## Notes
- Ensure that the directory path provided exists and contains the expected data files.
- The output directory will be created if it does not exist.
- The use of multi-processing can significantly speed up the plotting process, especially for large datasets.

## Example
```bash
mbapy-cli plot-mass -d /root/to/data -r -o /root/to/output --mass --min-height 100 --show-fig
```

This example command will plot mass spectrum data from the `/root/to/data` directory, recursively including all subdirectories, and save the plots to `/path/to/output`. It will plot the mass data, filter out peaks with a height less than 100, and display each plot after it is generated.

---

# Command: explore-mass

## Introduction
The `explore-mass` command provides an interactive GUI for exploring mass spectrum data. It allows users to load data, apply filters, and visualize the results in real-time.

## Parameters
The parameters for `explore-mass` are largely the same as those for `plot-mass`, with the addition of GUI-specific settings.

## Behavior
The command will launch a GUI that enables users to load mass data, adjust filtering parameters, and plot the results interactively. Users can switch between different datasets, save plots, and customize the appearance of the plots.

## Notes
- The GUI is designed for interactive exploration and may not be suitable for batch processing of large datasets.
- Real-time plotting allows for quick adjustments and visualization of the effects of different filter settings.

## Example
```bash
mbapy-cli explore-mass -d /root/to/data --recursive
```

This example command will open the GUI for exploring mass spectrum data located in `/root/to/data`, including all subdirectories. Users can then interactively load, filter, and plot the data as needed.
