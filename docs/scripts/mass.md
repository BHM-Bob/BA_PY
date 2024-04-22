
*`Kimi` generated*.

## plot-mass
### Brief Description
The `plot-mass` command is designed to process and visualize mass spectrometry data from text files. It reads text files, interprets the content based on the header information, and creates various plots such as base peak, absorbance, peak list, and mass-charge plots. The command allows for customization of the plots through command-line arguments, including setting the directory for input and output, recursive search, plot colors, and labels for specific mass values.

### Parameters
- `-d`, `--dir`: The directory containing the text files to be processed. Defaults to the current directory if not specified.
- `-r`, `--recursive`: Flag to search the input directory recursively for text files.
- `-o`, `--output`: The directory where the output plots and CSV files will be saved. Defaults to the input directory if not specified.
- `--use-peaks-cache`: A flag to use a cached version of peaks for faster plotting.
- `-m`, `-.mass`: A flag to plot mass instead of mass/charge.
- `-min`, `--min-height`: The minimum height for filtering data in peak list plots.
- `-minp`, `--min-height-percent`: The minimum height percent relative to the highest value for filtering data in mass/charge plots.
- `--min-peak-width`: The minimum width for filtering peaks in mass/charge plots.
- `-xlim`: The x-axis limit for the plots, input as a comma-separated string like "200,2000".
- `-col`, `--color`: The color used for plotting the data.
- `-labels`, `--labels`: A string defining labels for specific masses, formatted as "mass,label[,color];mass,label".
- `-labels-eps`: The epsilon value for label recognition.
- `-expand`: The factor by which to expand the x and y axes of the plots.
- `-lpos`, `--legend-pos`: The position of the legend in the plot, can be a string or a pair of floats.
- `-lposbbox1`, `--legend-pos-bbox1`: The bbox_to_anchor value for the legend position.
- `-lposbbox2`, `--legend-pos-bbox2`: The second bbox_to_anchor value for the legend position.

### Behavior
The script performs the following actions:
1. Parses command-line arguments to configure the behavior of the script.
2. Searches for text files in the specified directory, recursively if the `--recursive` flag is set.
3. Reads the text files and categorizes them based on their content.
4. Generates plots for each recognized file type and saves them to the specified output directory.
5. Optionally saves a CSV file for each processed file.

### Notes
- The script uses several libraries including `scipy`, `numpy`, `pandas`, and `matplotlib` for data processing and visualization.
- The script defines several plotting functions (`plot_mass_plot_basepeak`, `plot_mass_plot_absorbance`, etc.) for different types of mass spectrometry data.
- The `plot_mass` function handles the main logic of processing files and creating plots.
- Error handling is done through a custom `put_err` function from the `mbapy` module.

### Examples
To plot mass spectrometry data from the current directory and save the plots in a directory called "plots":
```bash
mbapy-cli mass plot-mass -d . --output plots
```

To plot data recursively from a directory called "mass_data" with specific labels and a custom color:
```bash
mbapy-cli mass plot-mass -d mass_data -r -labels "1000,Pep1;1050,Pep2" -col red
```

# Functions

## plot_mass_load_file
### Brief Description
`plot_mass_load_file` is a function that reads a text file and converts its content into a pandas DataFrame. Depending on the header and the content of the file, it interprets the data as either base peak, absorbance, mass-charge, or peak list data and assigns a 'content_type' attribute to the DataFrame accordingly.

### Parameters
- `path`: A `Path` object representing the file to be read.

### Returns
- A pandas DataFrame with the content of the file, including a 'content_type' attribute that indicates the type of data (base peak, absorbance, mass-charge, or peak list).

### Behavior
- Reads the file line by line, splitting the lines based on the tab character.
- The first line is considered as the header with column names.
- The subsequent lines are split into columns based on the header.
- Depending on the number of columns and their names, the function determines the type of mass spectrometry data and sets the 'content_type' attribute.

### Notes
- The function uses pandas for data manipulation.
- It assumes that the text file has a specific structure with either two or ten columns, which helps in determining the 'content_type'.

### Examples
```python
df = plot_mass_load_file(Path('/path/to/mass_spec_data.txt'))
print(df)
```

## plot_mass_plot_basepeak
### Brief Description
`plot_mass_plot_basepeak` generates a plot for base peak data from mass spectrometry, showing the intensity of the signal over time.

### Parameters
- `name`: A string representing the name of the plot, typically the filename without the extension.
- `base_peak`: A pandas DataFrame containing the base peak data with 'Time' and 'Intensity' columns.
- `args`: An object containing arguments that define the plot's appearance, such as color and output directory.

### Behavior
- Creates a matplotlib figure and axis for plotting.
- Plots the 'Time' versus 'Intensity' data on a log-scaled y-axis.
- Customizes the title, axis labels, and tick sizes based on the provided arguments.
- Saves the plot as a PNG file in the specified output directory with a high DPI.

### Notes
- The function uses matplotlib for plotting.
- It assumes that the `base_peak` DataFrame has been properly formatted with a 'content_type' of 'base peak'.

### Examples
```python
plot_mass_plot_basepeak('example_base_peak', df_base_peak, args)
```

## plot_mass_plot_absorbance
### Brief Description
`plot_mass_plot_absorbance` is a function that generates a plot for absorbance data from mass spectrometry, showing the absorbance values over time.

### Parameters
- `name`: A string representing the name of the plot, typically the filename without the extension.
- `df`: A pandas DataFrame containing the absorbance data with 'Time' and 'Absorbance' columns.
- `args`: An object containing arguments that define the plot's appearance, such as color and output directory.

### Behavior
- Creates a matplotlib figure and axis for plotting.
- Plots the 'Time' versus 'Absorbance' data.
- Customizes the plot with a title, axis labels, and tick sizes.
- Sets the x-axis limit if provided in `args.xlim`.
- Saves the plot as a PNG file in the specified output directory with a high DPI.

### Notes
- The function uses matplotlib for plotting.
- It assumes that the `df` DataFrame has been properly formatted with a 'content_type' of 'absorbance'.

### Examples
```python
plot_mass_plot_absorbance('example_absorbance', df_absorbance, args)
```

## plot_mass_plot_peaklist
### Brief Description
`plot_mass_plot_peaklist` creates a detailed plot for peak list data from mass spectrometry, highlighting individual peaks with their respective intensities.

### Parameters
- `name`: A string representing the name of the plot, typically the filename without the extension.
- `df`: A pandas DataFrame containing the peak list data with columns for mass/charge, height, charge, and other metadata.
- `args`: An object containing arguments that define the plot's appearance and behavior, such as labels, color, and output directory.

### Behavior
- Filters the DataFrame based on user-defined x-axis limits and monoisotopic peaks.
- Uses vertical lines and scatter plots to represent peaks, with optional labels for specific masses.
- Customizes the plot with a title, axis labels, and tick sizes.
- Applies log scaling to the y-axis and adjusts axis limits based on the 'expand' argument.
- Saves the plot as a PNG file in the specified output directory with a high DPI.

### Notes
- The function uses matplotlib for plotting.
- It assumes that the `df` DataFrame has been properly formatted with a 'content_type' of 'peak list'.

### Examples
```python
plot_mass_plot_peaklist('example_peak_list', df_peak_list, args)
```

## plot_mass_plot_masscharge
### Brief Description
`plot_mass_plot_masscharge` is responsible for generating a plot for mass-charge data from mass spectrometry, identifying and visualizing peaks in the data.

### Parameters
- `name`: A string representing the name of the plot, typically the filename without the extension.
- `df`: A pandas DataFrame containing the mass-charge data with columns for mass/charge and intensity.
- `args`: An object containing arguments that define the plot's appearance and behavior, such as color, min peak width, and output directory.

### Behavior
- Searches for peaks in the intensity data using a continuous wavelet transform (CWT).
- Filters the DataFrame based on user-defined x-axis limits and minimum height criteria.
- Saves the filtered DataFrame as a CSV file.
- Uses vertical lines to represent peaks, with optional labels for specific masses.
- Customizes the plot with a title, axis labels, and tick sizes.
- Applies log scaling to the y-axis and adjusts axis limits based on the 'expand' argument.
- Saves the plot as a PNG file in the specified output directory with a high DPI.

### Notes
- The function uses scipy for peak detection and matplotlib for plotting.
- It assumes that the `df` DataFrame has been properly formatted with a 'content_type' of 'mass-charge'.

### Examples
```python
plot_mass_plot_masscharge('example_mass_charge', df_mass_charge, args)
```

Each of these functions is designed to handle specific types of mass spectrometry data and produce visualizations that are commonly used in the analysis of such data. They all follow a similar pattern of data filtering, plotting, customization, and saving the output.