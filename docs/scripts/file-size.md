<!--
 * @Date: 2024-04-22 11:36:17
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-04-22 11:36:39
 * @Description: 
-->
## Command: `file-size`

### Overview
This Python script utilizes the `argparse` library to parse command-line arguments and perform file counting operations based on specific criteria. It can count files with a given suffix or a substring in the name within a specified directory, and provides detailed statistics about the file types and sizes.

### Parameters

- `-i`, `--input`: The path to the source files or directory. Defaults to the current directory if not specified.
- `-t`, `--type`: A comma-separated list of file formats to count. Defaults to an empty list, which means all file types will be counted.
- `-n`, `--name`: A substring of the file name to filter files by. Defaults to an empty string, which means no name filtering will be applied.
- `-r`, `--recursive`: A flag that, when set, enables recursive search through directories.
- `--enable-case`: A flag that, when set, makes the file counting case-sensitive.
- `--sort-by-name`: A flag that, when set, sorts the result by file name instead of by size.

### Behavior
The script performs the following actions:
1. Parses the provided command-line arguments.
2. Cleans and validates the input path.
3. Retrieves a list of file paths that match the specified criteria.
4. Counts the number of files and accumulates their sizes for each file type.
5. Sorts the results by file name or size, depending on the `--sort-by-name` flag.
6. Prints detailed statistics about each file type, including the total number of files, total size, and the proportion of each type relative to the overall count and size.

### Notes
- The script defaults to counting all file types if no file types are specified.
- The script defaults to a non-recursive and case-insensitive search if the respective flags are not set.
- The script uses the `tqdm` library for progress bars and `mbapy` utilities for file operations and error handling.

### Examples

To count all files in the current directory and its subdirectories, ignoring the case of file names, and sort the results by file size, you can use the following command:

```bash
mbapy-cli file-size --recursive --enable-case --sort-by-name
```

To count only `.txt` and `.pdf` files with the name containing "report" in a specified directory, and sort the results by file name, use:

```bash
mbapy-cli file-size -i /path/to/directory -t txt,pdf -n report --sort-by-name
```

This command will output the count and size statistics for `.txt` and `.pdf` files with "report" in their names within the specified directory, sorted alphabetically by file name.