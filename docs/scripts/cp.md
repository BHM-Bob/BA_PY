<!--
 * @Date: 2024-04-22 11:29:49
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-04-22 20:18:16
 * @Description: 
-->
*`Kimi` generated*.

## Command: cp

### Overview
The script provided is a Python utility designed to move files with specific suffixes or substrings in their names from a source directory to a destination directory. It uses the `argparse` library to parse command-line arguments and provides a user-friendly interface for file manipulation tasks.

### Parameters

- `-i`, `--input`: The path to the source files or directory.
- `-o`, `--output`: The path to the destination directory where files will be moved.
- `-t`, `--type`: A comma-separated list of file formats to consider for moving. Defaults to an empty string, which means no specific file type filtering.
- `-n`, `--name`: A substring of the file name to match for moving files. Defaults to an empty string, indicating no substring filtering.
- `-r`, `--recursive`: A flag that, when set, enables recursive search through the input directory. Defaults to `False`.

### Behavior
The script performs the following actions:
1. Parses the command-line arguments.
2. Cleans and validates the input and output paths.
3. If a single file is specified and the output is not a directory, it copies the file directly.
4. Retrieves a list of file paths that match the specified criteria (file type, name substring, and recursive search).
5. Copies each file to the destination directory, preserving the directory structure relative to the input path.

### Notes
- The script uses the `tqdm` library to show progress bars for the file copying process.
- Error handling is implemented to catch issues during the file copying process and log them without stopping the script.
- The script is designed to be run from the command line and does not have a graphical user interface.

### Examples

To move all `.jpg` files from the directory `E:\\1` to `E:\\2`, the following command can be used:

```bash
mbapy-cli cp -i "E:\1" -o "E:\2" -t JPG
```

To move all files with the substring "report" in their names from `E:\\1` to `E:\\2` recursively, use:

```bash
mbapy-cli cp -i "E:\1" -o "E:\2" -n report -r
```