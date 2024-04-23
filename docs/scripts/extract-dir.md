<!--
 * @Date: 2024-04-22 11:33:55
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-04-22 11:34:09
 * @Description: 
-->
*`Kimi` generated*.

## Command: `extract-dir`
### Introduction
The `extract-dir` script is designed to move files with specific suffixes or substrings in their names to the root directory. This utility is particularly useful for organizing files in a hierarchical structure or consolidating files for easier access.
### Parameters
- `-t`, `--type`: A comma-separated list of file formats to target for moving. Defaults to an empty string, which implies no specific file type.
- `-n`, `--name`: A substring of the file name to target for moving. Defaults to an empty string, indicating no specific name substring.
- `-i`, `--input`: The path to the files or directory from which to start the search. Defaults to the current directory (`.`).
- `-r`, `--recursive`: A flag that, when set, enables recursive search through the directory structure. Defaults to `False`.
- `-j`, `--join-str`: The string to use when joining file names. Defaults to a space (`' '`).

### Behavior
The script performs the following actions:
1. Parses the command-line arguments provided by the user.
2. Cleans and validates the input path.
3. Searches for files matching the specified criteria within the given directory (and recursively, if enabled).
4. Prompts the user for confirmation before proceeding with the file moves.
5. Moves the matched files to the root directory, preserving their relative paths from the input directory.

### Notes
- The script uses the `argparse` library for command-line argument parsing.
- The `tqdm` library is utilized to display a progress bar during the file moving process.
- The script includes basic error handling and will skip files that cannot be moved, reporting the error to the user.
- The script is designed to be run from the command line and is not intended for interactive use within a Python environment.

### Examples
To move all `.txt` and `.pdf` files with the name containing "report" from the directory `/home/user/documents` to the root of that directory, recursively, you would use the following command:
```bash
mbapy-cli extract-dir -i /home/user/documents -t txt,pdf -n report -r
```

To move all `.jpg` files from the current directory to the root without recursion, you would use:
```bash
mbapy-cli extract-dir -t jpg
```