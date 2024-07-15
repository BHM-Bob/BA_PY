<!--
 * @Date: 2024-04-22 19:58:07
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-07-15 12:52:38
 * @Description: 
-->
*`Kimi` generated*.

## Command: mv

### Introduction
The `mv` command is a utility script that allows for the movement of files with specific suffixes or substrings in their names from a source directory to a destination directory. It provides options for recursive searching and filtering files by type and name.

### Parameters

- `-i`, `--input`: The path to the source directory or file. This is the location from which files will be moved.
- `-o`, `--output`: The path to the destination directory. This is where the files will be moved to.
- `-t`, `--type`: A comma-separated list of file types to move. The file types should be specified without the dot (e.g., 'jpg' for JPEG images).
- `-n`, `--name`: A substring of the file names to move. Only files with names containing this substring will be moved.
- `-r`, `--recursive`: A flag that, when set, tells the script to search for files recursively within the input directory.
- `--just-name`: A flag that, when set, tells the script to only move the file names, not the entire path(sub-path).

### Behavior
The script first checks if the input is a single file and the output is not a directory. If so, it moves the file directly. If the input is a directory or multiple files need to be moved, the script uses the `get_paths_with_extension` function to generate a list of file paths that match the specified criteria. It then iterates over these paths, creating any necessary directories in the destination path and moving each file.

### Notes
- The script uses the `shutil` module to move files and the `os` module to handle file paths and directories.
- If the destination directory does not exist for a particular file, the script will create it before moving the file.
- The script provides feedback on the console about the number of files that will be moved and skips any files that it cannot move, reporting these in the console.

### Examples
To move all JPEG files from a directory to another:

```bash
mbapy-cli mv -i /path/to/source -o /path/to/destination -t JPG
```

To move all files with a specific substring in their names, non-recursively:

```bash
mbapy-cli mv -i /path/to/source -o /path/to/destination -n mysubstring
```

To move all files of a specific type, recursively searching the source directory:

```bash
mbapy-cli mv -i /path/to/source -o /path/to/destination -t docx -r
```