
*`Kimi` generated*.

## Command: rm
### Introduction
This utility command is designed to delete files with specific suffixes or substrings in their names. It provides options to specify file types, substrings in file names, the input directory, and whether to perform a recursive search within the directory.
### Parameters
- `-t`, `--type`: A comma-separated list of file types (suffixes) to remove. Defaults to an empty string, which means no specific file type will be targeted.
- `-n`, `--name`: A substring of the file names that should be matched for deletion. Defaults to an empty string, which means all file names will be considered.
- `-i`, `--input`: The path to the files or directory from which to delete files.
- `-r`, `--recursive`: A flag that, when set, indicates that the search for files to delete should be recursive. Defaults to `False`.

### Behavior
The command operates as follows:
1. It parses the provided arguments to determine the file types, name substring, input path, and whether the operation should be recursive.
2. It cleans and validates the input path.
3. It searches for files that match the specified criteria within the given directory (and recursively, if specified).
4. It prompts the user for confirmation before proceeding with the deletion.
5. If the user confirms, it deletes the matched files and provides feedback on the progress.
6. In case of any errors during deletion, it logs the error and continues with the next file.

### Notes
- This command should be used with caution as it will permanently delete files.
- It is recommended to have a backup of important files before using this utility.
- The command utilizes the `tqdm` library for progress indication and `mbapy.file` for file path handling.

### Example
```
mbapy-cli rm -i "path/to/directory" -t "txt,py" -n "backup" -r
```