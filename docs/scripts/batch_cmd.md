# Batch Command Execution

The `batch_cmd` script allows you to execute commands on multiple files that match specific criteria.

## Usage

```bash
python -m mbapy.scripts.batch_cmd [options]
```

Or using mbapy-cli:

```bash
mbapy-cli batch_cmd [options]
```

## Options

- `-t, --type`: File extensions to match (space-separated). Example: `txt csv pdf`
- `-n, --name`: Sub-string in file names to match
- `-i, --input`: Input file path or directory path
- `-r, --recursive`: Enable recursive search in directories
- `-c, --cmd`: Command to execute. Use `%s` as placeholder for file path
- `-d, --use-dir`: Pass directory path instead of file path
- `-cd, --change-dir`: Change to file's directory before executing command

## Examples

### Copy all .txt files to backup directory

```bash
python -m mbapy.scripts.batch_cmd -t txt -i /path/to/files -c "cp %s /backup/"
```

### Delete all files containing "temp" in their name

```bash
python -m mbapy.scripts.batch_cmd -n temp -i /path/to/files -c "rm %s"
```

### Convert all images to PNG format recursively

```bash
python -m mbapy.scripts.batch_cmd -t jpg jpeg -i /path/to/images -r -c "convert %s %s.png"
```

### Execute command in each file's directory

```bash
python -m mbapy.scripts.batch_cmd -t py -i /path/to/scripts -cd -c "python %s"
```

### Process files in their respective directories

```bash
python -m mbapy.scripts.batch_cmd -t csv -i /data -cd -d -c "python process.py"
```

### Interactive mode (command prompt will appear)

```bash
python -m mbapy.scripts.batch_cmd -t txt -i /path/to/files
```

## How It Works

1. The script searches for files matching the specified criteria (extension and/or name pattern)
2. For each matched file, it executes the specified command
3. The `%s` placeholder in the command is replaced with:
   - File path (default)
   - Directory path (if `-d` flag is used)
4. If `-cd` flag is used, the script changes to the file's directory before executing the command

## Notes

- The script uses `tqdm` to show progress during execution
- All commands are executed using `os.system()`
- File paths are automatically cleaned and normalized
- The script returns the list of processed file paths

## See Also

- [cp](cp.md) - Copy files
- [mv](mv.md) - Move files
- [rm](rm.md) - Remove files