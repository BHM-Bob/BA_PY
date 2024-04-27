<!--
 * @Date: 2024-04-22 11:12:08
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-04-22 20:17:36
 * @Description: 
-->
*`Kimi` generated*.

## Command: avif

### Introduction
This Python script is designed to convert image files from JPEG format to AVIF format. It utilizes the `argparse` library to accept command-line arguments and the `PIL` (Python Imaging Library) with `pillow_heif` for handling AVIF images.

### Parameters

- `-t`, `--to`: The format of the output file. Default is 'avif'. Other choices are 'heic' and 'jpg'.
- `-q`, `--quality`: The quality of the output file. Default is 85.
- `-i`, `--input`: The input file path or directory path. Default is the current directory ('.').
- `-r`, `--recursive`: A flag to indicate if the search for input files should be recursive. Default is False.
- `-o`, `--output`: The output file path or directory path. Default is the current directory ('.').
- `-rm`, `--remove-origin`: A flag to indicate if the original files should be removed after conversion. Default is False.
- `-ifmt`, `--input-format`: A comma-separated list of input file formats to search for. Default is 'jpg,jpeg,png,JPG,JPEG,PNG'.
- `-m`, `--multi-process`: The number of processes for parallel processing. Default is 4.
- `-b`, `--batch`: The number of batch size for a process. Default is 10.

### Behavior
The script will search for files with the specified input formats in the given input directory. It will then convert these files to the specified output format in the output directory. If the `--recursive` flag is set, the search will include subdirectories. If the `--remove-origin` flag is set, the original files will be deleted after successful conversion. The script supports multiprocessing to speed up the conversion process.

### Notes
- The script modifies the `Image.MAX_IMAGE_PIXELS` to allow loading of large images.
- It registers AVIF and HEIF openers with PIL to ensure compatibility with these formats.
- The `clean_path` function is used to normalize the paths provided as arguments.
- The `show_args` function is used to display the arguments that will be used by the script.
- The `transfer_img` function handles the conversion process, including copying or converting files and updating file sizes.

### Examples

To convert all JPEG files in the current directory to AVIF format in a new directory called 'avif', using a quality of 80 and removing the original files, the following command can be used:

```bash
mbapy-cli avif -i . -o avif -t avif -q 80 -rm -ifmt jpg
```

To perform the same operation recursively on a directory located at 'E:\My_Progs\z_Progs_Data_HC', using 2 processes, the command would be:

```bash
mbapy-cli avif -i "E:\My_Progs\z_Progs_Data_HC" -o "E:\My_Progs\z_Progs_Data_HC\avif" -r -t avif -q 80 -rm -ifmt jpg -m 2
```