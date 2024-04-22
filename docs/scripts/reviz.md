<!--
 * @Date: 2024-04-22 20:03:29
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-04-22 20:19:38
 * @Description: 
-->
*`Kimi` generated*.

## Command: reviz
### Introduction
The `reviz` command is designed to launch a Visdom service and re-visualize data from a JSON record file. It is particularly useful for revisiting and re-analyzing previously saved visualization data in a Visdom environment.
### Parameters
- `-i`, `--input`: The path to the input JSON record file or the directory containing the file.

### Behavior
The command performs the following actions:
1. Parses the command line arguments to obtain the input file path.
2. Cleans the input path to ensure it is a valid system path.
3. Checks if the input file exists; if not, it prints an error message and exits.
4. If the input is a directory, the command searches for a JSON file with 'record' in its name or any JSON file and sets it as the input.
5. Launches the Visdom service using the parent directory name of the input file as the environment.
6. Initiates re-visualization of the data from the JSON record file using `re_viz_from_json_record`.

### Notes
- The command relies on the Visdom service for visualization, which is launched within the environment named after the input file's parent directory.
- It is assumed that the input JSON file contains the necessary data for re-visualization.

### Example
```
mbapy-cli reviz --input "path/to/your/record.json"
```