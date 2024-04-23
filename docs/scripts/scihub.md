<!--
 * @Date: 2024-04-22 20:05:27
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-04-22 20:14:11
 * @Description: 
-->
*`Kimi` generated*.

## Command: scihub
### Introduction
This script is designed to download academic papers from SCI-HUB using a RIS file that contains the references. It can also download the references of the main papers if the `--ref` flag is set. The script provides functionality to save the download progress and resume later if needed.
### Parameters
- `-i`, `--ris`: The file path to the RIS file containing the references of the papers to download.
- `-o`, `--out`: The directory where the downloaded papers will be saved.
- `-r`, `--ref`: A flag to enable the download of referenced papers.
- `-l`, `--log`: A flag to enable logging.

### Behavior
The script performs the following actions:
1. Parses the command-line arguments to configure the download settings.
2. Initializes a `Record` object to keep track of downloaded papers and their references.
3. Sets up logging and exception handling to manage errors and save progress.
4. Parses the RIS file to extract paper information.
5. Downloads the main papers and, if enabled, their references from SCI-HUB.
6. Handles exceptions by attempting to save the current session and logging the error.
7. Uses a progress bar to display the download status.
8. Allows for manual interruption of the process with the option to save the session.

### Notes
- The script requires the `requests` and `tqdm` libraries for HTTP requests and progress bar display, respectively.
- It uses the `Record` class to manage and store information about downloaded papers and their references.
- The `download_by_scihub` function from the `paper` module is used to download individual papers.
- The script includes random delays between downloads to avoid being blocked by SCI-HUB.
- The `handle_exception` function is used as a custom exception handler to manage exceptions and save the progress when they occur.

### Example
```
mbapy-cli --ris "path/to/references.ris" --out "downloaded/papers" --ref --log
```

### Additional Information
Please replace `"path/to/references.ris"` with the actual path to your RIS file and `"downloaded/papers"` with your desired output directory when using the command. Ensure that you have the necessary permissions to write to the output directory and that the RIS file is formatted correctly.