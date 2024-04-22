<!--
 * @Date: 2024-04-22 20:05:22
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-04-22 20:19:52
 * @Description: 
-->

*`Kimi` generated*.

## Command: scihub_selenium
### Introduction
This script is a command-line utility for downloading academic papers from SCI-HUB based on a provided RIS file containing references. It supports downloading the main papers as well as their references, and it can handle interruptions by saving the current session.
### Parameters
- `-i`, `--ris`: The path to the RIS file containing the references of the papers to be downloaded.
- `-o`, `--out`: The directory where the downloaded papers will be saved.
- `-r`, `--ref`: A flag to enable the download of referenced papers in addition to the main papers.
- `-g`, `--gui`: A flag to enable the graphical user interface for the web browser.
- `-u`, `--undetected`: A flag to use an undetected Chrome driver for the browser.
- `-l`, `--log`: A flag to enable verbose logging of the `mbapy` library.

### Behavior
The command performs the following actions:
1. Parses command-line arguments to configure the utility.
2. Sets up logging and error handling.
3. Retrieves available SCI-HUB URLs and selects one to use for downloading papers.
4. Parses the RIS file to extract paper information.
5. Initializes a `Record` object to keep track of downloaded papers and their references.
6. Sets up a web browser for downloading, with options for a graphical or headless mode.
7. Downloads the main papers and, if enabled, their references from SCI-HUB.
8. Handles exceptions by attempting to save the current session and logging the error.
9. Uses a progress bar to display the download status.
10. Allows for manual interruption of the process with the option to save the session.

### Notes
- The script requires the `requests`, `tqdm`, `wget`, and `mbapy` libraries.
- It uses a `Record` class to manage and store information about downloaded papers and their references.
- The `Browser` class from `mbapy.web` is used to interact with web pages for downloading papers.
- The script includes exception handling to manage unexpected interruptions and attempts to save the progress.

### Example
```
mbapy-cli scihub_selenium --ris "path/to/references.ris" --out "downloaded/papers" --ref --gui
```

### Additional Information
- The script sets environment variables to control the behavior of the `mbapy` library and Pygame.
- It uses multithreading to handle web server launch and record saving concurrently.
- The `handle_exception` function is used as a custom exception handler to manage exceptions and save the progress when they occur.
- The `main` function is the entry point of the script and orchestrates the paper download process.

Please replace `"path/to/references.ris"` with the actual path to your RIS file and `"downloaded/papers"` with your desired output directory when using the command. Ensure that you have the necessary permissions to write to the output directory and that the RIS file is formatted correctly.