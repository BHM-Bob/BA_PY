
*`Kimi` generated*.

## cnpia
### Introduction
This Python script is designed to automate the process of searching for and downloading patent information from the China National Intellectual Property Administration (CNIPA) website using a web browser. It utilizes the `argparse` library to accept command line arguments and the `easyocr` library for captcha recognition. The script also includes functionality to handle browser interactions and JSON file operations for storing the results.
### Parameters
- `-q`, `--query`: The search query for the patent.
- `-o`, `--out`: The directory where the output files will be saved.
- `-m`, `--model_path`: The path to the EasyOCR model directory (optional).
- `-l`, `--log`: A flag to enable logging (optional).

### Behavior
1. The script initializes a web browser and sets it up to allow popups.
2. It navigates to the CNIPA advanced search page and inputs the search query provided by the user.
3. The script performs the search and navigates through the results, downloading detailed information for each patent that matches the query.
4. It uses OCR to handle captcha challenges that may appear during the download process.
5. The script saves the downloaded patent information in a JSON file specified by the `-o` argument.

### Notes
- The script requires the `easyocr`, `numpy`, `pyautogui`, and `mbapy` libraries to be installed.
- The `mbapy` library seems to be a custom or third-party library that is not part of the standard Python libraries. It is used for web interaction and logging.
- The script uses XPath expressions to interact with the CNIPA website, which may need to be updated if the website's structure changes.
- The script includes a retry mechanism for handling captcha challenges and network-related issues.

### Examples
To run the script with a search query and specify an output directory:
```bash
mbapy-cli cnipa -q "search term" -o "/path/to/output/directory"
```

To run the script with logging enabled:
```bash
mbapy-cli cnipa -q "search term" -o "/path/to/output/directory" -l
```

To run the script with a custom OCR model path:
```bash
mbapy-cli cnipa -q "search term" -o "/path/to/output/directory" -m "/path/to/easyocr/model"
```