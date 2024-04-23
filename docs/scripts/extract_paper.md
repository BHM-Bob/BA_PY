<!--
 * @Date: 2024-04-22 19:54:44
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-04-22 20:18:46
 * @Description: 
-->
*`Kimi` generated*.

## extract_paper

### Introduction
This Python script is designed to extract text content from PDF documents and format it into a structured JSON file. It uses various libraries such as `argparse`, `glob`, `os`, and `tqdm` for argument parsing, file path expansion, environment variable setup, and progress indication, respectively. The script also utilizes the `mbapy` library for PDF processing and JSON serialization.

### Parameters
- `-i`, `--input`: The directory path containing the PDF files to be processed.
- `-o`, `--output`: The output file name for the JSON file that will store the extracted data. Defaults to `_mbapy_extract_paper.json`.
- `-b`, `--backend`: The backend library to use for PDF conversion. Defaults to `pdfminer`.
- `-l`, `--log`: A flag to enable logging. Defaults to `False`.

### Behavior
1. The script sets environment variables to control the behavior of certain libraries.
2. It parses command line arguments to get the input directory, output file name, backend for PDF conversion, and logging flag.
3. It finds all PDF files in the specified input directory.
4. For each PDF file, the script attempts to extract bookmarks and convert the PDF to plain text.
5. The extracted text and bookmarks are formatted into a structured data format.
6. The structured data for all PDF files are compiled into a JSON object.
7. The JSON object is saved to a file with the specified output name in the input directory.
8. The script provides a progress bar to indicate the completion status.
9. It allows for an early stop by the user by pressing the letter `e`.

### Notes
- The script uses the `glob` module to find all PDF files in the given directory.
- The `mbapy` library is expected to provide functions such as `get_section_bookmarks`, `convert_pdf_to_txt`, and `format_paper_from_txt`.
- The script includes error handling to skip over PDF files that cannot be parsed.
- Logging is configurable and can be enabled by the user through the `--log` flag.
- The `Configs.err_warning_level` is set to a high value to suppress warnings if logging is not enabled.

### Examples
To run the script on a directory of PDFs and save the extracted data to a file named `extracted_papers.json`:
```bash
mbapy-cli extract_paper -i . -o extracted_papers.json
```

To run the script with logging enabled and using a specific backend for PDF conversion:
```bash
mbapy-cli extract_paper -i . -l -b pdfminer.six
```