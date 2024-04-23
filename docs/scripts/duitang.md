
*`Kimi` generated*.

## duitang
### Introduction
This Python script is designed to download images from the Duitang website based on a user-provided search query. It uses the `argparse` library to parse command line arguments, `BeautifulSoup` for parsing HTML content, `selenium` for browser automation, and `tqdm` for progress display.

### Parameters
- `-q`, `--query`: The search query string to use on Duitang. If the query contains spaces, it should be enclosed in quotation marks. Defaults to an empty string.
- `-n`, `--num-pages`: The number of pages from which to download images. Defaults to 5.
- `-o`, `--output`: The directory path where the downloaded images will be saved. Defaults to the current directory.
- `-t`, `--type`: The file type of the image files to download. Can be either 'jpg' or 'avif'. Defaults to 'jpg'.
- `-g`, `-gui`: A flag to indicate whether to use Selenium's GUI mode for downloading images. Defaults to False (headless mode).
- `-u`, `--undetected-chromedriver`: A flag to indicate whether to use an undetected ChromeDriver with Selenium. This can help avoid detection by certain websites. Defaults to False.

### Behavior
1. The script sets up command line argument parsing and defines the expected arguments.
2. It initializes a Selenium `Browser` object with the specified options, including whether to run in headless mode and whether to use an undetected ChromeDriver.
3. The script uses the Duitang base URL to start a search based on the provided query.
4. It scrolls through each page, extracts image URLs, and initiates downloads for each image.
5. The script uses a coroutine pool to asynchronously download images, which can improve download speed.
6. It saves a record of downloaded URLs to avoid re-downloading the same images in future runs.

### Notes
- The script requires the `selenium`, `beautifulsoup4`, `tqdm`, and `pillow_heif` (for AVIF support) libraries to be installed.
- The `Configs.web.request_header` is used as the User-Agent string for HTTP requests, which should be defined within the `mbapy` library or the script's configuration.
- The script includes error handling and logging functionality through the `put_err` function from the `mbapy` library.
- The `clean_path` function is used to sanitize the output directory path.
- The `show_args` function displays the provided arguments in a formatted manner.

### Examples
To run the script with a search query and download 10 pages of images in AVIF format to a specified directory:
```bash
mbapy-cli duitang -q "search term" -n 10 -o "/path/to/output" -t avif
```

To run the script in GUI mode with headless disabled:
```bash
mbapy-cli duitang -q "search term" -g
```

To run the script with an undetected ChromeDriver:
```bash
mbapy-cli duitang -q "search term" -u
```