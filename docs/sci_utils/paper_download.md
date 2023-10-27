# mbapy.sci_utils.paper_download

### get_clean_doi -> str
This function takes a DOI (Digital Object Identifier) as input and returns a cleaned version of the DOI.

#### Params
- doi (str): The DOI to be cleaned.

#### Returns
- str: The cleaned DOI.

#### Notes
- The function uses regular expressions to extract the DOI from the input string.
- If a valid DOI is found, it is returned. Otherwise, an empty string is returned.

#### Example
```python
doi = "10.1234/abcd.1234"
clean_doi = get_clean_doi(doi)
print(clean_doi)
# Output: "10.1234/abcd.1234"
```

### _get_available_scihub_urls -> List[str]
This function finds available SciHub URLs by scraping the website http://tool.yovisun.com/scihub/.

#### Returns
- List[str]: A list of available SciHub URLs.

#### Notes
- The function sends a GET request to the website and parses the HTML response to extract the available URLs.
- The function checks the status of each URL and only includes the ones that are labeled as "可用" (available) in the list of available URLs.

#### Example
```python
available_urls = _get_available_scihub_urls()
print(available_urls)
# Output: ["https://sci-hub.se", "https://sci-hub.st"]
```

### _update_available_scihub_urls -> List[str]
This function updates the list of available SciHub URLs.

#### Returns
- List[str]: The updated list of available SciHub URLs.

#### Notes
- The function checks if the global variable `available_scihub_urls` is None.
- If it is None, the function calls `_get_available_scihub_urls()` to update the list of available URLs and assigns it to `available_scihub_urls`.
- If `available_scihub_urls` is not None, it returns the current value of `available_scihub_urls`.

#### Example
```python
available_urls = _update_available_scihub_urls()
print(available_urls)
# Output: ["https://sci-hub.se", "https://sci-hub.st"]
```

### _download_from_scihub_webpage -> Dict[str, Union[str, requests.Response]]
This function downloads a file from the SciHub webpage.

#### Params
- webpage (requests.Response): The response object of the SciHub webpage.
- proxies (dict, optional): The proxies to be used for the request. Defaults to None.

#### Returns
- Dict[str, Union[str, requests.Response]]: A dictionary containing the title, DOI, and the response object of the download request.

#### Notes
- The function parses the HTML response of the SciHub webpage to extract the title, DOI, and download link.
- It then sends a GET request to the download link and returns the response object.

#### Example
```python
webpage = requests.get("https://sci-hub.se")
download_result = _download_from_scihub_webpage(webpage)
print(download_result)
# Output: {'title': 'Example Paper', 'doi': '10.1234/abcd.1234', 'res': <Response [200]>}
```

### download_from_scihub_by_doi -> Dict[str, Union[str, requests.Response]] or None
This function downloads a file from the Sci-Hub database using the DOI.

#### Params
- doi (str): The DOI of the file to download.
- proxies (dict): A dictionary of proxies to use for the request.

#### Returns
- Dict[str, Union[str, requests.Response]] or None: A dictionary containing the title, DOI, and the response object of the download request. If an error occurs, None is returned.

#### Raises
- Exception: If the DOI does not exist or if there is an error fetching the file from Sci-Hub.

#### Example
```python
doi = "10.1234/abcd.1234"
download_result = download_from_scihub_by_doi(doi)
print(download_result)
# Output: {'title': 'Example Paper', 'doi': '10.1234/abcd.1234', 'res': <Response [200]>}
```

### download_from_scihub_by_title -> Dict[str, Union[str, requests.Response]] or None
This function downloads a document from Scihub by title.

#### Params
- title (str): The title of the document to be downloaded.
- proxies (dict, optional): A dictionary of proxies to be used for the HTTP request.

#### Returns
- Dict[str, Union[str, requests.Response]] or None: A dictionary containing the title, DOI, and the response object of the download request. If an error occurs, None is returned.

#### Raises
- Exception: If the document with the given title does not exist on Scihub.

#### Example
```python
title = "Example Paper"
download_result = download_from_scihub_by_title(title)
print(download_result)
# Output: {'title': 'Example Paper', 'doi': '10.1234/abcd.1234', 'res': <Response [200]>}
```

### download_by_scihub -> Dict[str, Union[str, None]] or None
This function downloads a paper from Sci-Hub using its DOI or title.

#### Params
- dir (str): The directory where the downloaded file will be saved.
- doi (str): The DOI (Digital Object Identifier) of the paper.
- title (str): The title of the paper.
- file_full_name (str, optional): The name of the downloaded file, including the file extension (.pdf). Defaults to None.
- use_title_as_name (bool, optional): Whether to use the paper's title as the file name. Defaults to True.
- valid_path_chr (str, optional): The character used to replace invalid characters in the file name. Defaults to '_'.

#### Returns
- Dict[str, Union[str, None]] or None: If successful, returns a dictionary containing information about the downloaded paper. If unsuccessful, returns None.

#### Notes
- The function first checks if the specified directory exists. If not, it creates the directory.
- It then checks whether the DOI or title is specified. If neither is specified, it returns an error.
- The function downloads the paper from Sci-Hub using the DOI or title and saves it in the specified directory.
- If the file_full_name is specified, it uses that as the file name. Otherwise, it uses the paper's title (or DOI if use_title_as_name is False) as the file name.
- The function replaces invalid characters in the file name with the valid_path_chr character.
- The function returns a dictionary containing the file name, file path, title, DOI, and the response object of the download request.

#### Example
```python
dir = "/path/to/directory"
doi = "10.1234/abcd.1234"
download_result = download_by_scihub(dir, doi=doi)
print(download_result)
# Output: {'file_name': 'Example_Paper.pdf', 'file_path': '/path/to/directory/Example_Paper.pdf', 'title': 'Example Paper', 'doi': '10.1234/abcd.1234', 'res': <Response [200]>}
```