# mbapy.sci_utils.paper_download

### \_get_available_scihub_urls -> list
**Finds available scihub urls via http://tool.yovisun.com/scihub/**

#### Params
- proxies (dict, optional): A dictionary of proxies to be used for the HTTP request. Defaults to None.

#### Returns
- list: A list of available SciHub URLs.

#### Notes
- This function sends a GET request to 'http://tool.yovisun.com/scihub/' to find available SciHub URLs.

#### Example
```python
available_urls = _get_available_scihub_urls(proxies={'http': 'http://proxy.example.com:8080'})
```

### \_update_available_scihub_urls -> list
**Updates the list of available SciHub URLs.**

#### Returns
- list: The updated list of available SciHub URLs.

#### Notes
- This function updates the global variable `available_scihub_urls` by calling the `_get_available_scihub_urls()` function if `available_scihub_urls` is None. Otherwise, it returns the current value of `available_scihub_urls`.

#### Example
```python
updated_urls = _update_available_scihub_urls()
```

### get_clean_doi -> str
**Cleans and returns the DOI string.**

#### Params
- doi (str): The DOI string to be cleaned.

#### Returns
- str: The cleaned DOI string.

#### Example
```python
cleaned_doi = get_clean_doi('10.123/abc.456')
```

### \_get_scihub_valid_download_link -> str
**Generates the valid Sci-Hub download link for the given input link.**

#### Params
- link (str): The input link for which the valid Sci-Hub download link needs to be generated.

#### Returns
- str: The valid Sci-Hub download link.

#### Example
```python
valid_link = _get_scihub_valid_download_link('http://example.com/paper.pdf')
```

### \_download_from_scihub_webpage -> dict
**Downloads a file from the SciHub webpage.**

#### Params
- webpage (requests.Response): The response object of the SciHub webpage.
- proxies (dict, optional): The proxies to be used for the request. Defaults to None.
- try_times (int): The number of times to attempt the download.

#### Returns
- dict: A dictionary containing the title, DOI, and the response object of the download request.

#### Notes
- This function attempts to download a file from the SciHub webpage and returns the result as a dictionary.

#### Example
```python
download_result = _download_from_scihub_webpage(webpage, proxies={'http': 'http://proxy.example.com:8080'}, try_times=3)
```

### download_from_scihub_by_doi -> dict or None
**Downloads a file from the Sci-Hub database using the DOI.**

#### Params
- doi (str): The DOI of the file to download.
- proxies (dict): A dictionary of proxies to use for the request.
- try_times (int): The number of times to attempt the download.

#### Returns
- dict or None: A dictionary containing the title, DOI, and the response object of the download request. If meets error, returns None.

#### Raises
- Exception: If the DOI does not exist or if there is an error fetching the file from Sci-Hub.

#### Example
```python
download_result = download_from_scihub_by_doi('10.123/abc.456', proxies={'http': 'http://proxy.example.com:8080'}, try_times=3)
```

### download_from_scihub_by_title -> dict or None
**Downloads a document from Scihub by title.**

#### Params
- title (str): The title of the document to be downloaded.
- proxies (dict, optional): A dictionary of proxies to be used for the HTTP request.
- try_times (int): The number of times to attempt the download.

#### Returns
- dict or None: A dictionary containing the title, DOI, and the response object of the download request. If meets error, returns None.

#### Raises
- Exception: If the document with the given title does not exist on Scihub.

#### Example
```python
download_result = download_from_scihub_by_title('Sample Paper Title', proxies={'http': 'http://proxy.example.com:8080'}, try_times=3)
```

### download_by_scihub -> dict or None
**Download a paper from Sci-Hub using its DOI.**

#### Params
- dir (str): The directory where the downloaded file will be saved.
- doi (str): The DOI (Digital Object Identifier) of the paper.
- title (str): The title of the document to be downloaded.
- file_full_name (str, optional): The name of the downloaded file, include the file extension(.pdf). Defaults to None.
- use_title_as_name (bool, optional): Whether to use the paper's title as the file name. Defaults to True.
- valid_path_chr (str, optional): The character used to replace invalid characters in the file name. Defaults to '_'.
- try_times (int): The number of times to attempt the download.

#### Returns
- dict or None: If successful, returns a dictionary containing information about the downloaded paper. If unsuccessful, returns None.

#### Notes
- If doi is None and can't get doi from sci-hub webpage, doi will be set as %Y%m%d.%H%M%S.

#### Example
```python
download_result = download_by_scihub(dir='path/to/save', doi='10.123/abc.456', try_times=3)
```