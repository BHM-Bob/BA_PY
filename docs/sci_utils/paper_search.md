# mbapy.sci_utils.paper_search

### search_IF -> dict
This function performs a search on the 'ablesci' website using the provided query. It retrieves the first result, extracts relevant information from the page, and returns a dictionary containing the full name, abbreviated name, subjects, impact factor, WOS query, and CAS query.

#### Params
- query (str): The search query to be used.
- proxies (Optional[dict]): A dictionary of proxy settings to be used for the request. Defaults to None.

#### Returns
- dict: A dictionary containing the following information:
    - full_name (str): The full name of the journal.
    - abbr_name (str): The abbreviated name of the journal.
    - subjects (list[str]): A list of subjects covered by the journal.
    - impact_factor (str): The impact factor of the journal.
    - wos_q (str): The WOS query associated with the journal.
    - cas_q (str): The CAS query associated with the journal.

#### Notes
If no results are found, an error message is returned.

#### Example
```python
result = search_IF('journal name')
print(result)
```

### search_by_baidu -> list
This function is used to search for articles on Baidu Scholar using a given query.

Note: If use_browser_first is False, you may open baidu xueshu website before this function being called.

#### Params
- query (str): The search query.
- limit (int, optional): The maximum number of articles to retrieve. Defaults to 1.
- use_browser_first (bool, optional): Whether to use selenium to get website for the first time, since it make more successful.

#### Returns
- list: A list of dictionaries containing information about the articles found.
    Each dictionary has the following keys:
    - title (str): The title of the article.
    - abstract (str): The abstract of the article.
    - keyword (list): A list of keywords associated with the article.
    - doi (str): The DOI (Digital Object Identifier) of the article.
    - journal (str): The name of the journal associated with the article.

#### Example
```python
results = search_by_baidu('article title')
print(results)
```

### search_by_pubmed -> list
Searches for articles in the PubMed database based on a query.

#### Params
- query (str): The search query.
- email (str, optional): The email address used for Entrez.email. Defaults to None.
- limit (int, optional): The maximum number of results to return. Defaults to 1.

#### Returns
list: A list of dictionaries containing information about the articles found.
Each dictionary has the following keys:
- title (str): The title of the article.
- abstract (str): The abstract of the article.
- doi (str): The DOI (Digital Object Identifier) of the article.
- journal (str): The name of the journal associated with the article.

#### Raises
- parameter_checker.ParameterLengthError: If the length of the parameters is not equal to check_parameters_len.

#### Notes
- This function uses the Bio.Entrez module from the Biopython library to search for articles in the PubMed database.
- The Entrez.email parameter is used to specify the email address to be used for the search. This is required by the PubMed API.
- The limit parameter specifies the maximum number of results to return. By default, only the first result is returned.

#### Example
```python
results = search_by_pubmed('cancer')
for result in results:
    print(result['title'])
    print(result['abstract'])
    print(result['doi'])
    print(result['journal'])
    print('---')
```


### search_by_wos -> list
Perform a search on the Web of Science (WOS) database using a given query string.

#### Params
- query (str): The query string to search for in the WOS database.
- limit (int, optional): The maximum number of search results to return. Defaults to 1.
- browser (str, optional): The browser to use for performing the search. Defaults to 'Chrome'.
- browser_driver_path (str, optional): The path to the browser driver executable. Defaults to None.
- proxies (optional): The proxies to use for the browser. Defaults to None.

#### Returns
list: A list of dictionaries containing information about the search results. Each dictionary contains the following keys:
- title (str): The title of the paper.
- authors (str): The authors of the paper.
- doi (str): The DOI of the paper.
- date (str): The indexed date of the paper.
- article_type (str): The type of the article (e.g., journal article, conference paper, etc.).
- abstract (str): The abstract of the paper.
- keywords (str): The keywords associated with the paper.
- journal (str): The journal in which the paper was published.
- journal_subjects (str): The subjects covered by the journal.
- impact_factor (str): The impact factor of the journal.

#### Notes
- This function performs a search on the Web of Science (WOS) database using a given query string.
- The limit parameter specifies the maximum number of search results to return. By default, only the first result is returned.
- The browser parameter specifies the browser to use for performing the search. By default, Chrome is used.
- The browser_driver_path parameter specifies the path to the browser driver executable. By default, it is set to None, which means the system PATH will be used to locate the driver.
- The proxies parameter specifies the proxies to use for the browser. By default, no proxies are used.

#### Example
```python
results = search_by_wos('cancer')
for result in results:
    print(result['title'])
    print(result['authors'])
    print(result['doi'])
    print(result['date'])
    print(result['article_type'])
    print(result['abstract'])
    print(result['keywords'])
    print(result['journal'])
    print(result['journal_subjects'])
    print(result['impact_factor'])
    print('---')
```

### search -> list
General description:
This function searches for a given query using a specified search engine and returns the results.

Params:
- query (str): The query string to search for.
- limit (int): The maximum number of results to return. Default is 1.
- search_engine (str): The search engine to use. Default is 'baidu xueshu'. Allowed values are 'baidu xueshu', 'pubmed', 'wos'. If not recognized, returns None.
- email (str): The email address to use for searching on Pubmed. Required if search_engine is 'pubmed'.
- browser (str): The browser to use for searching on WOS. Default is 'Chrome'.
- browser_driver_path (str): The path to the browser driver executable. Default is web.CHROME_DRIVER_PATH.
- use_browser (bool): Whether to use a browser for searching. Default is False.
- use_undetected (bool): Whether to use undetected browser for searching. Default is True.

Returns:
- The search results as a list of dictionaries. Each dictionary contains the following keys: 'title', 'abstract', 'keyword', and 'doi'.

Notes:
- The search_engine parameter must be one of the allowed values. If it is not recognized, the function will return None.
- The email parameter is required if the search_engine is 'pubmed'. If it is not provided, the function will return None.
- The browser parameter is only used when the search_engine is 'wos'. If it is not provided, the function will use the default browser 'Chrome'.
- The browser_driver_path parameter is only used when the search_engine is 'wos' and use_browser is True. If it is not provided, the function will use the default driver path web.CHROME_DRIVER_PATH.
- The use_browser parameter is only used when the search_engine is 'wos'. If it is set to True, the function will use a browser for searching. If it is set to False, the function will not use a browser and return None.
- The use_undetected parameter is only used when the search_engine is 'baidu xueshu'. If it is set to True, the function will use an undetected browser for searching. If it is set to False, the function will not use an undetected browser and return None.

Example:
```
search('python', limit=5, search_engine='pubmed', email='example@example.com')
```


### get_reference_by_doi -> dict
General description:
This function retrieves the reference information for a given DOI.

Params:
- doi (str): The DOI to retrieve the reference for.

Returns:
- A dictionary containing the reference information. The dictionary has the following keys: 'key', 'doi-asserted-by', 'first-page', 'DOI', 'article-title' (may not exist), 'volume', 'author', 'year', 'journal-title'.

Notes:
- This function uses the crossref_commons.retrieval.get_publication_as_json() function to retrieve the reference information.
- If the retrieval fails, the function will return None.

Example:
```
get_reference_by_doi('10.1053/j.gastro.2005.11.061')
```