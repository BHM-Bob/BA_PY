# Summary
This module provides a collection of Python classes and functions designed for web content retrieval, data parsing, and asynchronous task handling. It includes functions for setting HTTP request headers, asynchronously fetching web page content, downloading files, and managing an asynchronous task pool based on coroutines. Additionally, the module defines several classes for structuring and processing web page data, as well as for managing sequences of web interactions through a page-based approach.

# Functions

## `install_headers`
### Function Overview
Sets the user-agent header for urllib request.

### Parameters
- `agent` (str, optional): The user-agent string to use for the HTTP request header. Defaults to a standard Chrome browser user-agent string.

### Return Values
None

### Notes
This function is used to prevent receiving web content that might be restricted to certain user-agents.

## `get_web_html_async`
### Function Overview
Asynchronously fetches the HTML content of a given URL.

### Parameters
- `url` (str): The URL from which to fetch the HTML content.
- `headers` (Dict[str, str], optional): A dictionary containing HTTP headers to send with the request.
- `encoding` (str, optional): The encoding to use when decoding the response. Defaults to 'utf-8'.

### Return Values
- A tuple containing a `TaskStatus` value and a string message. The message will be the HTML content if the request is successful, or an error message if it fails.

### Notes
This function uses `aiohttp` to perform an asynchronous HTTP GET request.

## `retrieve_file_async`
### Function Overview
Asynchronously downloads a file from a given URL and saves it to a specified file path.

### Parameters
- `url` (str): The URL of the file to download.
- `file_path` (str): The local file path where the downloaded file will be saved.
- `headers` (Dict[str, str], optional): A dictionary containing HTTP headers to send with the request.

### Return Values
- A tuple containing a `TaskStatus` value and a string message. The message will be the file path if the download is successful, or an error message if it fails.

### Notes
This function creates directories if they do not exist before saving the file.

## `only_sleep`
### Function Overview
A utility function that sleeps for a specified number of seconds, with an optional random delay up to a maximum value.

### Parameters
- `seconds` (float, optional): The base number of seconds to sleep. Defaults to 1.
- `rand` (bool, optional): Whether to include a random delay up to `max`. Defaults to True.
- `max` (float, optional): The maximum additional seconds to randomly wait. Defaults to 5.

### Return Values
- `True`

### Notes
This function is typically used to introduce delays between requests to avoid being rate-limited or blocked by a website.

## `text_fn`
### Function Overview
Extracts text from an element or a list of elements, which could be `etree._Element` objects or strings.

### Parameters
- `x` (Union[str, List[str], etree._Element, List[etree._Element]]): The input element(s) from which to extract text.

### Return Values
- The extracted text, which could be a string or a list of strings.

## `Compose`
### Function Overview
Composes a list of functions into a single function that is the sequential application of the functions in the list.

### Parameters
- `lst` (List[Callable]): A list of functions to compose.

### Return Values
- A new function that is the result of the composition.

# Classes

## `AsyncResult`
### Class Overview
A data class that represents the result of an asynchronous task.

#### Members
- `async_pool` (CoroutinePool): The pool from which the task was executed.
- `name` (str): The name of the task.
- `result` (Any, optional): The result of the task. Defaults to `TaskStatus.NOT_RETURNED`.

### Methods
#### `get`
- Blocks until the result of the asynchronous task is available and returns it.

## `BasePage`
### Class Overview
A base class representing a web page, including methods for parsing and storing data.

#### Members
- `name` (str): The name of the page.
- `xpath` (Union[str, List[str]]): The XPath expression(s) used to extract data from the page.
- `findall_fn` (Callable, optional): An alternative function to extract data using `bs4.find_all`.
- `_async_task_pool` (CoroutinePool): The async task pool for executing async tasks.
- `_headers` (Dict[str, str]): Headers used for web page requests.
- `result` (List[Any]): The parsed result data.
- `father_page` (BasePage, optional): The parent page of the current page.
- `next_pages` (Dict[str, BasePage]]): A dictionary containing the next pages linked from the current page.

### Methods
#### `add_next_page`
- Adds a child page to the current page.

#### `parse`
- Parses data from a given list of results and stores it in `self.result`.

#### `perform`
- Performs the parsing and processing of the page.

## `PagePage`
### Class Overview
A specialized page class that starts by storing a new web page or a list of web pages.

## `UrlIdxPagesPage`
### Class Overview
A specialized page class that starts by parsing pages, storing web pages for further parsing, and getting page URLs from a given base URL.

## `DownloadPage`
### Class Overview
A page class designed to download files from given URLs and store file paths.

## `ItemsPage`
### Class Overview
A page class that parses and stores data from the father page.

## `Actions`
### Class Overview
A class that manages pages and performs actions, providing methods to add, perform, and manage page results.

#### Members
- `pages` (Dict[str, BasePage]): A dictionary of page objects.
- `results` (Dict): A dictionary of all results from the pages.
- `use_thread_listen` (bool): A flag to use a thread for listening to keyboard input.
- `k2a` (List[Tuple[str, Key2Action]]): A list of key to action mappings for controlling the program via keyboard input.
- `_headers` (Dict[str, str]): A dictionary of headers for HTTP requests.
- `_async_task_pool` (CoroutinePool): A coroutine pool for managing asynchronous tasks.

### Methods
#### `get_page`
- Retrieves a page by name from a given set of pages or a father page's next pages.

#### `add_page`
- Adds a page to the `pages` dictionary with optional before and after functions.

#### `del_page`
- Deletes a page from the `pages` dictionary. (Not implemented)

#### `perform`
- Performs all pages to get results, starting necessary threads and async task pools.

#### `close`
- Closes the async task pool.