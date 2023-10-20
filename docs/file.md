# mbapy.file

This module provides utility functions for file operations, including reading and writing files, working with different file formats, and handling file paths.  

## Functions
### get_paths_with_extension -> List[str]
Returns a list of file paths within a given folder that have a specified extension.

#### Params
- folder_path (str): The path of the folder to search for files.
- file_extensions (List[str]): A list of file extensions to filter the search by.

#### Returns
- List[str]: A list of file paths that match the specified file extensions.

#### Notes
None

#### Example
```python
folder_path = '/path/to/folder'
file_extensions = ['.txt', '.csv']
file_paths = get_paths_with_extension(folder_path, file_extensions)
print(file_paths)
```

### extract_files_from_dir
Move all files in subdirectories to the root directory and add the subdirectory name as a prefix to the file name.

#### Params
- root (str): The root directory path.
- file_extensions (list[str]): specific file types string (without '.'), if None, means all types.
- extract_sub_dir (bool, optional): Whether to recursively extract files from subdirectories. If set to False, only files in the immediate subdirectories will be extracted. Defaults to True.
- join_str (str): string for link prefix and the file name.

#### Returns
None

#### Notes
None

#### Example
```python
root = '/path/to/root'
file_extensions = ['.txt', '.csv']
extract_files_from_dir(root, file_extensions, extract_sub_dir=True, join_str='_')
```

### replace_invalid_path_chr -> str
Replaces any invalid characters in a given path with a specified valid character.

#### Params
- path (str): The path string to be checked for invalid characters.
- valid_chrs (str, optional): The valid characters that will replace any invalid characters in the path. Defaults to '_'.

#### Returns
- str: The path string with all invalid characters replaced by the valid character.

#### Notes
None

#### Example
```python
path = '/path/with/invalid?characters'
valid_path = replace_invalid_path_chr(path, valid_chrs='_')
print(valid_path)
```

### get_valid_file_path -> str
Returns a valid file path by replacing any invalid characters in the given path with a specified valid character and truncating the path to a specified length.

#### Params
- path (str): The path string to be checked for invalid characters.
- valid_chrs (str, optional): The valid characters that will replace any invalid characters in the path. Defaults to '_'.
- valid_len (int, optional): The maximum length of the valid file path. Defaults to 250.

#### Returns
- str: The valid file path.

#### Notes
None

#### Example
```python
path = '/path/with/invalid?characters'
valid_path = get_valid_file_path(path, valid_chrs='_', valid_len=100)
print(valid_path)
```

### opts_file
A function that reads or writes data to a file based on the provided options.

#### Params
- path (str): The path to the file.
- mode (str, optional): The mode in which the file should be opened. Defaults to 'r'.
- encoding (str, optional): The encoding of the file. Defaults to 'utf-8'.
- way (str, optional): The way in which the data should be read or written. Defaults to 'lines'.
- data (Any, optional): The data to be written to the file. Only applicable in write mode. Defaults to None.

#### Returns
- list or str or dict or None: The data read from the file, or None if the file was opened in write mode and no data was provided.

#### Notes
None

#### Example
```python
path = '/path/to/file.txt'
data = ['line 1', 'line 2', 'line 3']
read_data = opts_file(path, mode='w', data=data)
print(read_data)
```

### read_bits -> bytes
Reads a file in binary mode and returns the content as bytes.

#### Params
- path (str): The path to the file.

#### Returns
- bytes: The content of the file as bytes.

#### Notes
None

#### Example
```python
path = '/path/to/file.bin'
content = read_bits(path)
print(content)
```

### read_text -> str or List[str]
Reads a file in text mode and returns the content as a string or a list of lines.

#### Params
- path (str): The path to the file.
- decode (str, optional): The encoding of the file. Defaults to 'utf-8'.
- way (str, optional): The way in which the data should be read. Defaults to 'lines'.

#### Returns
- str or List[str]: The content of the file as a string or a list of lines.

#### Notes
None

#### Example
```python
path = '/path/to/file.txt'
content = read_text(path, decode='utf-8', way='lines')
print(content)
```

### detect_byte_coding(bits:bytes) -> str

Detects the byte coding of a given byte array.  

Parameters:  
- bits (bytes): The byte array to be analyzed.  

Returns:  
- str: The detected byte coding of the input sequence.  

Example:  
```python
detect_byte_coding(b'\xe4\xb8\xad\xe6\x96\x87')
```

### decode_bits_to_str(bits:bytes) -> str

Decodes a bytes object to a string using either GB2312 or utf-8 encoding.  

Parameters:  
- bits (bytes): The bytes object to decode.  

Returns:  
- str: The decoded string.  

Example:  
```python
decode_bits_to_str(b'\xe4\xb8\xad\xe6\x96\x87')
```

### save_json(path:str, obj, encoding:str = 'utf-8', forceUpdate = True) -> None

Saves an object as a JSON file at the specified path.  

Parameters:  
- path (str): The path where the JSON file will be saved.  
- obj: The object to be saved as JSON.  
- encoding (str): The encoding of the JSON file. Default is 'utf-8'.  
- forceUpdate (bool): Determines whether to overwrite an existing file at the specified path. Default is True.  

Returns:  
- None

Example:  
```python
data = {'name': 'John', 'age': 30}
save_json('data.json', data)
```

### read_json(path:str, encoding:str = 'utf-8', invalidPathReturn = None) -> Union[dict, Any]

Reads a JSON file from the given path and returns the parsed JSON data.  

Parameters:  
- path (str): The path to the JSON file.  
- encoding (str, optional): The encoding of the file. Defaults to 'utf-8'.  
- invalidPathReturn (any, optional): The value to return if the path is invalid. Defaults to None.  

Returns:  
- dict: The parsed JSON data.  
- invalidPathReturn (any): The value passed as `invalidPathReturn` if the path is invalid.  

Example:  
```python
read_json('data.json')
```

### save_excel(path:str, obj:List[List[str]], columns:List[str], encoding:str = 'utf-8', forceUpdate = True) -> bool

Save a list of lists as an Excel file.  

Parameters:  
- path (str): The path where the Excel file will be saved.  
- obj (List[List[str]]): The list of lists to be saved as an Excel file.  
- columns (List[str]): The column names for the Excel file.  
- encoding (str, optional): The encoding of the Excel file. Defaults to 'utf-8'.  
- forceUpdate (bool, optional): If True, the file will be saved even if it already exists. Defaults to True.  

Returns:  
- bool: True if the file was successfully saved, False otherwise.  

Example:  
```python
data = [['Name', 'Age'], ['John', '30'], ['Jane', '25']]
columns = ['Name', 'Age']
save_excel('data.xlsx', data, columns)
```

### read_excel(path:str, sheet_name:str = None, ignore_head:bool = True, ignore_first_col:bool = True, invalid_path_return = None) -> Union[pandas.DataFrame, Any]

Reads an Excel file and returns a pandas DataFrame.  

Parameters:  
- path (str): The path to the Excel file.  
- sheet_name (str, optional): The name of the sheet to read. Defaults to None.  
- ignore_head (bool, optional): Whether to ignore the first row (header) of the sheet. Defaults to True.  
- ignore_first_col (bool, optional): Whether to ignore the first column of the sheet. Defaults to True.  
- invalid_path_return (Any, optional): The value to return if the path is invalid. Defaults to None.  

Returns:  
- pandas.DataFrame: The DataFrame containing the data from the Excel file.  
- invalid_path_return (Any): The value specified if the path is invalid.  

Example:  
```python
read_excel('data.xlsx')
```

### write_sheets(path:str, sheets:Dict[str, pd.DataFrame]) -> None

Write multiple sheets to an Excel file.  

Parameters:  
- path (str): The path to the Excel file.  
- sheets (Dict[str, pd.DataFrame]): A dictionary mapping sheet names to dataframes.  

Returns:  
- None

Example:  
```python
data1 = pd.DataFrame({'Name': ['John', 'Jane'], 'Age': [30, 25]})
data2 = pd.DataFrame({'City': ['New York', 'Los Angeles'], 'Country': ['USA', 'USA']})
sheets = {'Sheet1': data1, 'Sheet2': data2}
write_sheets('data.xlsx', sheets)
```

### update_excel(path:str, sheets:Dict[str, pd.DataFrame] = None) -> Union[Dict[str, pd.DataFrame], None]

Updates an Excel file with the given path by adding or modifying sheets.  

Parameters:  
- path (str): The path of the Excel file.  
- sheets (Dict[str, pd.DataFrame], optional): A dictionary of sheets to add or modify. 
    The keys are sheet names and the values are pandas DataFrame objects. 
    Defaults to None.  

Returns:  
- Union[Dict[str, pd.DataFrame], None]: If the Excel file exists and sheets is None, 
    returns a dictionary containing all the sheets in the Excel file. 
    Otherwise, returns None.  

Raises:  
- None

Example:  
```python
data1 = pd.DataFrame({'Name': ['John', 'Jane'], 'Age': [30, 25]})
data2 = pd.DataFrame({'City': ['New York', 'Los Angeles'], 'Country': ['USA', 'USA']})
sheets = {'Sheet1': data1, 'Sheet2': data2}
update_excel('data.xlsx', sheets)
```

### convert_pdf_to_txt(path: str, backend = 'PyPDF2') -> str

Convert a PDF file to a text file.  

Parameters:  
- path: The path to the PDF file.  
- backend: The backend library to use for PDF conversion. Defaults to 'PyPDF2'.  

Returns:  
- The extracted text from the PDF file as a string.  

Raises:  
- NotImplementedError: If the specified backend is not supported.  

Example:  
```python
convert_pdf_to_txt('document.pdf')
```
### is_jsonable -> bool
This function checks if the given data is JSON serializable.

#### Params
- data (any): The data to be checked.

#### Returns
- bool: True if the data is JSON serializable, False otherwise.

#### Notes
- The function checks if the data is of type str, int, float, bool, or None. These types are JSON serializable.
- If the data is a mapping (e.g. dict), the function recursively checks if all values in the mapping are JSON serializable.
- If the data is a sequence (e.g. list, tuple), the function recursively checks if all items in the sequence are JSON serializable.
- If the data is of any other type, it is not JSON serializable.

#### Example
```python
data1 = "Hello"
print(is_jsonable(data1))  # Output: True

data2 = {"name": "John", "age": 30}
print(is_jsonable(data2))  # Output: True

data3 = [1, 2, 3, {"name": "John"}]
print(is_jsonable(data3))  # Output: True

data4 = {"name": "John", "age": datetime.datetime.now()}
print(is_jsonable(data4))  # Output: False
```

### convert_pdf_to_txt -> str
Convert a PDF file to a text file.

#### Params
- path: The path to the PDF file.
- backend: The backend library to use for PDF conversion. 
    - 'PyPDF2' is the default.
    - 'pdfminer'.

#### Returns
The extracted text from the PDF file as a string.

#### Raises
- NotImplementedError: If the specified backend is not supported.

#### Example
```python
text = convert_pdf_to_txt('path/to/pdf/file.pdf')
print(text)
```

```python
text = convert_pdf_to_txt('path/to/pdf/file.pdf', backend='pdfminer')
print(text)
```