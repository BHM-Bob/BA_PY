# Module Name: mbapy.file

This module provides utility functions for file operations, including reading and writing files, working with different file formats, and handling file paths.  

## Functions

### replace_invalid_path_chr(path:str, valid_chrs:str = '_') -> str

Replaces any invalid characters in a given path with a specified valid character.  

Parameters:  
- path (str): The path string to be checked for invalid characters.  
- valid_chrs (str, optional): The valid characters that will replace any invalid characters in the path. Defaults to '_'.  

Returns:  
- str: The path string with all invalid characters replaced by the valid character.  

Example:  
```python
replace_invalid_path_chr('C:/Users/John?Doe/Documents')
```

### opts_file(path:str, mode:str = 'r', encoding:str = 'utf-8', way:str = 'str', data = None, **kwargs) -> Union[list, str, dict, None]

A function that reads or writes data to a file based on the provided options.  

Parameters:  
- path (str): The path to the file.  
- mode (str, optional): The mode in which the file should be opened. Defaults to 'r'.  
- encoding (str, optional): The encoding of the file. Defaults to 'utf-8'.  
- way (str, optional): The way in which the data should be read or written. Defaults to 'lines'.  
- data (Any, optional): The data to be written to the file. Only applicable in write mode. Defaults to None.  

Returns:  
- list or str or dict or None: The data read from the file, or None if the file was opened in write mode and no data was provided.  

Example:  
```python
opts_file('data.txt', mode='r', way='lines')
```

### read_bits(path:str) -> bytes

Reads a file as a byte array.  

Parameters:  
- path (str): The path to the file.  

Returns:  
- bytes: The byte array read from the file.  

Example:  
```python
read_bits('data.bin')
```

### read_text(path:str, decode:str = 'utf-8', way:str = 'lines') -> Union[list, str]

Reads a text file and returns its content as a list of lines or a single string.  

Parameters:  
- path (str): The path to the text file.  
- decode (str, optional): The encoding to use when decoding the file. Defaults to 'utf-8'.  
- way (str, optional): The way in which the data should be read. Defaults to 'lines'.  

Returns:  
- list or str: The content of the text file.  

Example:  
```python
read_text('data.txt')
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
