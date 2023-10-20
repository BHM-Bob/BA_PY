# mbapy.sci_utils.paper_parse

### _flatten_pdf_bookmarks -> List[Any]
This function takes a variable number of bookmark lists and returns a flattened list of all bookmarks.

#### Params
- `*bookmarks (List[Any])`: A variable number of bookmark lists.

#### Returns
- `List[Any]`: A flattened list of all bookmarks.

#### Notes
None

#### Example
```python
bookmarks = [
    ['Chapter 1', 'Section 1.1', 'Section 1.2'],
    ['Chapter 2', 'Section 2.1', 'Section 2.2'],
    ['Chapter 3', 'Section 3.1', 'Section 3.2']
]
flattened_bookmarks = _flatten_pdf_bookmarks(*bookmarks)
print(flattened_bookmarks)
# Output: ['Chapter 1', 'Section 1.1', 'Section 1.2', 'Chapter 2', 'Section 2.1', 'Section 2.2', 'Chapter 3', 'Section 3.1', 'Section 3.2']
```

### has_sci_bookmarks
Checks if a PDF document has bookmarks for scientific sections.

#### Params
- `pdf_obj`: The PDF object(Being opened!). Defaults to None.
- `pdf_path (str)`: The path to the PDF document. Defaults to None.
- `section_names (list[str])`: A list of section names to check for bookmarks. Defaults to an empty list.

#### Returns
- `list[str]` or `bool`: list of section names if the PDF has bookmarks, False otherwise.

#### Notes
None

#### Example
```python
pdf_path = 'path/to/pdf/document.pdf'
section_names = ['Abstract', 'Introduction', 'Materials', 'Methods', 'Results', 'Discussion', 'References']
result = has_sci_bookmarks(pdf_path, section_names)
print(result)
# Output: ['Abstract', 'Introduction', 'Materials', 'Methods', 'Results', 'Discussion', 'References']
```

### get_sci_bookmarks_from_pdf -> List[str]
Returns a list of section names from a scientific PDF.

#### Params
- pdf_path (str): The path to the PDF file. Default is None.
- pdf_obj: The PDF object. Default is None.
- section_names (List[str]): A list of section names to search for. If None, all sections include 'Abstract', 'Introduction', 'Materials', 'Methods', 'Results', 'Conclusions, 'Discussion', 'References' will be searched.

#### Returns
- List[str]: A list of section names found in the PDF.

#### Notes
None

#### Example
```python
pdf_path = 'example.pdf'
section_names = ['Abstract', 'Introduction', 'Methods']
result = get_sci_bookmarks_from_pdf(pdf_path, section_names)
print(result)
# Output: ['Abstract', 'Introduction', 'Methods']
```

### get_section_bookmarks -> List[str]
Returns a list of titles of bookmark sections in a PDF.

#### Params
- pdf_path (str): The path to the PDF file. Defaults to None.
- pdf_obj: The PDF object(Being opened!). Defaults to None.

#### Returns
- list: A list of titles of bookmark sections in the PDF. Returns None if there are no bookmark sections or if the PDF file does not exist.

#### Notes
None

#### Example
```python
pdf_path = 'example.pdf'
result = get_section_bookmarks(pdf_path)
print(result)
# Output: ['Abstract', 'Introduction', 'Methods']
```

### get_english_part_of_bookmarks -> list[str]
Retrieves the English part of the given list of bookmarks.

#### Params
- bookmarks (list[str]): A list of bookmarks.

#### Returns
- list[str]: A list containing only the English part of the bookmarks.

#### Notes
None

#### Example
```python
bookmarks = ['Introduction', '方法', 'Results', 'Discussion']
result = get_english_part_of_bookmarks(bookmarks)
print(result)
# Output: ['Introduction', 'Results', 'Discussion']
```

### get_section_from_paper -> str
Extracts a section of a science paper by key.

#### Params
- paper (str): A science paper.
- key (str): One of the sections in the paper. Can be 'Title', 'Authors', 'Abstract', 'Keywords', 'Introduction', 'Materials & Methods', 'Results', 'Discussion', 'References'
- keys (List[str], optional): A list of keys to extract. Defaults to ['Title', 'Authors', 'Abstract', 'Keywords', 'Introduction', 'Materials & Methods', 'Results', 'Discussion', 'References'].

#### Returns
- str: The extracted section of the paper.

#### Notes
- The function searches for the specified key in the paper and returns the corresponding section.
- If the key is not found, an error message is returned.

#### Example
```python
paper = "This is a science paper. It has a Title, Authors, Abstract, Introduction, and References."
section = get_section_from_paper(paper, "Abstract")
print(section)
# Output: "This is the abstract of the paper."
```

### format_paper_from_txt -> dict
Formats a science paper from plain text into a structured dictionary.

#### Params
- content (str): The content of the paper in plain text.
- struct (List[str], optional): A list of section names in the desired structure. Defaults to ['Title', 'Authors', 'Abstract', 'Keywords', 'Introduction', 'Materials & Methods', 'Results', 'Discussion', 'References'].

#### Returns
- dict: A dictionary containing the formatted sections of the paper.

#### Notes
- The function uses the `get_section_from_paper` function to extract each section from the plain text content.
- The sections are stored in a dictionary with the section names as keys.

#### Example
```python
content = "This is a science paper. It has a Title, Authors, Abstract, Introduction, and References."
paper = format_paper_from_txt(content)
print(paper['Abstract'])
# Output: "This is the abstract of the paper."
```