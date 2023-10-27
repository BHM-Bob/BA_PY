# mbapy.paper

### parse_ris -> list
Parses a RIS file and returns the contents as a list of dictionaries.

#### Params
- ris_path (str): The path to the RIS file.
- fill_none_doi (str, optional): The DOI value to fill in for missing entries. Defaults to None.

#### Returns
- list: A list of dictionaries containing the parsed contents of the RIS file.

#### Notes
- This function uses the rispy library to parse the RIS file.
- If the `fill_none_doi` parameter is provided, it will fill in the DOI value for any entries that do not have a DOI.

#### Example
```python
ris_path = 'example.ris'
fill_none_doi = '1234567890'
parsed_ris = parse_ris(ris_path, fill_none_doi)
print(parsed_ris)
```
