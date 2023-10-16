<!--
 * @Date: 2023-10-16 23:52:52
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2023-10-17 00:13:36
 * @Description: 
-->
# mbapy.game
This module provides utility functions for dataclass reading/writing from/to files and some game GUI.

## Functions

### transfer_bytes_to_base64 -> str
This function takes a bytes object and converts it to a base64 encoded string. It also has an optional parameter to compress the data using gzip before encoding.

#### Params
- data (bytes): The bytes object to be encoded.
- use_gzip (bool): Optional. If True, the data will be compressed using gzip before encoding. Default is True.

#### Returns
- str: The base64 encoded string.

#### Notes
- This function uses the gzip module to compress the data if the use_gzip parameter is set to True.
- The resulting base64 encoded string is decoded using the 'utf-8' encoding.

#### Example
```python
data = b'Hello, World!'
encoded_data = transfer_bytes_to_base64(data)
print(encoded_data)
```

Output:
```
'SGVsbG8sIFdvcmxkIQ=='
```

### transfer_base64_to_bytes -> bytes
This function takes a base64 encoded string and converts it to bytes. It also has an optional parameter to decompress the data using gzip.

#### Params
- data (str): The base64 encoded string to be converted to bytes.
- use_gzip (bool): Optional parameter to indicate whether to decompress the data using gzip. Default is True.

#### Returns
- data (bytes): The converted bytes.

#### Notes
- This function assumes that the input data is a valid base64 encoded string.
- If use_gzip is set to True, the function will attempt to decompress the data using gzip.

#### Example
```python
import base64

data = "SGVsbG8gd29ybGQh"
result = transfer_base64_to_bytes(data)
print(result)
# Output: b'Hello world!'

compressed_data = "H4sIAAAAAAAA/8vPBgBHw6WAgAAAA=="
result = transfer_base64_to_bytes(compressed_data, use_gzip=True)
print(result)
# Output: b'Hello world!'
```


## Classes

### BaseInfo
BaseInfo is a base class for common information storage.

#### Methods:
- to_dict(): convert the atributes but not the methods to a dictionary.
- from_dict(): convert a dictionary to an object.
- to_json(): update and save a dict from to_dict() to a json file.
- from_json(): update and load a dict from a json file.
- update(): update the un-jsonable attributes of the object after obj.__init__() and from_dict.
    
#### Example:
```python
class MyInfo(BaseInfo):
    def __init__(self, **kwargs):
        super().__init__()
        self.arr = np.array([1, 2, 3])
o1 = MyInfo()
d = o1.to_dict()
o2 = MyInfo().from_dict(d)
```

### ColorSur
This class represents a color surface. It generates a color map by calculating the color values at each pixel based on the positions and colors of a set of dots.

#### Attributes
- `size` (Size): The size of the color surface.
- `sum_dots` (int): The number of dots on the color surface.
- `timestamp` (numpy.ndarray): The timestamp of each dot.
- `delta_ts` (numpy.ndarray): The change in timestamp of each dot.
- `delta_x` (numpy.ndarray): The change in x-coordinate of each dot.
- `delta_y` (numpy.ndarray): The change in y-coordinate of each dot.
- `dots_x` (numpy.ndarray): The x-coordinates of the dots.
- `dots_y` (numpy.ndarray): The y-coordinates of the dots.
- `dots_col` (numpy.ndarray): The colors of the dots.
- `map_x` (numpy.ndarray): The x-coordinates of the color map.
- `map_y` (numpy.ndarray): The y-coordinates of the color map.
- `mat` (numpy.ndarray): The color map.

#### Methods
- `__init__(self, size: Size, sum_dots: int = 4, max_delta_x = 5, max_delta_y = 5) -> None`: Initializes the ColorSur object with the given size, number of dots, maximum delta x and maximum delta y.
- `_update_dots_pos(self, dot_x_or_y: np.ndarray, dleta_x_or_y: np.ndarray, x_or_y_boundry: int)`: Updates the positions of the dots based on their current positions, delta x or y, and the boundary.
- `_update_dots_col(self)`: Updates the colors of the dots based on the timestamp.
- `_calc_once(self)`: Calculates the color map based on the positions and colors of the dots.
- `update(self)`: Updates the color surface by calculating the color map and updating the positions and colors of the dots. Returns the color map as a uint8 numpy array.

#### Example
```python
size = Size(100, 100)
color_sur = ColorSur(size)
color_map = color_sur.update()
```