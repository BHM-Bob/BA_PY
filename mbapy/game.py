'''
Date: 2023-10-02 22:53:27
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-10-02 22:59:23
Description: 
'''

import os
from typing import Callable, Dict, List, Tuple

if __name__ == '__main__':
    # dev mode import
    import mbapy.base as mb
    import mbapy.file as mf
else:
    from . import base as mb
    from . import file as mf

class BaseInfo:
    """
    BaseInfo is a base class for common information storage.
    
    Methods:
        - to_dict(): convert the atributes but not the methods to a dictionary.
        - from_dict(): convert the dictionary to an object.
        - to_json(): update and save __dict__ to a json file.
        - from_json(): update and load __dict__ from a json file.
    """
    def __init__(self) -> None:
        self.__dict__ = {}
    def update(self):
        pass
    def add_attr(self, key, value):
        setattr(self, key, value)
    def del_attr(self, key):
        delattr(self, key)
    def to_dict(self, force_update: bool = True, to_json = False):
        """
        Converts the object to a dictionary representation.

        Args:
            - force_update (bool, optional): Determines whether to force an 
                    update of the object's dictionary representation. 
                    Defaults to True.
            - to_json (bool, optional): Determines whether to convert nested 
                    objects to their JSON representation. Defaults to False.

        Returns:
            dict: A dictionary representation of the object.
            
        Notes:
            - The __psd_type__ attribute is added to the dictionary 
                    representation of the object to specify it is from BaseInfo.
        """
        if self.__dict__ or force_update:
            _dict_, self.__dict__ = vars(self), {}
            for k, v in _dict_.items():
                if not isinstance(v, Callable):
                    if hasattr(v, 'to_dict'):
                        v = v.to_dict(force_update, to_json)
                    if not to_json or mf.is_jsonable(v):
                        self.__dict__[k] = v
        self.__dict__['__psd_type__'] = type(self).__name__
        return self.__dict__
    def from_dict(self, dict_: dict):
        new_obj = None # make vscode happy
        for k, v in dict_.items():
            if isinstance(v, Dict) and '__psd_type__' in v:
                exec(f'globals()["new_obj"] = {v["__psd_type__"]}()', globals())
                new_obj = globals()['new_obj']
                setattr(self, k, new_obj.from_dict(v))
                v = new_obj
            self.__dict__[k] = v
    def to_json(self, path: str):
        """
        Convert the object to a JSON string and save it to a file.

        Parameters:
            - path (str): The path to the file where the JSON string will be saved.
                    If it does not exist, it will be created.

        Returns:
            dict: A dictionary representation of the object or an empty dictionary if something goes wrong.
        """
        try:
            if not mb.check_parameters_path(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            mf.save_json(path, self.to_dict(True, True))
            return self.__dict__
        except:
            return {}
    def from_json(self, path: str):
        self.from_dict(mf.read_json(path, invalidPathReturn={}))