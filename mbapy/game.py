'''
Date: 2023-10-02 22:53:27
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-10-03 23:00:40
Description: 
'''

import collections
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
        - from_dict(): convert a dictionary to an object.
        - to_json(): update and save a dict from to_dict() to a json file.
        - from_json(): update and load a dict from a json file.
        - update(): update the un-jsonable attributes of the object after obj.__init__() and from_dict.
    """
    def __init__(self) -> None:
        self.__psd_type__ = type(self).__name__
    def update(self):
        """
        有些不能序列化保存的info需要以另一些info来生成, 所以在调用from_dict,
        重新生成class, 加载可序列化info后, 调用update来用加载的info更新那部分不能保存而直接默认生成的info.
        
        Notes:
            - from_dict内部调用update并不会以update返回值赋值给obj
        """
        return self
    def add_attr(self, key, value):
        setattr(self, key, value)
    def del_attr(self, key):
        delattr(self, key)
    def to_dict(self, force_update: bool = True, to_json = False):
        """
        Converts the object to a dictionary representation,
        if obj contains class which is subclass of BaseInfo, it will be converted by to_dict.

        Args:
            - force_update (bool, optional): If True, forces the update of the object's dictionary representation. Defaults to True.
            - to_json (bool, optional): If True, converts the dictionary representation to JSON format. Defaults to False.
- 
        Returns:
            dict: The dictionary representation of the object, including its attributes and their values.
            
        Notes:
            - __psd_type__ will be added to the dictionary to indicate the type of the object, and will work when reconverting.
        """
        if self.__dict__ or force_update:
            _dict_ = {}
            for k, v in vars(self).items():
                # v就是可to_dict的对象, 直接用to_dict方法转换
                if hasattr(v, 'to_dict'):
                    v = v.to_dict(force_update, to_json)
                # 亦或v是字典类，并且含有可to_dict的对象。将可to_dict的对象用to_dict方法转换后合并，转换为字典
                elif isinstance(v, collections.abc.Mapping) and any(hasattr(i, 'to_dict') for i in v.values()):
                    v = {k_i:v_i.to_dict(force_update, to_json) for \
                            k_i, v_i in v.items() if hasattr(v_i, 'to_dict')}
                # 亦或v是列表类，并且含有可to_dict的对象。将可to_dict的对象用to_dict方法转换后合并，转换为列表
                elif isinstance(v, collections.abc.Sequence) and any(hasattr(i, 'to_dict') for i in v):
                    v = [v_i.to_dict(force_update, to_json) for v_i in v]
                # 如果需要转换为json，并且v是可to_dict的对象，将v追加到__dict__
                if not to_json or mf.is_jsonable(v):
                    _dict_[k] = v
        _dict_['__psd_type__'] = type(self).__name__
        return _dict_
    def from_dict(self, dict_: dict):
        """
        Deserialize the object from a dictionary representation.

        Parameters:
            dict_ (dict): The dictionary representation of the object.

        Returns:
            self: The deserialized object.
        """
        for k, v in dict_.items():
            if isinstance(v, Dict) and '__psd_type__' in v:
                exec(f'globals()["new_obj"] = {v["__psd_type__"]}()', globals())
                new_obj: BaseInfo = globals()['new_obj']
                new_obj = new_obj.from_dict(v)
                new_obj.update()
                setattr(self, k, new_obj)
                v = new_obj
            self.__dict__[k] = v
        return self
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
        """
        Parses a JSON file located at the specified `path` and updates the current object with the data from the JSON file.

        Parameters:
            path (str): The path to the JSON file.

        Returns:
            self: The updated object.
        """
        self.from_dict(mf.read_json(path, invalidPathReturn={}))
        return self