'''
Date: 2023-10-02 22:53:27
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-10-12 21:46:19
Description: 
'''

import collections
import inspect
import os
from typing import Callable, Dict, List, Tuple

import numpy as np

if __name__ == '__main__':
    # dev mode import
    import mbapy.base as mb
    import mbapy.file as mf
else:
    from . import base as mb
    from . import file as mf
    
Size = collections.namedtuple('Size', ['w', 'h'])
Rect = collections.namedtuple('Rect', ['x', 'y', 'w', 'h'])
Sur = collections.namedtuple('Sur', ['name', 'sur', 'rect'])


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
                is_jsonable = mf.is_jsonable(v)
                # 如果不需要转为json或者v可json, 直接纳入
                if not to_json or is_jsonable:
                    _dict_[k] = v
                # 如果需要转换为json, 并且v总体上不是可直接json的对象, 那么假定v有以下三种情况, 分类处理
                elif to_json and not is_jsonable:
                    # v是BaseInfo类或继承自BaseInfo的类, 直接用to_dict方法转换
                    if issubclass(type(v), BaseInfo):
                        _dict_[k] = v.to_dict(force_update, to_json)
                    # 亦或v是字典类, 并且含有可json或继承自BaseInfo的对象. 将可json的直接合并, 继承自BaseInfo的对象用to_dict方法转换后合并, 转换为字典, 其余不管.
                    elif isinstance(v, collections.abc.Mapping):
                        _v = {}
                        for k_i, v_i in v.items():
                            if mf.is_jsonable(v_i):
                                _v[k_i] = v_i
                            elif issubclass(type(v_i), BaseInfo):
                                _v[k_i] = v_i.to_dict(force_update, to_json)
                        _dict_[k] = _v
                    # 亦或v是列表类, 并且含有可json或继承自BaseInfo的对象. 将可json的直接合并, 继承自BaseInfo的对象用to_dict方法转换后合并, 转换为字典, 其余不管.
                    elif isinstance(v, collections.abc.Sequence):
                        _v = []
                        for v_i in v:
                            if mf.is_jsonable(v_i):
                                _v.append(v_i)
                            elif issubclass(type(v_i), BaseInfo):
                                _v.append(v_i.to_dict(force_update, to_json))
                        _dict_[k] = _v
        _dict_['__psd_type__'] = type(self).__name__
        return _dict_
    def from_dict(self, dict_: dict, global_vars:Dict = None):
        """
        Deserialize the object from a dictionary representation.

        Parameters:
            - dict_ (dict): The dictionary representation of the object.
            - global_vars (dict): The globals() to init __psd_type__, if None, use inspect.currentframe automatically

        Returns:
            self: The deserialized object.
        """
        if inspect.currentframe().f_back.f_code.co_name == 'from_json':
            from_json_frame = inspect.currentframe().f_back
            outer_caller_frame = from_json_frame.f_back
        else:
            outer_caller_frame = inspect.currentframe().f_back
        global_vars = mb.get_default_for_None(global_vars,
                                              outer_caller_frame.f_globals)
        for k, v in dict_.items():
            if isinstance(v, Dict) and '__psd_type__' in v:
                new_obj: BaseInfo = eval(f'{v["__psd_type__"]}()', global_vars)
                new_obj = new_obj.from_dict(v, global_vars)
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
            _dict_ = self.to_dict(True, True)
            mf.save_json(path, _dict_)
            return _dict_
        except:
            return self.__dict__
    def from_json(self, path: str, global_vars:Dict = None):
        """
        Parses a JSON file located at the specified `path` and updates the current object with the data from the JSON file.

        Parameters:
            - path (str): The path to the JSON file.
            - global_vars (dict): The globals() to init __psd_type__, if None, use inspect.currentframe automatically

        Returns:
            self: The updated object.
        """
        global_vars = mb.get_default_for_None(global_vars,
                                              inspect.currentframe().f_back.f_globals)
        self.from_dict(mf.read_json(path, invalidPathReturn={}),
                       global_vars = global_vars)
        return self
    
class ColorSur:
    def __init__(self, size: Size, sum_dots: int = 4, max_delta_x = 5, max_delta_y = 5) -> None:
        self.size = size
        self.sum_dots = sum_dots
        self.timestamp = 100 * np.random.rand(sum_dots, 1, 1)
        self.delta_ts = 0.1 * np.random.rand(sum_dots, 1, 1)
        self.delta_x = np.random.randint(-max_delta_x, max_delta_x, size = [sum_dots, 1, 1])
        self.delta_y = np.random.randint(-max_delta_y, max_delta_y, size = [sum_dots, 1, 1])
        self.dots_x = np.random.randint(1, self.size.w, [sum_dots, 1, 1])
        self.dots_y = np.random.randint(1, self.size.h, [sum_dots, 1, 1])
        self.dots_col = np.random.rand(3, sum_dots, 1, 1)
        self.map_x, self.map_y = np.meshgrid(np.arange(self.size.w), np.arange(self.size.h))
        self.map_x = self.map_x.reshape([1, self.size.h, self.size.w]).repeat(sum_dots, axis=0)
        self.map_y = self.map_y.reshape([1, self.size.h, self.size.w]).repeat(sum_dots, axis=0)
        self.mat = np.zeros([self.size.h, self.size.w, 3], dtype = np.float32)
    def _update_dots_pos(self, dot_x_or_y: np.ndarray, dleta_x_or_y: np.ndarray, x_or_y_boundry: int):
        dot_x_or_y += dleta_x_or_y
        # err is 1, else is 0
        mask = (dot_x_or_y <= 0) | (dot_x_or_y >= x_or_y_boundry)
        # err is -1, else is 1
        mask = 1 - 2 * mask.astype(np.int32)
        # change move position if dot meats boundry
        dleta_x_or_y *= mask
        # return the updated dots and delta
        return dot_x_or_y, dleta_x_or_y
    def _update_dots_col(self):
        # update timestamp
        self.timestamp += self.delta_ts
        # update color:~[0,1]
        self.dots_col[0,:,:,:] = (0.5 * np.sin(self.timestamp) + 0.5).reshape(self.sum_dots,1,1)
        self.dots_col[1,:,:,:] = (0.5 * np.cos(self.timestamp) + 0.5).reshape(self.sum_dots,1,1)
        self.dots_col[2,:,:,:] = (0.5 * np.sin(0.8 * self.timestamp) + 0.5).reshape(self.sum_dots,1,1)
    def _calc_once(self):
        # calcu 1 / (dx**2 + dy**2)
        dx = self.map_x - self.dots_x
        dy = self.map_y - self.dots_y
        dx += (dx == 0).astype(np.int32) # if there has 0, add 1
        dy += (dy == 0).astype(np.int32) # if there has 0, add 1
        lengths = 1 / (np.power(dx, 2) + np.power(dy, 2)) # [sum_dots, H, W]
        ratio = lengths / lengths.sum(axis = 0).reshape(1, self.size.h, self.size.w) # [sum_dots, H, W]
        #mat[:,:,0]:[H,W,]  ratio:[num, H, W]  mat[0,:,:,:]:[num,1,1]
        self.mat[:,:,0] = (ratio * self.dots_col[0,:,:,:]).sum(axis = 0)
        self.mat[:,:,1] = (ratio * self.dots_col[1,:,:,:]).sum(axis = 0)
        self.mat[:,:,2] = (ratio * self.dots_col[2,:,:,:]).sum(axis = 0)
        return self.mat
    def update(self):
        """
        calcu once and return a uint8 numpy.NDArray with shape [h, w, 3]
        """
        self._calc_once()
        self.dots_x, self.delta_x = self._update_dots_pos(self.dots_x, self.delta_x, self.size.w)
        self.dots_y, self.delta_y = self._update_dots_pos(self.dots_y, self.delta_y, self.size.h)
        self._update_dots_col()
        return (self.mat * 255.).astype(np.uint8)
        
if __name__ == '__main__':
    import time

    import cv2
    startTime = time.time()
    FPSTime = time.time()
    sur = ColorSur(Size(660,340))
    while cv2.waitKey(1) != ord('q') :
        print(f"{1 / (time.time() - FPSTime):.1f} fps")
        FPSTime = time.time()
        cv2.imshow('animation', sur.update())

    cv2.destroyAllWindows()