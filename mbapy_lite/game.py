'''
Date: 2023-10-02 22:53:27
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-05-02 18:53:28
Description: 
'''

import base64
import collections
import gzip
import inspect
import os
import pickle
import traceback
from typing import Callable, Dict, List, Tuple

import numpy as np

if __name__ == '__main__':
    # dev mode import
    import mbapy_lite.base as mb
    import mbapy_lite.file as mf
else:
    from . import base as mb
    from . import file as mf


def transfer_bytes_to_base64(data: bytes, use_gzip: bool = True) -> str:
    """
    Convert a byte data to base64 string.

    Args:
        data (bytes): The byte data to be converted.
        use_gzip (bool, optional): Indicates whether to compress the data using gzip. Defaults to True.

    Returns:
        str: The base64-encoded string.
    """
    if use_gzip: 
        data = gzip.compress(data)
    return base64.b64encode(data).decode('utf-8')


def transfer_base64_to_bytes(data: str, use_gzip: bool = True) -> bytes:
    """
    Convert a base64 encoded string to bytes.

    Parameters:
        data (str): The base64 encoded string to convert.
        use_gzip (bool, optional): Whether to use gzip decompression. Defaults to True.

    Returns:
        bytes: The converted bytes.
    """
    data = base64.b64decode(data.encode('utf-8'))
    if use_gzip:
        data = gzip.decompress(data)
    return data

class BaseInfo:
    """
    BaseInfo is a base class for common information storage.
    
    Methods:
        - to_dict(): convert the atributes but not the methods to a dictionary.
        - from_dict(): convert a dictionary to an object.
        - to_json(): update and save a dict from to_dict() to a json file.
        - from_json(): update and load a dict from a json file.
        - update(): update the un-jsonable attributes of the object after obj.__init__() and from_dict.
    
    Example:
    ```python
    class MyInfo(BaseInfo):
        def __init__(self, **kwargs):
            super().__init__()
            self.arr = np.array([1, 2, 3])
    o1 = MyInfo()
    d = o1.to_dict()
    o2 = MyInfo().from_dict(d)
    ```
    """
    @mb.autoparse
    def __init__(self, *args, **kwargs) -> None:
        self.__psd_type__ = type(self).__name__
        self.__psd_id__ = id(self) # 通过id来避免在to_dict时出现循环解析
        
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
        
    def to_dict(self, force_update: bool = True, to_json: bool = True,
                use_gzip: bool = True, id_pool: Dict[int, str] = {}):
        """
        Converts the object to a dictionary representation,
        if obj contains class which is subclass of BaseInfo, it will be converted by to_dict.
        
        Args:
            - force_update (bool, optional): If True, forces the update of the object's dictionary representation. Defaults to True.
            - to_json (bool, optional): If True, converts the dictionary representation to JSON format. Defaults to True.
            - use_gzip (bool, optional): If true, when comes with bytes(such as numpy.ndarray), use gzip to compress.

        Returns:
            dict: The dictionary representation of the object, including its attributes and their values.
            
        Notes:
            - __psd_type__ will be added to the dictionary to indicate the type of the object, and will work when reconverting.
            - If CAN NOT transfer, mark it as __psd_type__NON_TRANSFERABLE__, when reconverting, 
                it will be ignored and remains to default value after __init__().
        """
        def _check_all_default_value(obj):
            """
            检查obj.__init__的参数是否都有默认值
            TODO: 最大目标是实现不强制要求所有参数都有默认值, 但目前还不知道怎么实现
            """
            sig = inspect.signature(obj.__init__)
            for k, v in sig.parameters.items():
                if v.default == inspect.Parameter.empty:
                    return False
            return True
        def _transfer_np_ndarray(v: np.ndarray, to_json: bool, use_gzip: bool):
            """将numpy.ndarray转为PSD dict格式(array转为list, 或压缩过的bytes再转为base64)"""
            if to_json:
                return {
                    '__psd_type__NP_NDARRAY__': type(v).__name__,
                    'use_gzip': use_gzip,
                    'shape': v.shape,
                    'dtype': str(v.dtype) if str(v.dtype) != 'bool' else 'bool_', # https://github.com/numpy/numpy/issues/22021
                    'data':  transfer_bytes_to_base64(v.reshape(-1).tobytes(), use_gzip)
                }
            else:
                return v.tolist()
        def _transfer_np_scallar(v: np.ScalarType, to_json: bool, use_gzip: bool):
            """将numpy.ScalarType转为PSD dict格式(取item)"""
            if to_json:
                return {
                    '__psd_type__NP_SCALAR__': type(v).__name__,
                    'dtype': str(v.dtype) if str(v.dtype) != 'bool' else 'bool_', # https://github.com/numpy/numpy/issues/22021
                    'data':  v.item()
                }
            else:
                return v
        def _transfer_pickle(v: bytes, to_json: bool, use_gzip: bool):
            """
            将其它不能转换的类型用pickle序列化(压缩过的bytes再转为base64)
            TODO: 最大目标是实现不强制要求所有参数都有默认值, 但目前还不知道怎么实现
            """
            if to_json:
                data = transfer_bytes_to_base64(pickle.dumps(v), use_gzip)
                return {
                    '__psd_type__PICKLE__': data.decode('utf-8')
                }
            else:
                return v
        def _check_case_transfer(v, to_json, use_gzip):
            """
            检查v的各种受支持的类型并做转换
            - 可json的对象: 直接返回;
            - 继承自BaseInfo的对象: 调用to_dict;
            - numpy.ndarray: 转为bytes后(压缩)再转为base64;
            - numpy.ScalarType: 直接转为item;
            - pygame.SurfaceType: 转为bytes后(压缩)再转为base64;
            - pygame.Rect: 转为psd格式的dict;
            - Mapping或Sequence: 递归调用_check_transfer;
            - 其它不能转换的类型, 且不是None: 记为__psd_type__NON_TRANSFERABLE__;
            """
            # 如果不需要转为json或者v可json, 直接纳入
            if mf.is_jsonable(v):
                return v
            # v是BaseInfo类或继承自BaseInfo的类, 直接用to_dict方法转换
            elif issubclass(type(v), BaseInfo):
                return v.to_dict(force_update, to_json, use_gzip, id_pool)
            # v是numpy.ndarray, 调用_transfer_np_ndarray方法转换
            elif isinstance(v, np.ndarray):
                return _transfer_np_ndarray(v, to_json, use_gzip)
            # v是numpy的标量, 调用_transfer_np_ndarray方法转换
            elif any(isinstance(v, ty) for ty in [np.int_, np.float_, np.bool_]):
                return _transfer_np_scallar(v, to_json, use_gzip)
            # 亦或v是字典类, 并且含有可json或继承自BaseInfo的对象(存在嵌套则递归). 将可json的直接合并, 继承自BaseInfo的对象用to_dict方法转换后合并, 转换为字典, 其余不管.
            elif isinstance(v, collections.abc.Mapping):
                _v = {}
                for idx, (k_i, v_i) in enumerate(v.items()):
                    # 由于josn中的key必须是字符串, 所以需要检查一下
                    if isinstance(k_i, str):
                        v_i = _check_case_transfer(v_i, to_json, use_gzip)
                        if v_i is not None:
                            _v[k_i] = v_i
                    # 如果key不是字符串，用_check_case_transfer检查并转换，不过不能转换(返回None)，则忽略
                    else:
                        k_i_type = type(k_i).__name__
                        k_i = _check_case_transfer(k_i, to_json, use_gzip)
                        if k_i is not None:
                            v_i = _check_case_transfer(v_i, to_json, use_gzip)
                            _v[f'__psd_type__KEY_VALUE_{idx}__'] = {'key': k_i, 'key_type': k_i_type, 'value': v_i}
                return _v
            # 亦或v是列表类, 并且含有可json或继承自BaseInfo的对象(存在嵌套则递归). 将可json的直接合并, 继承自BaseInfo的对象用to_dict方法转换后合并, 转换为字典, 其余不管.
            elif isinstance(v, collections.abc.Sequence):
                _v = []
                for v_i in v:
                    v_i = _check_case_transfer(v_i, to_json, use_gzip)
                    if v_i is not None:
                        _v.append(v_i)
                return _v
            # 其它不能转换的类型, 且不是None
            elif v is not None:
                return {'__psd_type__NON_TRANSFERABLE__': type(v).__name__}
            # 是None, 则返回None
            return None
        # check id_pool
        if self.__psd_id__ not in id_pool:
            id_pool[self.__psd_id__] = self.__psd_type__
        else:
            return {'__psd_type__ID_LINK__': [self.__psd_id__, self.__psd_type__]}
        # recursively convert all attributes to dictionary
        if self.__dict__ or force_update:
            _dict_ = {}
            for k, v in vars(self).items():
                is_jsonable = mf.is_jsonable(v)
                # 如果不需要转为json或者v可json, 直接纳入
                if not to_json or is_jsonable:
                    _dict_[k] = v
                # 如果需要转换为json, 并且v总体上不是可直接json的对象, 分类处理
                elif to_json and not is_jsonable:
                    _dict_[k] = _check_case_transfer(v, to_json, use_gzip)
        _dict_['__psd_type__'] = type(self).__name__
        return _dict_
    
    def from_dict(self, dict_: dict, global_vars:Dict = None,
                  obj_pool:Dict[int, object] = {}, first_run: bool = True):
        """
        Deserialize the object from a dictionary representation.

        Parameters:
            - dict_ (dict): The dictionary representation of the object.
            - global_vars (dict): The globals() to init __psd_type__, if None, use inspect.currentframe automatically.
            - obj_pool (dict): The object pool to store the objects which have been deserialized.
            - first_run (bool): If True, it means this is the first run of from_dict, and the object pool is empty.

        Returns:
            self: The deserialized object.
        """
        def _check_case_transfer(v):
            # 继承自BaseInfo类的反序列化
            if isinstance(v, Dict) and '__psd_type__' in v:
                obj_type = eval(f'{v["__psd_type__"]}', global_vars)
                try:
                    new_obj: BaseInfo = obj_type()
                except:
                    # accept situations when __init__ has not default value
                    new_obj: BaseInfo = obj_type.__new__(obj_type)
                new_obj = new_obj.from_dict(v, global_vars, obj_pool, False)
                new_obj.update()
                # setattr(self, k, new_obj) # probablly junk code
                v = new_obj
                obj_pool[v.__psd_id__] = v
            # 继承自BaseInfo类的反序列化(通过id_link)
            elif isinstance(v, Dict) and '__psd_type__ID_LINK__' in v:
                if v['__psd_type__ID_LINK__'][0] in obj_pool:
                    v = obj_pool[v['__psd_type__ID_LINK__'][0]]
                else:
                    v = None
            # numpy.ndarray反序列化
            elif isinstance(v, Dict) and '__psd_type__NP_NDARRAY__' in v:
                v['data'] = transfer_base64_to_bytes(v['data'], v['use_gzip'])
                v = np.frombuffer(v['data'], dtype=eval(f'np.{v["dtype"]}')).reshape(v['shape'])
            # numpy.ScalarType反序列化
            elif isinstance(v, Dict) and '__psd_type__NP_SCALAR__' in v:
                v = eval(f'np.{v["dtype"]}({v["data"]})') # '_' for https://github.com/numpy/numpy/issues/22021
            # non_transferable类型
            elif isinstance(v, Dict) and '__psd_type__NON_TRANSFERABLE__' in v:
                pass # 直接返回字典{'__psd_type__NON_TRANSFERABLE__': type(v).__name__}
            # 一般list反序列化
            elif isinstance(v, List):
                v = [_check_case_transfer(v_i) for v_i in v]
            # 一般dict反序列化
            elif isinstance(v, Dict):
                _dict_ = {}
                for k_i, v_i in v.items():
                    if k_i.startswith('__psd_type__KEY_VALUE_'):
                        k_i_type = v_i['key_type']
                        new_k_i, new_v_i = _check_case_transfer(v_i['key']), _check_case_transfer(v_i['value'])
                        if k_i_type == 'tuple':
                            new_k_i = tuple(new_k_i)
                        _dict_[new_k_i] = new_v_i
                    else:
                        _dict_[k_i] = _check_case_transfer(v_i)
                v = _dict_
            return v
        # get global_vars
        if inspect.currentframe().f_back.f_code.co_name == 'from_json':
            from_json_frame = inspect.currentframe().f_back
            outer_caller_frame = from_json_frame.f_back
        else:
            outer_caller_frame = inspect.currentframe().f_back
        global_vars = mb.get_default_for_None(global_vars,
                                              outer_caller_frame.f_globals)
        # add this BaseInfo obj's id to obj_pool
        obj_pool[dict_['__psd_id__']] = self
        # recursively convert all attributes from dictionary
        for k, v in dict_.items():
            if v is not None:
                v = _check_case_transfer(v)
            if not (isinstance(v, Dict) and '__psd_type__NON_TRANSFERABLE__' in v):
                self.__dict__[k] = v
        self.update()
        # recorver each object's id in obj_pool
        if first_run:
            for obj in obj_pool.values():
                obj.__psd_id__ = id(obj)
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
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _dict_ = self.to_dict(True, True)
            mf.save_json(path, _dict_)
            return _dict_
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
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


if __name__ == '__main__':
    pass