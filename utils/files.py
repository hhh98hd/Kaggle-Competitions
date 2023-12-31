import os
import ntpath
import math
from typing import Any
from datetime import datetime

import pandas as pd
import torch
import numpy as np

CSV = 'csv'
JSON = 'json'
EXCEL = 'xls'

def read_file(path: str):
    """Read the file using its path. Currently only CSV, JSON, and EXCEL files are supported.

    Args:
        path (str): Path to the file
        
    Exception:
        TypeError: This error is raised when the file's type is not supported

    Returns:
        file (object): The file
    """
    
    file = None
    file_type = get_file_extension(path)
    
    if(file_type == CSV):
        file = pd.read_csv(path)
    elif(file_type == JSON):
        file = pd.read_json(path)
    elif(file_type == EXCEL):
        file = pd.read_excel(path)
    else:
        raise TypeError('{type} files are not supported!'.format(type=file_type.upper()))

    return file

def export_to_csv(predictions, first_col: str, second_col: str, path='', start_idx=0):
    """Export model's predictions to a CSV file

    Args:
        predictions (list): A list of predictions
        first_col (str): Name of the first column
        second_col (str): Name of the second column
        path (str, optional): Location of the exported file. Defaults to the `util` directory
        start_idx (int, optional): ID of the first prediction. Defaults to 0.
    """
        
    if path == '':
        path = os.path.join(os.path.dirname(__file__), './output_' + datetime.now().strftime("%H%M%S_%d%b%y") + '.csv')
        
    i = start_idx
    file = open(path, 'w')
    
    file.write(first_col + ',' + second_col + '\n')
    for pred in predictions:
        file.write("%d,%d\n" % (i, pred))
        i += 1
    
    file.close()

def get_file_extension(file_path: str):
    """Get the file extension from a path
    Ex: app.exe -> exe; data.json -> json

    Args:
        file_path (str): Path to the file

    Returns:
        type (str): Type of the file
    """
    
    type = ''
    file_name = ntpath.basename(file_path)
    
    pos = file_name.rfind('.')
    type = file_name[pos + 1 :]
    
    return type

def to_tensor(data : pd.DataFrame, dtype=np.float32):
    """Convert a Pandas DataFrame to a PyTorch tensor

        Args:
            dtype: Data type of the tensor

        Returns:
            torch.Tensor: The converted DataFrame
    """
    
    return torch.tensor(data.values.astype(dtype))
    


class DataFile():
    """DataFile is a class for handling and munipulating data files (csv, xls, json, ...).
    """
    
    def __init__(self, path: str) -> None:
        """Construtor for DataFile class

        Args:
            path (str): Path to the file
        """        
        self._data = read_file(path)
        self.shape = self._data.shape
    
    def __getattribute__(self, __name: str) -> Any:
        """Access to an attribute of the class

        Args:
            __name (str): Name of the attribute. If the attribute is "shape", this method will return the shape of its content (pandas DataFrame)

        Returns:
            Any: Attribute value
        """
        if __name == "shape":
            return object.__getattribute__(self._data, 'shape')
        else:
            return super().__getattribute__(__name)
        
    def __getitem__(self, key) -> pd.DataFrame:
        """Get a row or a column from the file

        Args:
            key (_type_): If key is an integer, the corresponding row will be returned.
            If key is a string, the corresponding column will be returned.
            NOTE: A copy will be returned

        Returns:
            pd.DataFrame: A row/column corresponding to key
        """
        res = None
        
        if(type(key) == str):
            return self._data.loc[:, (key)].copy()
        elif(type(key) == int):
            return self._data.iloc[key].copy()
        
        return res
    
    def __setitem__(self, key, value):
        self._data[key] = value
        
    
    def to_string(self, truncate_H=True, truncate_V=True) -> str:
        """Represent the file content using a string

        Args:
            truncate_H (bool, optional): Truncate horizontally. Defaults to True.
            truncate_V (bool, optional): Truncate vertically. Defaults to True.

        Returns:
            str: The string representation of the file
        """
        
        if not truncate_V:
            pd.set_option('display.max_rows', None)
        
        if not truncate_H:
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
        
        return str(self._data)
    
    def apply_hot_encoding(self, column: str, drop_first=False, dtype='Int8'):
        """Apply one-hot encoding on a specific column

        Args:
            column (str): Name of the column\n
            drop_first (boolean, optional): Whether to get k-1 dummies out of k categorical levels by removing the first level. The default value is False\n
            dtype (str, optional): Data type for encoding. The default data type is integer ("Int8").
            NOTE: After performing one-hot encoding, name of the column may be modified.
        """
        self._data = pd.get_dummies(self._data, columns=[column], drop_first=drop_first, dtype=dtype)
        
    def remove_column(self, column: str):
        """Remove a column by its name

        Args:
            column (str): Name of the column
        """
        self._data = self._data.drop([column], axis=1)
        
    def remove_row(self, row: int):
        """Remove a row by its index

        Args:
            row (int): Index of a row
        """
        self._data = self._data.drop([row])
        
    
    def get_all_columns_except(self, column_name: str) -> pd.DataFrame:
        """" Get all columns from the file excep the `column_name` column
        
        Args:
            column_name (str): Name of the column to be obmited

        Returns:
            (DataFrame): All columns except the `column_name` column
        
        """
        
        return self._data.loc[:, self._data.columns != column_name]
        
        
    def get_column_avg(self, column: str) -> float:
        """Get average value of a column. Only applicable to DataFrame of integers or floats

        Args:
            column (str): Name of the column

        Returns:
            (float): Average value of the column
        """
        sum = 0
        cnt = 0
        col = self._data[column]
        
        for i in range(0, col.size):
            if(not math.isnan(col[i])):
                cnt += 1
                sum += col[i]

        return sum / cnt
        
    def to_tensor(self) -> torch.Tensor:
        """Convert the file to a Pytorch tensor\n
        NOTE: Integer or float data only

        Returns:
            torch.Tensor: The converted data
        """
        
        return to_tensor(self._data)
    