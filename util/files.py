import ntpath

import pandas as pd

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


class DataFile():
    def __init__(self, path: str) -> None:
        """Construtor for DataFile class

        Args:
            path (str): Path to the file
        """
        
        self._data = read_file(path)
        
        
    def __getitem__(self, key) -> pd.DataFrame:
        """Get a row or a column from the file

        Args:
            key (_type_): If key is an integer, the corresponding row will be returned.
            If key is a string, the corresponding column will b returned.

        Returns:
            pd.DataFrame: A row/column corresponding to key
        """
        res = None
        
        if(type(key) == str):
            return self._data[key]
        elif(type(key) == int):
            return self._data.iloc[key]
        
        return res
    
    def __str__(self) -> str:
        """Stringlyfy the file

        Returns:
            str: The string representation of the file
        """
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        return str(self._data)
    
    def apply_hot_encoding(self, column: str, drop_first=False, dtype='Int8'):
        """Apply one-hot encoding on a specific column

        Args:
            column (str): Name of the column\n
            drop_first (boolean, optional): Whether to get k-1 dummies out of k categorical levels by removing the first level. The default value is False\n
            dtype (str, optional): Data type for encoding. The default data type is integer'.
            NOTE: After performing one-hot encoding, name of the column may be modified.
        """
        self._data = pd.get_dummies(self._data, columns=[column], drop_first=drop_first, dtype=dtype)
        
    def remove_column(self, column: str):
        """Remove a column by name

        Args:
            column (str): Name of the column
        """
        self._data = self._data.drop([column], axis=1)
        
    def remove_row(self, row: int):
        """Remove a row by index

        Args:
            row (int): Index of a row
        """
        self._data = self._data.drop([row])
        